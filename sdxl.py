import os
import random
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
import webdataset as wds
from diffusers import (AutoencoderKL, EulerDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline,
                       StableDiffusionXLControlNetPipeline,
                       StableDiffusionXLPipeline, T2IAdapter)
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)

from sdxl_controlnet import SDXLControlNet
from sdxl_controlnet_full import SDXLControlNetFull
from sdxl_controlnet_pre_encoded_controlnet_cond import \
    SDXLControlNetPreEncodedControlnetCond
from sdxl_unet import SDXLUNet
from training_config import training_config
from utils import (load_safetensors_state_dict, maybe_ddp_dtype,
                   maybe_ddp_module)

repo = "stabilityai/stable-diffusion-xl-base-1.0"

device_id = int(os.environ["LOCAL_RANK"])

vae: AutoencoderKL = None

tokenizer_one = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")

text_encoder_one: CLIPTextModel = None

tokenizer_two = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer_2")

text_encoder_two: CLIPTextModelWithProjection = None

unet: SDXLUNet = None

scheduler: EulerDiscreteScheduler = None

adapter: T2IAdapter = None

controlnet: Union[SDXLControlNet, SDXLControlNetFull]

_init_sdxl_called = False


def init_sdxl():
    global _init_sdxl_called, vae, text_encoder_one, text_encoder_two, unet, scheduler, adapter, controlnet

    if _init_sdxl_called:
        raise ValueError("`init_sdxl` called more than once")

    _init_sdxl_called = True

    text_encoder_one = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
    text_encoder_one.to(device=device_id)
    text_encoder_one.requires_grad_(False)
    text_encoder_one.eval()

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(repo, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16)
    text_encoder_two.to(device=device_id)
    text_encoder_two.requires_grad_(False)
    text_encoder_two.eval()

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae.to(device=device_id)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

    if training_config.training == "sdxl_unet":
        if training_config.resume_from is not None:
            unet = SDXLUNet.load(training_config.resume_from)
        else:
            unet = SDXLUNet.load_fp32()
    elif training_config.training == "sdxl_controlnet" and training_config.controlnet_train_base_unet:
        unet = SDXLUNet.load_fp32()

        if training_config.resume_from is not None:
            unet_state_dict = load_safetensors_state_dict(os.path.join(training_config.resume_from, "unet.safetensors"))

            unet_state_dict = {k: v.to(torch.float32) for k, v in unet_state_dict.items()}

            load_sd_results = unet.up_blocks.load_state_dict(unet_state_dict, strict=False)

            if len(load_sd_results.unexpected_keys) > 0:
                raise ValueError(f"error loading state dict: {load_sd_results.unexpected_keys}")
    else:
        unet = SDXLUNet.load_fp16()

    unet.to(device=device_id)
    # TODO - add back
    # unet.enable_gradient_checkpointing()

    if training_config.training == "sdxl_controlnet" and training_config.controlnet_train_base_unet:
        unet.requires_grad_(False)
        unet.eval()

        unet.up_blocks.requires_grad_(True)
        unet.up_blocks.train()

        unet = DDP(unet, device_ids=[device_id], find_unused_parameters=True)
    elif training_config.training == "sdxl_unet":
        unet.requires_grad_(True)
        unet.train()
        unet = DDP(unet, device_ids=[device_id])
    else:
        unet.requires_grad_(False)
        unet.eval()

    if training_config.training == "sdxl_adapter":
        if training_config.resume_from is None:
            adapter = T2IAdapter(
                in_channels=3,
                channels=(320, 640, 1280, 1280),
                num_res_blocks=2,
                downscale_factor=16,
                adapter_type="full_adapter_xl",
            )
        else:
            adapter_repo = os.path.join(training_config.resume_from, "adapter")
            adapter = T2IAdapter.from_pretrained(adapter_repo)

        adapter.to(device=device_id)
        adapter.train()
        adapter.requires_grad_(True)
        adapter.enable_xformers_memory_efficient_attention()
        adapter = DDP(adapter, device_ids=[device_id])

    if training_config.training == "sdxl_controlnet":
        if training_config.controlnet_variant == "default":
            controlnet_cls = SDXLControlNet
        elif training_config.controlnet_variant == "full":
            controlnet_cls = SDXLControlNetFull
        elif training_config.controlnet_variant == "pre_encoded_controlnet_cond":
            controlnet_cls = SDXLControlNetPreEncodedControlnetCond
        else:
            assert False

        if training_config.resume_from is None:
            controlnet = controlnet_cls.from_unet(unet)
        else:
            controlnet_repo = os.path.join(training_config.resume_from, "controlnet")
            controlnet = controlnet_cls.from_pretrained(controlnet_repo)

        controlnet.to(device=device_id)
        controlnet.train()
        controlnet.requires_grad_(True)
        # TODO add back
        # controlnet.enable_gradient_checkpointing()
        # TODO - should be able to remove find_unused_parameters. Comes from pre encoded controlnet
        controlnet = DDP(controlnet, device_ids=[device_id], find_unused_parameters=True)


def get_sdxl_dataset():
    if training_config.dummy_dataset:
        return get_sdxl_dummy_dataset()

    dataset = (
        wds.WebDataset(
            training_config.train_shards,
            resampled=True,
            handler=wds.ignore_and_continue,
        )
        .shuffle(training_config.shuffle_buffer_size)
        .decode("pil", handler=wds.ignore_and_continue)
        .rename(
            image="jpg;png;jpeg;webp",
            text="text;txt;caption",
            metadata="json",
            handler=wds.warn_and_continue,
        )
        .map(make_sample)
    )

    if training_config.training == "sdxl_adapter":
        dataset = dataset.select(adapter_image_is_not_none)

    dataset = dataset.batched(training_config.batch_size, partial=False, collation_fn=default_collate)

    return dataset


def get_sdxl_dummy_dataset():
    image = Image.open("./validation_data/two_birds_on_branch.png").convert("RGB")

    metadata = {"original_height": image.height, "original_width": image.width}

    text = "two birds on a branch"

    sample = {
        "image": image,
        "metadata": metadata,
        "text": text,
    }

    from torch.utils.data.dataset import IterableDataset

    class Dataset(IterableDataset):
        def __iter__(self):
            while True:
                batch = []

                for _ in range(training_config.batch_size):
                    batch.append(make_sample(sample))

                yield default_collate(batch)

    return Dataset()


@torch.no_grad()
def make_sample(d):
    image = d["image"]
    metadata = d["metadata"]

    if random.random() < training_config.proportion_empty_prompts:
        text = ""
    else:
        text = d["text"]

    image = image.convert("RGB")

    resized_image = TF.resize(
        image,
        training_config.resolution,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(
        resized_image,
        output_size=(training_config.resolution, training_config.resolution),
    )

    resized_and_cropped_image = TF.crop(
        resized_image,
        c_top,
        c_left,
        training_config.resolution,
        training_config.resolution,
    )
    resized_and_cropped_image_tensor = TF.to_tensor(resized_and_cropped_image)
    resized_and_cropped_and_normalized_image_tensor = TF.normalize(resized_and_cropped_image_tensor, [0.5], [0.5])

    original_width = int(metadata.get("original_width", 0.0))
    original_height = int(metadata.get("original_height", 0.0))

    time_ids = torch.tensor(
        [
            original_width,
            original_height,
            c_top,
            c_left,
            training_config.resolution,
            training_config.resolution,
        ]
    )

    text_input_ids_one = tokenizer_one(
        text,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]

    text_input_ids_two = tokenizer_two(
        text,
        padding="max_length",
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]

    sample = {
        "time_ids": time_ids,
        "text_input_ids_one": text_input_ids_one,
        "text_input_ids_two": text_input_ids_two,
        "image": resized_and_cropped_and_normalized_image_tensor,
    }

    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type == "mediapipe_pose":
            from image_processing import mediapipe_pose_adapter_image

            adapter_image = mediapipe_pose_adapter_image(resized_and_cropped_image, return_type="vae_scaled_tensor")

            sample["adapter_image"] = adapter_image
        elif training_config.adapter_type == "openpose":
            from image_processing import openpose_adapter_image

            adapter_image = openpose_adapter_image(resized_and_cropped_image, return_type="vae_scaled_tensor")

            sample["adapter_image"] = adapter_image
        else:
            assert False

    if training_config.training == "sdxl_controlnet":
        if training_config.controlnet_type == "canny":
            controlnet_image = make_canny_conditioning(resized_and_cropped_image, return_type="controlnet_scaled_tensor")

            sample["controlnet_image"] = controlnet_image
        elif training_config.controlnet_type == "inpainting":
            from image_processing import make_masked_image

            if training_config.controlnet_variant == "pre_encoded_controlnet_cond":
                controlnet_image, controlnet_image_mask = make_masked_image(resized_and_cropped_image, return_type="vae_scaled_tensor")

                sample["controlnet_image"] = controlnet_image
                sample["controlnet_image_mask"] = controlnet_image_mask
            else:
                controlnet_image, _ = make_masked_image(resized_and_cropped_image, return_type="controlnet_scaled_tensor")

                sample["controlnet_image"] = controlnet_image
        else:
            assert False

    return sample


def adapter_image_is_not_none(sample):
    return sample["adapter_image"] is not None


def sdxl_train_step(batch, global_step):
    with torch.no_grad():
        unet_dtype = maybe_ddp_dtype(unet)

        time_ids = batch["time_ids"].to(device=device_id)

        image = batch["image"].to(device_id, dtype=vae.dtype)
        latents = vae.encode(image).latent_dist.sample().to(dtype=unet_dtype)
        latents = latents * vae.config.scaling_factor

        text_input_ids_one = batch["text_input_ids_one"].to(device_id)
        text_input_ids_two = batch["text_input_ids_two"].to(device_id)

        prompt_embeds, pooled_prompt_embeds_two = text_conditioning(text_input_ids_one, text_input_ids_two)

        prompt_embeds = prompt_embeds.to(dtype=unet_dtype)
        pooled_prompt_embeds_two = pooled_prompt_embeds_two.to(dtype=unet_dtype)

        bsz = latents.shape[0]

        if training_config.training == "sdxl_adapter":
            # Cubic sampling to sample a random timestep for each image
            timesteps = torch.rand((bsz,), device=device_id)
            timesteps = (1 - timesteps**3) * scheduler.config.num_train_timesteps
            timesteps = timesteps.long().to(scheduler.timesteps.dtype)
            timesteps = timesteps.clamp(0, scheduler.config.num_train_timesteps - 1)
        else:
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device_id)

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        sigmas = get_sigmas(timesteps)
        sigmas = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        scaled_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        if training_config.training == "sdxl_controlnet":
            if training_config.controlnet_variant == "pre_encoded_controlnet_cond":
                controlnet_dtype = maybe_ddp_dtype(controlnet)

                controlnet_image = batch["controlnet_image"].to(device_id, dtype=vae.dtype)
                controlnet_image = vae.encode(controlnet_image).latent_dist.sample().to(dtype=controlnet_dtype)
                controlnet_image = controlnet_image * vae.config.scaling_factor

                _, _, controlnet_image_height, controlnet_image_width = controlnet_image.shape
                controlnet_image_mask = batch["controlnet_image_mask"].to(device=device_id)
                controlnet_image_mask = TF.resize(controlnet_image_mask, (controlnet_image_height, controlnet_image_width)).to(dtype=controlnet_dtype)

                controlnet_image = torch.concat((controlnet_image, controlnet_image_mask), dim=1)
            else:
                controlnet_image = batch["controlnet_image"].to(device_id)

        if training_config.training == "sdxl_adapter":
            adapter_image = batch["adapter_image"].to(device_id)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        add_to_down_block_inputs = None
        add_to_output = None

        if training_config.training == "sdxl_adapter":
            down_block_additional_residuals = adapter(adapter_image)

            down_block_additional_residuals = [x * training_config.adapter_conditioning_scale for x in down_block_additional_residuals]

        if training_config.training == "sdxl_controlnet":
            down_block_additional_residuals, mid_block_additional_residual, add_to_down_block_inputs, add_to_output = controlnet(
                scaled_noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "time_ids": time_ids,
                    "text_embeds": pooled_prompt_embeds_two,
                },
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

        model_pred = unet(
            scaled_noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "time_ids": time_ids,
                "text_embeds": pooled_prompt_embeds_two,
            },
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            add_to_down_block_inputs=add_to_down_block_inputs,
            add_to_output=add_to_output,
        ).sample

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        if False and dist.get_rank() == 0:
            log_predicted_images(
                noisy_latents=noisy_latents,
                noise=noise,
                sigmas=sigmas,
                model_pred=model_pred,
                timesteps=timesteps,
                global_step=global_step,
            )

    return loss


@torch.no_grad()
def text_conditioning(text_input_ids_one, text_input_ids_two):
    prompt_embeds_1 = text_encoder_one(
        text_input_ids_one,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_1 = prompt_embeds_1.view(prompt_embeds_1.shape[0], prompt_embeds_1.shape[1], -1)

    prompt_embeds_2 = text_encoder_two(
        text_input_ids_two,
        output_hidden_states=True,
    )

    pooled_prompt_embeds_2 = prompt_embeds_2[0]

    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    prompt_embeds_2 = prompt_embeds_2.view(prompt_embeds_2.shape[0], prompt_embeds_2.shape[1], -1)

    prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)

    return prompt_embeds, pooled_prompt_embeds_2


@torch.no_grad()
def log_predicted_images(noisy_latents, sigmas, noise, model_pred, timesteps, global_step):
    os.makedirs("./output/test_out", exist_ok=True)

    with torch.no_grad():
        latent_predicted_img = noisy_latents[0:1] - sigmas[0:1] * model_pred[0:1]
        latent_predicted_img = latent_predicted_img / vae.config.scaling_factor
        latent_predicted_img = latent_predicted_img.to(torch.float16)
        predicted_img = vae.decode(latent_predicted_img).sample
        predicted_img = ((predicted_img * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        predicted_img = predicted_img[0]
        predicted_img = predicted_img.permute(1, 2, 0)
        predicted_img = predicted_img.cpu().numpy()
        predicted_img = Image.fromarray(predicted_img)
        predicted_img.save(f"./output/test_out/{global_step}-predicted.png")

    with torch.no_grad():
        actual_img = noisy_latents[0:1] - sigmas[0:1] * noise[0:1]
        actual_img = actual_img / vae.config.scaling_factor
        actual_img = actual_img.to(torch.float16)
        actual_img = vae.decode(actual_img).sample
        actual_img = ((actual_img * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        actual_img = actual_img[0]
        actual_img = actual_img.permute(1, 2, 0)
        actual_img = actual_img.cpu().numpy()
        actual_img = Image.fromarray(actual_img)
        actual_img.save(f"./output/test_out/{global_step}-actual.png")

    with open(f"./output/test_out/{global_step}-{timesteps[0].item()}", "w") as f:
        f.write("foo\n")


_validation_images_logged = False


@torch.no_grad()
def sdxl_log_validation(step):
    global _validation_images_logged

    # NOTE - this has to be different from the module level scheduler because
    # the pipeline mutates it.
    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

    if training_config.training == "sdxl_unet":
        unet_ = maybe_ddp_module(unet)
        unet_.eval()

        pipeline = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            unet=unet_,
            scheduler=scheduler,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
        )
    elif training_config.training == "sdxl_adapter":
        adapter_ = maybe_ddp_module(adapter)
        adapter_.eval()

        pipeline = StableDiffusionXLAdapterPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            unet=unet,
            adapter=adapter_,
            scheduler=scheduler,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
        )
    elif training_config.training == "sdxl_controlnet":
        controlnet_ = maybe_ddp_module(controlnet)
        controlnet_.eval()

        if training_config.controlnet_train_base_unet:
            unet_ = maybe_ddp_module(unet)
            unet_.eval()
        else:
            unet_ = unet

        pipeline = StableDiffusionXLControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            unet=unet_,
            controlnet=controlnet_,
            scheduler=scheduler,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
        )
    else:
        assert False

    formatted_validation_images = None

    if training_config.training in ["sdxl_adapter", "sdxl_controlnet"]:
        formatted_validation_images = []
        wandb_validation_images = []

        for validation_image_path in training_config.validation_images:
            validation_image, log_validation_image = get_validation_images(validation_image_path)

            formatted_validation_images.append(validation_image)
            wandb_validation_images.append(wandb.Image(log_validation_image))

        if training_config.controlnet_type == "inpainting" or not _validation_images_logged:
            wandb.log({"validation_conditioning": wandb_validation_images}, step=step)
            _validation_images_logged = True

    generator = torch.Generator().manual_seed(0)

    output_validation_images = []

    for i, validation_prompt in enumerate(training_config.validation_prompts):
        args = {
            "prompt": validation_prompt,
            "generator": generator,
            "height": training_config.resolution,
            "width": training_config.resolution,
        }

        if formatted_validation_images is not None:
            args["image"] = formatted_validation_images[i]

        if training_config.training == "sdxl_adapter":
            args["adapter_conditioning_scale"] = training_config.adapter_conditioning_scale
            args["adapter_conditioning_factor"] = training_config.adapter_conditioning_factor

        with torch.autocast("cuda"):
            image = pipeline(**args).images[0]
            output_validation_images.append(wandb.Image(image, caption=validation_prompt))

    wandb.log({"validation": output_validation_images}, step=step)

    if training_config.training == "sdxl_unet":
        unet_.train()
    elif training_config.training == "sdxl_adapter":
        adapter_.train()
    elif training_config.training == "sdxl_controlnet":
        controlnet_.train()

        if training_config.controlnet_train_base_unet:
            unet_.train()
    else:
        assert False


def get_validation_images(validation_image_path):
    validation_image = Image.open(validation_image_path)
    validation_image = validation_image.convert("RGB")
    validation_image = validation_image.resize((training_config.resolution, training_config.resolution))

    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type == "mediapipe_pose":
            from image_processing import mediapipe_pose_adapter_image

            validation_image = mediapipe_pose_adapter_image(validation_image, return_type="pil")
            log_validation_image = validation_image
        elif training_config.adapter_type == "openpose":
            from image_processing import openpose_adapter_image

            validation_image = openpose_adapter_image(validation_image, return_type="pil")
            log_validation_image = validation_image
        else:
            assert False
    elif training_config.training == "sdxl_controlnet":
        if training_config.controlnet_type == "canny":
            from image_processing import make_canny_conditioning

            validation_image = make_canny_conditioning(validation_image, return_type="pil")
            log_validation_image = validation_image
        elif training_config.controlnet_type == "inpainting":
            from image_processing import make_mask, make_masked_image

            controlnet_image_mask = make_mask(validation_image.height, validation_image.width)
            log_validation_image = Image.fromarray(np.array(validation_image) * (controlnet_image_mask[:, :, None] < 0.5))

            if training_config.controlnet_variant == "pre_encoded_controlnet_cond":
                validation_image, _ = make_masked_image(validation_image, return_type="vae_scaled_tensor", mask=controlnet_image_mask)

                validation_image = validation_image[None, :, :, :]

                validation_image = validation_image.to(device_id, dtype=vae.dtype)

                validation_image = vae.encode(validation_image).latent_dist.sample()
                validation_image = validation_image * vae.config.scaling_factor

                _, _, controlnet_image_height, controlnet_image_width = validation_image.shape
                controlnet_image_mask = TF.resize(torch.from_numpy(controlnet_image_mask)[None, None, :, :], (controlnet_image_height, controlnet_image_width)).to(
                    device=device_id, dtype=maybe_ddp_dtype(controlnet)
                )

                validation_image = torch.concat((validation_image, controlnet_image_mask), dim=1)
            else:
                validation_image, _ = make_masked_image(validation_image, return_type="controlnet_scaled_tensor", mask=controlnet_image_mask)

                validation_image = validation_image[None, :, :, :]

                validation_image = validation_image.to(device_id)
        else:
            assert False
    else:
        assert False

    return validation_image, log_validation_image


def get_sigmas(timesteps, n_dim=4):
    sigmas = scheduler.sigmas.to(device=timesteps.device)
    schedule_timesteps = scheduler.timesteps.to(device=timesteps.device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()

    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma
