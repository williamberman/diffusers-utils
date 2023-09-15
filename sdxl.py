import os
import random
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import webdataset as wds
from diffusers import (AutoencoderKL, ControlNetModel, EulerDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline,
                       StableDiffusionXLControlNetPipeline,
                       StableDiffusionXLPipeline, T2IAdapter,
                       UNet2DConditionModel)
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)

import wandb
from training_config import training_config

repo = "stabilityai/stable-diffusion-xl-base-1.0"

device_id = int(os.environ["LOCAL_RANK"])

vae: AutoencoderKL = None

tokenizer_one = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")

text_encoder_one: CLIPTextModel = None

tokenizer_two = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer_2")

text_encoder_two: CLIPTextModelWithProjection = None

unet: UNet2DConditionModel = None

scheduler: EulerDiscreteScheduler = None

adapter: T2IAdapter = None

controlnet: ControlNetModel

_init_sdxl_called = False


def init_sdxl():
    global _init_sdxl_called, vae, text_encoder_one, text_encoder_two, unet, scheduler, adapter, controlnet

    if _init_sdxl_called:
        raise ValueError("`init_sdxl` called more than once")

    _init_sdxl_called = True

    text_encoder_one = CLIPTextModel.from_pretrained(
        repo, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16
    )
    text_encoder_one.to(device=device_id)
    text_encoder_one.requires_grad_(False)
    text_encoder_one.eval()

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        repo, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16
    )
    text_encoder_two.to(device=device_id)
    text_encoder_two.requires_grad_(False)
    text_encoder_two.eval()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    vae.to(device=device_id)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

    if training_config.training == "sdxl_unet":
        if training_config.resume_from is not None:
            unet_repo = training_config.resume_from
        else:
            unet_repo = repo

        unet = UNet2DConditionModel.from_pretrained(
            unet_repo,
            subfolder="unet",
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            repo,
            subfolder="unet",
            variant="fp16",
            torch_dtype=torch.float16,
        )

    unet.to(device=device_id)
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()

    if training_config.training == "sdxl_unet":
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
        if training_config.resume_from is None:
            controlnet = ControlNetModel.from_unet(unet)
        else:
            controlnet_repo = os.path.join(training_config.resume_from, "controlnet")
            controlnet = ControlNetModel.from_pretrained(controlnet_repo)

        controlnet.to(device=device_id)
        controlnet.train()
        controlnet.requires_grad_(True)
        controlnet.enable_xformers_memory_efficient_attention()
        controlnet.enable_gradient_checkpointing()
        controlnet = DDP(controlnet, device_ids=[device_id])


def get_sdxl_dataset():
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

    dataset = dataset.batched(
        training_config.batch_size, partial=False, collation_fn=default_collate
    )

    return dataset


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
    resized_and_cropped_and_normalized_image_tensor = TF.normalize(
        resized_and_cropped_image_tensor, [0.5], [0.5]
    )

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
            from mediapipe_pose import mediapipe_pose_adapter_image

            adapter_image = mediapipe_pose_adapter_image(
                resized_and_cropped_image, return_type="vae_scaled_tensor"
            )

            sample["adapter_image"] = adapter_image
        elif training_config.adapter_type == "openpose":
            from openpose import openpose_adapter_image

            adapter_image = openpose_adapter_image(
                resized_and_cropped_image, return_type="vae_scaled_tensor"
            )

            sample["adapter_image"] = adapter_image
        else:
            assert False

    if training_config.training == "sdxl_controlnet":
        if training_config.controlnet_type == "canny":
            controlnet_image = make_canny_conditioning(
                resized_and_cropped_image, return_type="controlnet_scaled_tensor"
            )

            sample["controlnet_image"] = controlnet_image
        elif training_config.controlnet_type == "inpainting":
            from masking import make_masked_image

            controlnet_image = make_masked_image(
                resized_and_cropped_image, return_type="controlnet_scaled_tensor"
            )

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

        prompt_embeds, pooled_prompt_embeds_two = text_conditioning(
            text_input_ids_one, text_input_ids_two
        )

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
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bsz,), device=device_id
            )

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        sigmas = get_sigmas(timesteps)
        sigmas = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        scaled_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        down_block_additional_residuals = None
        mid_block_additional_residual = None

        if training_config.training == "sdxl_adapter":
            adapter_image = batch["adapter_image"].to(device_id)

            down_block_additional_residuals = adapter(adapter_image)

            down_block_additional_residuals = [
                x * training_config.adapter_conditioning_scale
                for x in down_block_additional_residuals
            ]

        if training_config.training == "sdxl_controlnet":
            controlnet_image = batch["controlnet_image"].to(device_id)

            down_block_additional_residuals, mid_block_additional_residual = controlnet(
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

    prompt_embeds_1 = prompt_embeds_1.view(
        prompt_embeds_1.shape[0], prompt_embeds_1.shape[1], -1
    )

    prompt_embeds_2 = text_encoder_two(
        text_input_ids_two,
        output_hidden_states=True,
    )

    pooled_prompt_embeds_2 = prompt_embeds_2[0]

    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    prompt_embeds_2 = prompt_embeds_2.view(
        prompt_embeds_2.shape[0], prompt_embeds_2.shape[1], -1
    )

    prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)

    return prompt_embeds, pooled_prompt_embeds_2


@torch.no_grad()
def log_predicted_images(
    noisy_latents, sigmas, noise, model_pred, timesteps, global_step
):
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

        pipeline = StableDiffusionXLControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            unet=unet,
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

        for validation_image in training_config.validation_images:
            validation_image = Image.open(validation_image)
            validation_image = validation_image.convert("RGB")
            validation_image = validation_image.resize(
                (training_config.resolution, training_config.resolution)
            )

            if training_config.training == "sdxl_adapter":
                if training_config.adapter_type == "mediapipe_pose":
                    from mediapipe_pose import mediapipe_pose_adapter_image

                    validation_image = mediapipe_pose_adapter_image(
                        validation_image, return_type="pil"
                    )
                elif training_config.adapter_type == "openpose":
                    from openpose import openpose_adapter_image

                    validation_image = openpose_adapter_image(
                        validation_image, return_type="pil"
                    )
                else:
                    assert False
            elif training_config.training == "sdxl_controlnet":
                if training_config.controlnet_type == "canny":
                    validation_image = make_canny_conditioning(
                        validation_image, return_type="pil"
                    )
                elif training_config.controlnet_type == "inpainting":
                    # TODO - because we can't get a PIL back here to pass to both the
                    # wandb lob and the pipeline, this is messy+redundant. Is there
                    # a better way to do this?
                    from masking import make_masked_image

                    validation_image = make_masked_image(
                        validation_image, return_type="controlnet_scaled_tensor"
                    )

                    validation_image = validation_image[None, :, :, :]

                    validation_image = validation_image.to("cuda")
                else:
                    assert False
            else:
                assert False

            formatted_validation_images.append(validation_image)

        if (
            training_config.controlnet_type == "inpainting"
            or not _validation_images_logged
        ):
            wandb_validation_images = []

            for validation_image in formatted_validation_images:
                if training_config.controlnet_type == "inpainting":
                    from masking import masked_image_as_pil

                    validation_image = masked_image_as_pil(validation_image[0])

                validation_image = wandb.Image(validation_image)

                wandb_validation_images.append(validation_image)

            wandb.log({"validation_conditioning": wandb_validation_images}, step=step)

            _validation_images_logged = True

    generator = torch.Generator().manual_seed(0)

    output_validation_images = []

    for i, validation_prompt in enumerate(training_config.validation_prompts):
        args = {
            "prompt": validation_prompt,
            "generator": generator,
        }

        if formatted_validation_images is not None:
            args["image"] = formatted_validation_images[i]

        if training_config.training == "sdxl_adapter":
            args[
                "adapter_conditioning_scale"
            ] = training_config.adapter_conditioning_scale
            args[
                "adapter_conditioning_factor"
            ] = training_config.adapter_conditioning_factor

        with torch.autocast("cuda"):
            output_validation_images += pipeline(**args).images

    output_validation_images = [
        wandb.Image(image) for image in output_validation_images
    ]

    wandb.log({"validation": output_validation_images}, step=step)

    if training_config.training == "sdxl_unet":
        unet_.train()
    elif training_config.training == "sdxl_adapter":
        adapter_.train()
    elif training_config.training == "sdxl_controlnet":
        controlnet_.train()
    else:
        assert False


def get_sigmas(timesteps, n_dim=4):
    sigmas = scheduler.sigmas.to(device=timesteps.device)
    schedule_timesteps = scheduler.timesteps.to(device=timesteps.device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()

    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma


def maybe_ddp_dtype(m):
    if isinstance(m, DDP):
        m = m.module
    return m.dtype


def maybe_ddp_module(m):
    if isinstance(m, DDP):
        m = m.module
    return m


def make_canny_conditioning(
    image,
    return_type: Literal[
        "controlnet_scaled_tensor", "pil"
    ] = "controlnet_scaled_tensor",
):
    import cv2

    controlnet_image = np.array(image)
    controlnet_image = cv2.Canny(controlnet_image, 100, 200)
    controlnet_image = controlnet_image[:, :, None]
    controlnet_image = np.concatenate(
        [controlnet_image, controlnet_image, controlnet_image], axis=2
    )

    if return_type == "controlnet_scaled_tensor":
        controlnet_image = TF.to_tensor(controlnet_image)
    elif return_type == "pil":
        controlnet_image = Image.fromarray(controlnet_image)
    else:
        assert False

    return controlnet_image
