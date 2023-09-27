import os
import random
from typing import Union

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
import webdataset as wds
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from diffusion import (default_num_train_timesteps, make_sigmas,
                       sdxl_diffusion_loop)
from sdxl_adapter import SDXLAdapter
from sdxl_controlnet import SDXLControlNet
from sdxl_controlnet_full import SDXLControlNetFull
from sdxl_controlnet_pre_encoded_controlnet_cond import \
    SDXLControlNetPreEncodedControlnetCond
from sdxl_unet import SDXLUNet
from sdxl_vae import SDXLVae
from training_config import training_config
from utils import (maybe_ddp_dtype, maybe_ddp_module, sdxl_text_conditioning,
                   sdxl_tokenize_one, sdxl_tokenize_two)

repo = "stabilityai/stable-diffusion-xl-base-1.0"

device_id = int(os.environ["LOCAL_RANK"])

vae: SDXLVae = None

text_encoder_one: CLIPTextModel = None

text_encoder_two: CLIPTextModelWithProjection = None

unet: SDXLUNet = None

sigmas: torch.Tensor = None

adapter: SDXLAdapter = None

controlnet: Union[SDXLControlNet, SDXLControlNetFull]

_init_sdxl_called = False


def init_sdxl():
    global _init_sdxl_called, vae, text_encoder_one, text_encoder_two, unet, sigmas, adapter, controlnet

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

    vae = SDXLVae.load_fp16_fix(device=device_id)
    vae.requires_grad_(False)
    vae.eval()

    sigmas = make_sigmas().to(device=device_id)

    if training_config.training == "sdxl_unet":
        if training_config.resume_from is not None:
            unet = SDXLUNet.load(training_config.resume_from)
        else:
            unet = SDXLUNet.load_fp32()
    elif training_config.training == "sdxl_controlnet" and training_config.controlnet_train_base_unet:
        unet = SDXLUNet.load_fp32()

        if training_config.resume_from is not None:
            import load_state_dict_patch

            unet_state_dict = safetensors.torch.load_file(os.path.join(training_config.resume_from, "unet.safetensors"), device=device_id)

            load_sd_results = unet.up_blocks.load_state_dict(unet_state_dict, strict=False, assign=True)

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
            adapter = SDXLAdapter()
        else:
            adapter_repo = os.path.join(training_config.resume_from, "adapter")
            adapter = SDXLAdapter.load(adapter_repo)

        adapter.to(device=device_id)
        adapter.train()
        adapter.requires_grad_(True)
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
            controlnet = controlnet_cls.load(controlnet_repo)

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

    micro_conditioning = torch.tensor(
        [
            original_width,
            original_height,
            c_top,
            c_left,
            training_config.resolution,
            training_config.resolution,
        ]
    )

    text_input_ids_one = sdxl_tokenize_one(text)

    text_input_ids_two = sdxl_tokenize_two(text)

    sample = {
        "micro_conditioning": micro_conditioning,
        "text_input_ids_one": text_input_ids_one,
        "text_input_ids_two": text_input_ids_two,
        "image": resized_and_cropped_and_normalized_image_tensor,
    }

    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type == "mediapipe_pose":
            from utils import mediapipe_pose_adapter_image

            adapter_image = mediapipe_pose_adapter_image(resized_and_cropped_image, return_type="vae_scaled_tensor")

            sample["adapter_image"] = adapter_image
        elif training_config.adapter_type == "openpose":
            from utils import openpose_adapter_image

            adapter_image = openpose_adapter_image(resized_and_cropped_image, training_config.resolution, return_type="vae_scaled_tensor")

            sample["adapter_image"] = adapter_image
        else:
            assert False

    if training_config.training == "sdxl_controlnet":
        if training_config.controlnet_type == "canny":
            controlnet_image = make_canny_conditioning(resized_and_cropped_image, return_type="controlnet_scaled_tensor")

            sample["controlnet_image"] = controlnet_image
        elif training_config.controlnet_type == "inpainting":
            from utils import make_masked_image

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

        prompt_embeds, pooled_prompt_embeds_two = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        prompt_embeds = prompt_embeds.to(dtype=unet_dtype)
        pooled_prompt_embeds_two = pooled_prompt_embeds_two.to(dtype=unet_dtype)

        bsz = latents.shape[0]

        if training_config.training == "sdxl_adapter":
            # Cubic sampling to sample a random timestep for each image
            timesteps = torch.rand((bsz,), device=device_id)
            timesteps = (1 - timesteps**3) * default_num_train_timesteps
            timesteps = timesteps.long()
            timesteps = timesteps.clamp(0, default_num_train_timesteps - 1)
        else:
            timesteps = torch.randint(0, default_num_train_timesteps, (bsz,), device=device_id)

        sigmas_ = sigmas[timesteps].to(dtype=latents.dtype)

        noise = torch.randn_like(latents)

        noisy_latents = latents + noise * sigmas_

        scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

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

    return loss


_validation_images_logged = False


@torch.no_grad()
def sdxl_log_validation(step):
    global _validation_images_logged

    unet_ = maybe_ddp_module(unet)
    unet_.eval()

    if training_config.training == "sdxl_adapter":
        adapter_ = maybe_ddp_module(adapter)
        adapter_.eval()
    else:
        adapter_ = None

    if training_config.training == "sdxl_controlnet":
        controlnet_ = maybe_ddp_module(controlnet)
        controlnet_.eval()
    else:
        controlnet_ = None

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

    for formatted_validation_image, validation_prompt in zip(formatted_validation_images, training_config.validation_prompts):
        for _ in range(training_config.num_validation_images):
            with torch.autocast("cuda"):
                x_0 = sdxl_diffusion_loop(
                    prompts=validation_prompt,
                    images=formatted_validation_image,
                    unet=unet_,
                    text_encoder_one=text_encoder_one,
                    text_encoder_two=text_encoder_two,
                    controlnet=controlnet_,
                    adapter=adapter_,
                    sigmas=sigmas,
                    generator=generator,
                )

                x_0 = vae.decode(x_0)[0]

                output_validation_images.append(wandb.Image(x_0, caption=validation_prompt))

    wandb.log({"validation": output_validation_images}, step=step)

    unet_.train()

    if adapter_ is not None:
        adapter_.train()

    if controlnet_ is not None:
        controlnet_.train()


def get_validation_images(validation_image_path):
    validation_image = Image.open(validation_image_path)
    validation_image = validation_image.convert("RGB")
    validation_image = validation_image.resize((training_config.resolution, training_config.resolution))

    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type == "mediapipe_pose":
            from utils import mediapipe_pose_adapter_image

            validation_image = mediapipe_pose_adapter_image(validation_image, return_type="pil")
            log_validation_image = validation_image
        elif training_config.adapter_type == "openpose":
            from utils import openpose_adapter_image

            validation_image = openpose_adapter_image(validation_image, training_config.resolution, return_type="pil")
            log_validation_image = validation_image
        else:
            assert False
    elif training_config.training == "sdxl_controlnet":
        if training_config.controlnet_type == "canny":
            from utils import make_canny_conditioning

            validation_image = make_canny_conditioning(validation_image, return_type="pil")
            log_validation_image = validation_image
        elif training_config.controlnet_type == "inpainting":
            from utils import make_mask, make_masked_image

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
