import os
import random
from typing import Union

import safetensors.torch
import torch
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional as TF
import wandb
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from diffusion import (default_num_train_timesteps, make_sigmas,
                       sdxl_diffusion_loop)
from sdxl_models import (SDXLAdapter, SDXLControlNet, SDXLControlNetFull,
                         SDXLControlNetPreEncodedControlnetCond, SDXLUNet,
                         SDXLVae)
from training_config import training_config
from utils import (get_random_crop_params, get_sdxl_conditioning_images,
                   maybe_ddp_dtype, maybe_ddp_module, sdxl_text_conditioning,
                   sdxl_tokenize_one, sdxl_tokenize_two)

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

    text_encoder_one = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
    text_encoder_one.to(device=device_id)
    text_encoder_one.requires_grad_(False)
    text_encoder_one.eval()

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16)
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

    if training_config.training in ["sdxl_controlnet", "sdxl_adapter"]:
        dataset = dataset.select(conditioning_image_present)

    dataset = dataset.batched(training_config.batch_size, partial=False, collation_fn=default_collate)

    return dataset


@torch.no_grad()
def make_sample(d):
    image = d["image"]
    metadata = d["metadata"]

    if random.random() < training_config.proportion_empty_prompts:
        text = ""
    else:
        text = d["text"]

    c_top, c_left, _, _ = get_random_crop_params([image.height, image.width], [1024, 1024])

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

    image = image.convert("RGB")

    image = TF.resize(
        image,
        training_config.resolution,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )

    image = TF.crop(
        image,
        c_top,
        c_left,
        training_config.resolution,
        training_config.resolution,
    )

    sample = {
        "micro_conditioning": micro_conditioning,
        "text_input_ids_one": text_input_ids_one,
        "text_input_ids_two": text_input_ids_two,
        "image": TF.normalize(TF.to_tensor(image), [0.5], [0.5]),
    }

    conditioning_image, conditioning_image_mask = get_sdxl_conditioning_images(
        image,
        conditioning_type=training_config.training,
        adapter_type=training_config.adapter_type,
        controlnet_type=training_config.controlnet_type,
        controlnet_variant=training_config.controlnet_variant,
    )

    if conditioning_image is not None:
        sample["conditioning_image"] = conditioning_image

    if conditioning_image_mask is not None:
        sample["conditioning_image_mask"] = conditioning_image_mask

    return sample


def conditioning_image_present(sample):
    return "conditioning_image" in sample and sample["conditioning_image"] is not None


def sdxl_train_step(batch):
    with torch.no_grad():
        unet_dtype = maybe_ddp_dtype(unet)

        micro_conditioning = batch["micro_conditioning"].to(device=device_id)

        image = batch["image"].to(device_id, dtype=vae.dtype)
        latents = vae.encode(image).latent_dist.sample().to(dtype=unet_dtype)
        latents = latents * vae.config.scaling_factor

        text_input_ids_one = batch["text_input_ids_one"].to(device_id)
        text_input_ids_two = batch["text_input_ids_two"].to(device_id)

        encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two)

        encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet_dtype)

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

                controlnet_image = batch["conditioning_image"].to(device_id, dtype=vae.dtype)
                controlnet_image = vae.encode(controlnet_image).to(dtype=controlnet_dtype)

                _, _, controlnet_image_height, controlnet_image_width = controlnet_image.shape
                controlnet_image_mask = batch["conditioning_image_mask"].to(device=device_id)
                controlnet_image_mask = TF.resize(controlnet_image_mask, (controlnet_image_height, controlnet_image_width)).to(dtype=controlnet_dtype)

                controlnet_image = torch.concat((controlnet_image, controlnet_image_mask), dim=1)
            else:
                controlnet_image = batch["controlnet_image"].to(device_id)

        if training_config.training == "sdxl_adapter":
            adapter_image = batch["conditioning_image"].to(device_id)

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

        if training_config.training == "sdxl_controlnet":
            controlnet_out = controlnet(
                x_t=scaled_noisy_latents,
                t=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                micro_conditioning=micro_conditioning,
                pooled_encoder_hidden_states=pooled_encoder_hidden_states,
                controlnet_cond=controlnet_image,
            )

            down_block_additional_residuals = controlnet_out["down_block_res_samples"]
            mid_block_additional_residual = controlnet_out["mid_block_res_sample"]
            add_to_down_block_inputs = controlnet_out.get("add_to_down_block_inputs", None)
            add_to_output = controlnet_out.get("add_to_output", None)

        model_pred = unet(
            x_t=scaled_noisy_latents,
            t=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
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
            validation_image, _ = get_sdxl_conditioning_images(
                validation_image_path,
                conditioning_type=training_config.training,
                adapter_type=training_config.adapter_type,
                controlnet_type=training_config.controlnet_type,
                controlnet_variant=training_config.controlnet_variant,
                vae=vae,
            )

            formatted_validation_images.append(validation_image)
            wandb_validation_images.append(wandb.Image(log_validation_image)) # TODO - need a printable image

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
