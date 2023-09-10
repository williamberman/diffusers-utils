import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import (AutoencoderKL, EulerDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline, T2IAdapter,
                       UNet2DConditionModel)
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPTextModel, CLIPTextModelWithProjection

import wandb
from training_config import training_config

vae: AutoencoderKL = None

text_encoder_one: CLIPTextModel = None

text_encoder_two: CLIPTextModelWithProjection = None

unet: UNet2DConditionModel = None

scheduler: EulerDiscreteScheduler = None

adapter: T2IAdapter = None

_init_sdxl_called = False


def init_sdxl():
    global _init_sdxl_called, vae, text_encoder_one, text_encoder_two, unet, scheduler, adapter

    if _init_sdxl_called:
        raise ValueError("`init_sdxl` called more than once")

    _init_sdxl_called = True

    device_id = dist.get_rank()

    repo = "stabilityai/stable-diffusion-xl-base-1.0"

    text_encoder_one = CLIPTextModel.from_pretrained(
        repo, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16
    )
    text_encoder_one.to(device=device_id)
    text_encoder_one.requires_grad_(False)
    text_encoder_one.train(False)

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        repo, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16
    )
    text_encoder_two.to(device=device_id)
    text_encoder_two.requires_grad_(False)
    text_encoder_two.train(False)

    unet = UNet2DConditionModel.from_pretrained(
        repo,
        subfolder="unet",  # variant="fp16", torch_dtype=torch.float16
    )
    unet.to(device=device_id)
    unet.requires_grad_(False)
    unet.train(False)
    unet.enable_xformers_memory_efficient_attention()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",  # torch_dtype=torch.float16
    )
    vae.to(device=device_id)
    vae.requires_grad_(False)
    vae.train(False)

    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

    if training_config.training == "sdxl_adapter":
        adapter = T2IAdapter(
            in_channels=3,
            channels=(320, 640, 1280, 1280),
            num_res_blocks=2,
            downscale_factor=16,
            adapter_type="full_adapter_xl",
        )
        adapter.to(device=device_id)
        adapter.train()
        adapter.requires_grad_(True)
        adapter.enable_xformers_memory_efficient_attention()
        adapter = DDP(adapter, device_ids=[device_id])
    else:
        assert False


def sdxl_train_step(batch):
    device_id = dist.get_rank()

    with torch.no_grad():
        time_ids = batch["time_ids"].to(device_id)

        image = batch["image"].to(device_id, dtype=vae.dtype)
        latents = vae.encode(image).latent_dist.sample()

        text_input_ids_one = batch["text_input_ids_one"].to(device_id)
        text_input_ids_two = batch["text_input_ids_two"].to(device_id)

        prompt_embeds, pooled_prompt_embeds_two = text_conditioning(
            text_input_ids_one, text_input_ids_two
        )
        prompt_embeds = prompt_embeds.to(dtype=unet.dtype)
        pooled_prompt_embeds_two = pooled_prompt_embeds_two.to(dtype=unet.dtype)

        bsz = latents.shape[0]

        # Cubic sampling to sample a random timestep for each image
        timesteps = torch.rand((bsz,), device=device_id)
        timesteps = (1 - timesteps**3) * scheduler.config.num_train_timesteps
        timesteps = timesteps.long().to(scheduler.timesteps.dtype)
        timesteps = timesteps.clamp(0, scheduler.config.num_train_timesteps - 1)

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        if training_config.training == "sdxl_adapter":
            adapter_image = batch["adapter_image"].to(device_id)
            down_block_additional_residuals = adapter(adapter_image)
        else:
            down_block_additional_residuals = None

        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "time_ids": time_ids,
                "text_embeds": pooled_prompt_embeds_two,
            },
            down_block_additional_residuals=down_block_additional_residuals,
        ).sample

        loss = F.mse_loss(model_pred.float(), noise, reduction="mean")

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
def sdxl_log_adapter_validation(step):
    adapter_ = adapter.module

    pipeline = StableDiffusionXLAdapterPipeline(
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        unet=unet,
        adapter=adapter_,
        scheduler=scheduler,
    )

    pipeline.set_progress_bar_config(disable=True)

    validation_images = [
        os.path.join(validation_image, f"{i}.png")
        for i in range(len(validation_prompt))
    ]

    validation_images = [Image.open(x).convert("RGB") for x in validation_images]

    image_logs = []

    output_validation_images = []

    for validation_prompt, validation_image in zip(
        training_config.validation_prompts, validation_images
    ):
        with torch.autocast("cuda"):
            output_validation_images += pipeline(
                prompt=validation_prompt,
                image=validation_image,
                num_images_per_prompt=training_config.num_validation_images,
                adapter_conditioning_scale=1.5,
            ).images

    for i, validation_prompt in enumerate(training_config.validation_prompts):
        validation_image = validation_images[i]

        output_validation_images_ = output_validation_images[
            i
            * training_config.num_validation_images : i
            * training_config.num_validation_images
            + training_config.num_validation_images
        ]

        image_logs.append(
            {
                "validation_image": validation_image,
                "images": output_validation_images_,
                "validation_prompt": validation_prompt,
            }
        )

    formatted_images = []

    for log in image_logs:
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        validation_image = log["validation_image"]

        formatted_images.append(
            wandb.Image(validation_image, caption="adapter conditioning")
        )

        for image in images:
            image = wandb.Image(image, caption=validation_prompt)
            formatted_images.append(image)

    wandb.log({"validation": formatted_images}, step=step)
