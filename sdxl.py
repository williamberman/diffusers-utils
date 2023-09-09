import os

import torch
import torch.distributed as dist
import torch.functional as F
from diffusers import (AutoencoderKL, EulerDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline, T2IAdapter,
                       UNet2DConditionModel)
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizerFast)

import wandb
from training_config import training_config

repo = "stabilityai/stable-diffusion-xl-base-1.0"

vae: AutoencoderKL = None

tokenizer_one: CLIPTokenizerFast = None
text_encoder_one: CLIPTextModel = None

tokenizer_two: CLIPTokenizerFast = None
text_encoder_two: CLIPTextModelWithProjection = None

unet: UNet2DConditionModel = None

scheduler: EulerDiscreteScheduler = None

adapter: T2IAdapter = None

_init_sdxl_called = False


def init_sdxl():
    global _init_sdxl_called, vae, tokenizer_one, text_encoder_one, tokenizer_two, text_encoder_two, unet, scheduler, adapter

    if _init_sdxl_called:
        raise ValueError("`init_sdxl_models` called more than once")

    _init_sdxl_called = True

    device_id = dist.get_rank()

    tokenizer_one = CLIPTokenizerFast.from_pretrained(repo, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizerFast.from_pretrained(repo, subfolder="tokenizer_2")

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
        repo, subfolder="unet", variant="fp16", torch_dtype=torch.float16
    )
    unet.to(device=device_id)
    unet.requires_grad_(False)
    unet.train(False)
    unet.enable_xformers_memory_efficient_attention()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
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

    time_ids = batch["time_ids"].to(device_id)
    latents = batch["latents"].to(device_id)
    prompt_embeds = batch["prompt_embeds"].to(device_id)
    text_embeds = batch["text_embeds"].to(device_id)
    adapter_image = batch["adapter_image"].to(device_id)

    bsz = latents.shape[0]

    # Cubic sampling to sample a random timestep for each image
    timesteps = torch.rand((bsz,), device=device_id)
    timesteps = (1 - timesteps**3) * scheduler.config.num_train_timesteps
    timesteps = timesteps.long().to(scheduler.timesteps.dtype)
    timesteps = timesteps.clamp(0, scheduler.config.num_train_timesteps - 1)

    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    if training_config.training == "sdxl_adapter":
        down_block_additional_residuals = adapter(adapter_image)
    else:
        down_block_additional_residuals = None

    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"time_ids": time_ids, "text_embeds": text_embeds},
        down_block_additional_residuals=down_block_additional_residuals,
    ).sample

    loss = F.mse_loss(model_pred.float(), noise, reduction="mean")

    return loss


@torch.no_grad()
def sdxl_log_adapter_validation(step):
    adapter_ = adapter.module

    pipeline = StableDiffusionXLAdapterPipeline(
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
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
