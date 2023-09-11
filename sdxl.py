import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import (AutoencoderKL, EulerDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline,
                       StableDiffusionXLPipeline, T2IAdapter,
                       UNet2DConditionModel)
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizerFast)

import wandb
from training_config import training_config

vae: AutoencoderKL = None

text_encoder_one: CLIPTextModel = None

text_encoder_two: CLIPTextModelWithProjection = None

unet: UNet2DConditionModel = None

scheduler: EulerDiscreteScheduler = None

adapter: T2IAdapter = None


repo = "stabilityai/stable-diffusion-xl-base-1.0"

_init_sdxl_unet_called = False


def init_sdxl_unet():
    global _init_sdxl_unet_called, vae, text_encoder_one, text_encoder_two, unet, scheduler

    if _init_sdxl_unet_called:
        raise ValueError("`init_sdxl_unet` called more than once")

    _init_sdxl_unet_called = True

    device_id = dist.get_rank()

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

    unet = UNet2DConditionModel.from_pretrained(
        repo,
        subfolder="unet",
    )
    unet.to(device=device_id)
    unet.requires_grad_(True)
    unet.train()
    unet.enable_xformers_memory_efficient_attention()
    unet = DDP(unet, device_ids=[device_id])


_init_sdxl_adapter_called = False


def init_sdxl_adapter():
    global _init_sdxl_adapter_called, vae, text_encoder_one, text_encoder_two, unet, scheduler, adapter

    if _init_sdxl_adapter_called:
        raise ValueError("`init_sdxl_adapter` called more than once")

    _init_sdxl_adapter_called = True

    device_id = dist.get_rank()

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

    unet = UNet2DConditionModel.from_pretrained(
        repo,
        subfolder="unet",
    )
    unet.to(device=device_id)
    unet.requires_grad_(False)
    unet.eval()
    unet.enable_xformers_memory_efficient_attention()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    vae.to(device=device_id)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

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


def sdxl_unet_train_step(batch):
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
        prompt_embeds = prompt_embeds.to(dtype=unet.module.dtype)
        pooled_prompt_embeds_two = pooled_prompt_embeds_two.to(dtype=unet.module.dtype)

        bsz = latents.shape[0]

        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,))

        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        sigmas = get_sigmas(timesteps)
        sigmas = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "time_ids": time_ids,
                "text_embeds": pooled_prompt_embeds_two,
            },
        ).sample

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    return loss


def sdxl_adapter_train_step(batch):
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

        sigmas = get_sigmas(timesteps)
        sigmas = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

    with torch.autocast(
        "cuda",
        training_config.mixed_precision,
        enabled=training_config.mixed_precision is not None,
    ):
        adapter_image = batch["adapter_image"].to(device_id)
        down_block_additional_residuals = adapter(adapter_image)

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

        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

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


tokenizer_one = CLIPTokenizerFast.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
)

tokenizer_two = CLIPTokenizerFast.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
)


@torch.no_grad()
def sdxl_log_unet_validation(step):
    unet_ = unet.module
    unet_.eval()

    # NOTE - this has to be different from the module level scheduler because
    # the pipeline mutates it.
    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        unet=unet_,
        scheduler=scheduler,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
    )

    pipeline.set_progress_bar_config(disable=True)

    output_validation_images = []

    for validation_prompt in training_config.validation_prompts:
        for _ in range(training_config.num_validation_images):
            with torch.autocast("cuda"):
                output_validation_images += pipeline(
                    prompt=validation_prompt,
                ).images

    output_validation_images = [
        wandb.Image(image) for image in output_validation_images
    ]

    wandb.log({"validation": output_validation_images}, step=step)

    unet_.train()


@torch.no_grad()
def sdxl_log_adapter_validation(step):
    adapter_ = adapter.module
    adapter_.eval()

    # NOTE - this has to be different from the module level scheduler because
    # the pipeline mutates it.
    scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

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

    pipeline.set_progress_bar_config(disable=True)

    formatted_validation_images = []

    for validation_image in training_config.validation_images:
        validation_image = Image.open(validation_image)
        validation_image = validation_image.convert("RGB")
        validation_image = validation_image.resize(
            (training_config.resolution, training_config.resolution)
        )
        formatted_validation_images.append(validation_image)

    output_validation_images = []

    for validation_prompt, validation_image in zip(
        training_config.validation_prompts, formatted_validation_images
    ):
        for _ in range(training_config.num_validation_images):
            with torch.autocast("cuda"):
                output_validation_images += pipeline(
                    prompt=validation_prompt,
                    image=validation_image,
                ).images

    output_validation_images = [
        wandb.Image(image) for image in output_validation_images
    ]

    wandb.log({"validation": output_validation_images}, step=step)

    adapter_.train()


def get_sigmas(timesteps, n_dim=4):
    sigmas = scheduler.sigmas
    schedule_timesteps = scheduler.timesteps

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()

    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma
