import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import (AutoencoderKL, EulerDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline, T2IAdapter,
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

_init_sdxl_called = False

repo = "stabilityai/stable-diffusion-xl-base-1.0"


def init_sdxl():
    global _init_sdxl_called, vae, text_encoder_one, text_encoder_two, unet, scheduler, adapter

    if _init_sdxl_called:
        raise ValueError("`init_sdxl` called more than once")

    _init_sdxl_called = True

    device_id = dist.get_rank()

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

    with torch.autocast("cuda"):
        output_validation_images = pipeline(
            prompt=training_config.validation_prompts,
            image=formatted_validation_images,
            num_images_per_prompt=training_config.num_validation_images,
        ).images

    output_validation_images = [
        wandb.Image(image) for image in output_validation_images
    ]

    wandb.log({"validation": output_validation_images}, step=step)

    adapter_.train()
