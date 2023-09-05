import os

import torch
import torch.functional as F
import torchvision.transforms.functional as TF
import wandb
import webdataset as wds
from accelerate import Accelerator
from diffusers import (AutoencoderKL, EulerDiscreteScheduler,
                       StableDiffusionXLAdapterPipeline, T2IAdapter,
                       UNet2DConditionModel)
from PIL import Image
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer)

from .training_loop import Config


class SDXLT2IOpenpose:
    @classmethod
    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        assert len(weights) == 1

        weights = weights.pop()
        model = models[0]
        model.save_pretrained(os.path.join(output_dir, "t2iadapter"))

    @classmethod
    def load_model_hook(models, input_dir):
        assert len(models) == 1
        model = models.pop()
        model.from_pretrained(input_dir, subfolder="t2iadapter")

    def __init__(self, config: Config, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator

        mixed_precision = accelerator.mixed_precision
        device = accelerator.device

        if mixed_precision == "no":
            dtype = torch.float32
        elif mixed_precision == "fp16":
            dtype = torch.float16
        elif mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            assert False

        repo = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_repo = "madebyollin/sdxl-vae-fp16-fix"

        tokenizer_one = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        tokenizer_two = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer_2")

        text_encoder_one = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder")
        text_encoder_one.to(device=device, dtype=dtype)
        text_encoder_one.requires_grad_(False)
        text_encoder_one.train(False)

        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            repo, subfolder="text_encoder_2"
        )
        text_encoder_two.to(device=device, dtype=dtype)
        text_encoder_two.requires_grad_(False)
        text_encoder_two.train(False)

        unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet")
        unet.to(device=device, dtype=dtype)
        unet.train(False)
        unet.requires_grad_(False)
        unet.enable_xformers_memory_efficient_attention()

        vae = AutoencoderKL.from_pretrained(vae_repo)
        vae.to(device=device, dtype=dtype)
        vae.requires_grad_(False)
        vae.train(False)

        adapter = T2IAdapter(
            in_channels=3,
            channels=(320, 640, 1280, 1280),
            num_res_blocks=2,
            downscale_factor=16,
            adapter_type="full_adapter_xl",
        )
        adapter.train()
        adapter.requires_grad_(True)
        adapter.enable_xformers_memory_efficient_attention()
        adapter = accelerator.prepare(adapter)

        scheduler = EulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.unet = unet
        self.vae = vae
        self.adapter = adapter
        self.scheduler = scheduler

    def get_dataset(self):
        def process_data(d, resolution):
            image = d["image"]
            text = d["text"]
            metadata = d["json"]

            image = TF.resize(
                image, resolution, interpolation=transforms.InterpolationMode.BILINEAR
            )

            c_top, c_left, _, _ = transforms.RandomCrop.get_params(
                image, output_size=(resolution, resolution)
            )

            image = TF.crop(image, c_top, c_left, resolution, resolution)
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            original_width = int(metadata.get("original_width", 0.0))
            original_height = int(metadata.get("original_height", 0.0))

            time_ids = torch.tensor(
                [original_width, original_height, c_top, c_left, resolution, resolution]
            )

            prompt_embeds, pooled_prompt_embeds_2 = self.text_conditioning(text)

            latents = self.vae.encode(
                image.to(device=self.vae.device, dtype=self.vae.dtype)
            ).latent_dist.sample()

            return {
                "time_ids": time_ids,
                "latents": latents.to("cpu"),
                "prompt_embeds": prompt_embeds.to("cpu"),
                "text_embeds": pooled_prompt_embeds_2.to("cpu"),
                "adapter_image": None,  # TODO
            }

        return wds.DataPipeline(
            wds.ResampledShards(self.train_shards),
            wds.shuffle(self.shuffle_buffer_size),
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp",
                text="text;txt;caption",
                metadata="json",
                handler=wds.warn_and_continue,
            ),
            wds.map(
                lambda x: process_data(
                    x, resolution=self.resolution, training_spec=self.training_spec
                )
            ),
            wds.batched(self.batch_size, partial=False, collation_fn=default_collate),
        )

    def training_parameters(self):
        return self.adapter.parameters()

    def training_model(self):
        return self.adapter

    def train_step(self, batch):
        time_ids = batch["time_ids"].to(self.accelerator.device)
        latents = batch["latents"].to(self.accelerator.device)
        prompt_embeds = batch["prompt_embeds"].to(self.accelerator.device)
        text_embeds = batch["text_embeds"].to(self.accelerator.device)
        adapter_image = batch["adapter_image"].to(self.accelerator.device)

        bsz = latents.shape[0]

        # Cubic sampling to sample a random timestep for each image
        timesteps = torch.rand((bsz,), device=latents.device)
        timesteps = (1 - timesteps**3) * self.scheduler.config.num_train_timesteps
        timesteps = timesteps.long().to(self.scheduler.timesteps.dtype)
        timesteps = timesteps.clamp(0, self.scheduler.config.num_train_timesteps - 1)

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        down_block_additional_residuals = self.adapter(adapter_image)
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"time_ids": time_ids, "text_embeds": text_embeds},
            down_block_additional_residuals=down_block_additional_residuals,
        ).sample

        loss = F.mse_loss(model_pred.float(), noise, reduction="mean")

        return loss

    def log_validation(self):
        pipeline = StableDiffusionXLAdapterPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            unet=self.unet,
            adapter=self.accelerator.unwrap_model(self.adapter),
            scheduler=self.scheduler,
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
            self.validation_prompts, validation_images
        ):
            with torch.autocast("cuda"):
                output_validation_images += pipeline(
                    prompt=validation_prompt,
                    image=validation_image,
                    num_images_per_prompt=self.num_validation_images,
                    adapter_conditioning_scale=1.5,
                ).images

        for i, validation_prompt in enumerate(self.validation_prompts):
            validation_image = validation_images[i]

            output_validation_images_ = output_validation_images[
                i * self.num_validation_images : i * self.num_validation_images
                + self.num_validation_images
            ]

            image_logs.append(
                {
                    "validation_image": validation_image,
                    "images": output_validation_images_,
                    "validation_prompt": validation_prompt,
                }
            )

        tracker = self.accelerator.trackers[0]

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

        tracker.log({"validation": formatted_images})

    @torch.no_grad()
    def text_conditioning(self, text):
        text_input_ids = self.tokenizer_1(
            text,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_embeds_1 = self.text_encoder_1(
            text_input_ids.to(
                dtype=self.text_encoder_1.dtype, device=self.text_encoder_1.device
            ),
            output_hidden_states=True,
        ).hidden_states[-2]

        prompt_embeds_1 = prompt_embeds_1.view(
            prompt_embeds_1.shape[0], prompt_embeds_1.shape[1], -1
        )

        text_input_ids = self.tokenizer_2(
            text,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_embeds_2 = self.text_encoder_2(
            text_input_ids.to(
                dtype=self.text_encoder_2.dtype, device=self.text_encoder_2.device
            ),
            output_hidden_states=True,
        )

        pooled_prompt_embeds_2 = prompt_embeds_2[0]

        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

        prompt_embeds_2 = prompt_embeds_2.view(
            prompt_embeds_2.shape[0], prompt_embeds_2.shape[1], -1
        )

        prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2))

        return prompt_embeds, pooled_prompt_embeds_2

    def save_results(self):
        adapter = self.accelerator.unwrap_model(self.adapter)
        adapter.save_pretrained(self.accelerator.project_dir)


def get_sigmas(noise_scheduler, timesteps):
    sigmas = noise_scheduler.sigmas
    schedule_timesteps = noise_scheduler.timesteps

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    return sigma
