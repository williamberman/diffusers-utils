import os
import random
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional as TF
import wandb
import webdataset as wds
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from diffusion import (default_num_train_timesteps, make_sigmas,
                       sdxl_diffusion_loop)
from sdxl_models import (SDXLAdapter, SDXLControlNet, SDXLControlNetFull,
                         SDXLControlNetPreEncodedControlnetCond, SDXLUNet,
                         SDXLVae)
from training_config import Config
from utils import (make_outpainting_mask, make_random_irregular_mask,
                   make_random_rectangle_mask, maybe_ddp_device,
                   maybe_ddp_dtype, maybe_ddp_module, sdxl_text_conditioning,
                   sdxl_tokenize_one, sdxl_tokenize_two)


class SDXLModels:
    text_encoder_one: CLIPTextModel
    text_encoder_two: CLIPTextModelWithProjection
    vae: SDXLVae
    sigmas: torch.Tensor
    unet: SDXLUNet
    adapter: Optional[SDXLAdapter]
    controlnet: Optional[Union[SDXLControlNet, SDXLControlNetFull]]

    mixed_precision: Optional[torch.dtype]
    timestep_sampling: Literal["uniform", "cubic"]

    validation_images_logged: bool
    log_validation_input_images_every_time: bool

    @classmethod
    def from_training_config(cls, training_config: Config, device):
        if training_config.training == "sdxl_controlnet":
            if training_config.controlnet_variant == "default":
                controlnet_cls = SDXLControlNet
            elif training_config.controlnet_variant == "full":
                controlnet_cls = SDXLControlNetFull
            elif training_config.controlnet_variant == "pre_encoded_controlnet_cond":
                controlnet_cls = SDXLControlNetPreEncodedControlnetCond
            else:
                assert False
        else:
            controlnet_cls = None

        if training_config.training == "sdxl_adapter":
            adapter_cls = SDXLAdapter
        else:
            adapter_cls = None

        if training_config.training == "sdxl_adapter":
            timestep_sampling = "cubic"
        else:
            timestep_sampling = "uniform"

        if training_config.training == "sdxl_controlnet" and training_config.controlnet_type == "inpainting":
            log_validation_input_images_every_time = True
        else:
            log_validation_input_images_every_time = False

        return cls(
            device=device,
            train_unet=training_config.training == "sdxl_unet",
            train_unet_up_blocks=training_config.training == "sdxl_controlnet" and training_config.controlnet_train_base_unet,
            unet_resume_from=training_config.resume_from is not None and os.path.join(training_config.resume_from, "unet.safetensors"),
            controlnet_cls=controlnet_cls,
            adapter_cls=adapter_cls,
            adapter_resume_from=training_config.resume_from is not None and os.path.join(training_config.resume_from, "adapter.safetensors"),
            timestep_sampling=timestep_sampling,
            log_validation_input_images_every_time=log_validation_input_images_every_time,
        )

    def __init__(
        self,
        device,
        train_unet,
        train_unet_up_blocks,
        unet_resume_from=None,
        controlnet_cls=None,
        controlnet_resume_from=None,
        adapter_cls=None,
        adapter_resume_from=None,
        mixed_precision=None,
        timestep_sampling="uniform",
        log_validation_input_images_every_time=True,
    ):
        self.text_encoder_one = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
        self.text_encoder_one.to(device=device)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_one.eval()

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16)
        self.text_encoder_two.to(device=device)
        self.text_encoder_two.requires_grad_(False)
        self.text_encoder_two.eval()

        self.vae = SDXLVae.load_fp16_fix(device=device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.sigmas = make_sigmas(device=device)

        if train_unet:
            if unet_resume_from is None:
                self.unet = SDXLUNet.load_fp32(device=device)
            else:
                self.unet = SDXLUNet.load(unet_resume_from, device=device)
            self.unet.requires_grad_(True)
            self.unet.train()
            self.unet = DDP(self.unet, device_ids=[device])
        elif train_unet_up_blocks:
            if unet_resume_from is None:
                self.unet = SDXLUNet.load_fp32(device=device)
            else:
                self.unet = SDXLUNet.load_fp32(device=device, overrides=[unet_resume_from])
            self.unet.requires_grad_(False)
            self.unet.eval()
            self.unet.up_blocks.requires_grad_(True)
            self.unet.up_blocks.train()
            self.unet = DDP(self.unet, device_ids=[device], find_unused_parameters=True)
        else:
            self.unet = SDXLUNet.load_fp16(device=device)
            self.unet.requires_grad_(False)
            self.unet.eval()

        if controlnet_cls is not None:
            if controlnet_resume_from is None:
                self.controlnet = controlnet_cls.from_unet(unet)
                self.controlnet.to(device)
            else:
                self.controlnet = controlnet_cls.load(controlnet_resume_from, device=device)
            self.controlnet.train()
            self.controlnet.requires_grad_(True)
            # TODO add back
            # controlnet.enable_gradient_checkpointing()
            # TODO - should be able to remove find_unused_parameters. Comes from pre encoded controlnet
            self.controlnet = DDP(self.controlnet, device_ids=[device], find_unused_parameters=True)
        else:
            self.controlnet = None

        if adapter_cls is not None:
            if adapter_resume_from is None:
                self.adapter = adapter_cls()
                self.adapter.to(device=device)
            else:
                self.adapter = adapter_cls.load(adapter_resume_from, device=device)
            self.adapter.train()
            self.adapter.requires_grad_(True)
            self.adapter = DDP(self.adapter, device_ids=[device])
        else:
            self.adapter = None

        self.mixed_precision = mixed_precision
        self.timestep_sampling = timestep_sampling

        self.validation_images_logged = False
        self.log_validation_input_images_every_time = log_validation_input_images_every_time

    def train_step(self, batch):
        with torch.no_grad():
            unet_dtype = maybe_ddp_dtype(self.unet)
            unet_device = maybe_ddp_device(self.unet)

            micro_conditioning = batch["micro_conditioning"].to(device=unet_device)

            image = batch["image"].to(self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(image).to(dtype=unet_dtype)

            text_input_ids_one = batch["text_input_ids_one"].to(self.text_encoder_one.device)
            text_input_ids_two = batch["text_input_ids_two"].to(self.text_encoder_two.device)

            encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(self.text_encoder_one, self.text_encoder_two, text_input_ids_one, text_input_ids_two)

            encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype)
            pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet_dtype)

            bsz = latents.shape[0]

            if self.timestep_sampling == "uniform":
                timesteps = torch.randint(0, default_num_train_timesteps, (bsz,), device=unet_device)
            elif self.timestep_sampling == "cubic":
                # Cubic sampling to sample a random timestep for each image
                timesteps = torch.rand((bsz,), device=unet_device)
                timesteps = (1 - timesteps**3) * default_num_train_timesteps
                timesteps = timesteps.long()
                timesteps = timesteps.clamp(0, default_num_train_timesteps - 1)
            else:
                assert False

            sigmas_ = self.sigmas[timesteps].to(dtype=latents.dtype)

            noise = torch.randn_like(latents)

            noisy_latents = latents + noise * sigmas_

            scaled_noisy_latents = noisy_latents / ((sigmas_**2 + 1) ** 0.5)

            if "conditioning_image" in batch:
                conditioning_image = batch["conditioning_image"].to(unet_device)

            if self.controlnet is not None and isinstance(self.controlnet, SDXLControlNetPreEncodedControlnetCond):
                controlnet_device = maybe_ddp_device(self.controlnet)
                controlnet_dtype = maybe_ddp_dtype(self.controlnet)
                conditioning_image = self.vae.encode(conditioning_image.to(self.vae.dtype)).to(device=controlnet_device, dtype=controlnet_dtype)
                conditioning_image_mask = TF.resize(batch["conditioning_image_mask"], conditioning_image.shape[2:]).to(device=controlnet_device, dtype=controlnet_dtype)
                conditioning_image = torch.concat((conditioning_image, conditioning_image_mask), dim=1)

        with torch.autocast(
            "cuda",
            self.mixed_precision,
            enabled=self.mixed_precision is not None,
        ):
            down_block_additional_residuals = None
            mid_block_additional_residual = None
            add_to_down_block_inputs = None
            add_to_output = None

            if self.adapter is not None:
                down_block_additional_residuals = self.adapter(conditioning_image)

            if self.controlnet is not None:
                controlnet_out = self.controlnet(
                    x_t=scaled_noisy_latents,
                    t=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    micro_conditioning=micro_conditioning,
                    pooled_encoder_hidden_states=pooled_encoder_hidden_states,
                    controlnet_cond=conditioning_image,
                )

                down_block_additional_residuals = controlnet_out["down_block_res_samples"]
                mid_block_additional_residual = controlnet_out["mid_block_res_sample"]
                add_to_down_block_inputs = controlnet_out.get("add_to_down_block_inputs", None)
                add_to_output = controlnet_out.get("add_to_output", None)

            model_pred = self.unet(
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

    @torch.no_grad()
    def log_validation(self, step, num_validation_images: int, validation_prompts: Optional[List[str]] = None, validation_images: Optional[List[str]] = None):
        unet = maybe_ddp_module(self.unet)
        unet.eval()

        if self.adapter is not None:
            adapter = maybe_ddp_module(self.adapter)
            adapter.eval()
        else:
            adapter = None

        if self.controlnet is not None:
            controlnet = maybe_ddp_module(self.controlnet)
            controlnet.eval()
        else:
            controlnet = None

        formatted_validation_images = None

        if validation_images is not None:
            formatted_validation_images = []
            wandb_validation_images = []

            for validation_image_path in validation_images:
                validation_image = Image.open(validation_image_path)
                validation_image = validation_image.convert("RGB")
                validation_image = validation_image.resize((1024, 1024))

                conditioning_images = get_sdxl_conditioning_images(validation_image)

                conditioning_image = conditioning_images["conditioning_image"]

                if self.controlnet is not None and isinstance(self.controlnet, SDXLControlNetPreEncodedControlnetCond):
                    conditioning_image = self.vae.encode(conditioning_image[None, :, :, :].to(self.vae.device, dtype=self.vae.dtype))
                    conditionin_mask_image = TF.resize(conditioning_images["conditioning_mask_image"], conditioning_image.shape[2:]).to(conditioning_image.dtype, conditioning_image.device)
                    conditioning_image = torch.concat(conditioning_image, conditionin_mask_image, dim=1)

                formatted_validation_images.append(conditioning_image)
                wandb_validation_images.append(wandb.Image(conditioning_images["conditioning_image_as_pil"]))

            if self.log_validation_input_images_every_time or not self.validation_images_logged:
                wandb.log({"validation_conditioning": wandb_validation_images}, step=step)
                self.validation_images_logged = True

        generator = torch.Generator().manual_seed(0)

        output_validation_images = []

        for formatted_validation_image, validation_prompt in zip(formatted_validation_images, validation_prompts):
            for _ in range(num_validation_images):
                with torch.autocast("cuda"):
                    x_0 = sdxl_diffusion_loop(
                        prompts=validation_prompt,
                        images=formatted_validation_image,
                        unet=unet,
                        text_encoder_one=self.text_encoder_one,
                        text_encoder_two=self.text_encoder_two,
                        controlnet=controlnet,
                        adapter=adapter,
                        sigmas=self.sigmas,
                        generator=generator,
                    )

                    x_0 = self.vae.decode(x_0)[0]

                    output_validation_images.append(wandb.Image(x_0, caption=validation_prompt))

        wandb.log({"validation": output_validation_images}, step=step)

        unet.train()

        if adapter is not None:
            adapter.train()

        if controlnet is not None:
            controlnet.train()


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
        dataset = dataset.select(conditioning_image_is_not_none)

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

    if training_config.training in ["sdxl_adapter", "sdxl_controlnet"]:
        conditioning_images = get_sdxl_conditioning_images(image)

        sample["conditioning_image"] = conditioning_images["conditioning_image"]

        if conditioning_images["conditioning_image_mask"] is not None:
            sample["conditioning_image_mask"] = conditioning_images["conditioning_image_mask"]

    return sample


def get_random_crop_params(input_size: Tuple[int, int], output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = input_size

    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()

    return i, j, th, tw


def conditioning_image_is_not_none(sample):
    return sample["conditioning_image"] is not None


if training_config.training == "sdxl_adapter" and training_config.adapter_type == "openpose":
    from controlnet_aux import OpenposeDetector

    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")


def get_sdxl_conditioning_images(image):
    resolution = image.width

    if training_config.training == "sdxl_adapter" and training_config.adapter_type == "openpose":
        conditioning_image = open_pose(image, detect_resolution=resolution, image_resolution=resolution, return_pil=False)

        if (conditioning_image == 0).all():
            return None, None

        conditioning_image_as_pil = Image.fromarray(conditioning_image)

        conditioning_image = TF.to_tensor(conditioning_image)

    if training_config.training == "sdxl_controlnet" and training_config.controlnet_type == "canny":
        import cv2

        conditioning_image = np.array(image)
        conditioning_image = cv2.Canny(conditioning_image, 100, 200)
        conditioning_image = conditioning_image[:, :, None]
        conditioning_image = np.concatenate([conditioning_image, conditioning_image, conditioning_image], axis=2)

        conditioning_image_as_pil = Image.fromarray(conditioning_image)

        conditioning_image = TF.to_tensor(conditioning_image)

    if training_config.training == "sdxl_controlnet" and training_config.controlnet_type == "inpainting" and training_config.controlnet_variant == "pre_encoded_controlnet_cond":
        if random.random() <= 0.25:
            conditioning_image_mask = np.ones((resolution, resolution), np.float32)
        else:
            conditioning_image_mask = random.choice([make_random_rectangle_mask, make_random_irregular_mask, make_outpainting_mask])(resolution, resolution)

        conditioning_image_mask = torch.from_numpy(conditioning_image_mask)

        conditioning_image_mask = conditioning_image_mask[None, :, :]

        conditioning_image = TF.to_tensor(image)

        if training_config.controlnet_variant == "pre_encoded_controlnet_cond":
            # where mask is 1, zero out the pixels. Note that this requires mask to be concattenated
            # with the mask so that the network knows the zeroed out pixels are from the mask and
            # are not just zero in the original image
            conditioning_image = conditioning_image * (conditioning_image_mask < 0.5)

            conditioning_image_as_pil = TF.to_pil_image(conditioning_image)

            conditioning_image = TF.normalize(conditioning_image, [0.5], [0.5])
        else:
            # Just zero out the pixels which will be masked
            conditioning_image_as_pil = TF.to_pil_image(conditioning_image * (conditioning_image_mask < 0.5))

            # where mask is set to 1, set to -1 "special" masked image pixel.
            # -1 is outside of the 0-1 range that the controlnet normalized
            # input is in.
            conditioning_image = conditioning_image * (conditioning_image_mask < 0.5) + -1.0 * (conditioning_image_mask > 0.5)

    return dict(conditioning_image=conditioning_image, conditioning_image_mask=conditioning_image_mask, conditioning_image_as_pil=conditioning_image_as_pil)
