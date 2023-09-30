import itertools
import os
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional as TF
import wandb
import webdataset as wds
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizerFast)

from diffusion import (default_num_train_timesteps, make_sigmas,
                       ode_solver_diffusion_loop)
from sdxl_models import (SDXLAdapter, SDXLControlNet, SDXLControlNetFull,
                         SDXLControlNetPreEncodedControlnetCond, SDXLUNet,
                         SDXLVae)
from training_config import Config
from utils import maybe_ddp_device, maybe_ddp_dtype, maybe_ddp_module


class SDXLTraining:
    text_encoder_one: CLIPTextModel
    text_encoder_two: CLIPTextModelWithProjection
    vae: SDXLVae
    sigmas: torch.Tensor
    unet: SDXLUNet
    adapter: Optional[SDXLAdapter]
    controlnet: Optional[Union[SDXLControlNet, SDXLControlNetFull]]

    train_unet: bool
    train_unet_up_blocks: bool

    mixed_precision: Optional[torch.dtype]
    timestep_sampling: Literal["uniform", "cubic"]

    validation_images_logged: bool
    log_validation_input_images_every_time: bool

    get_sdxl_conditioning_images: Callable[[Image.Image], Dict[str, Any]]

    @classmethod
    def from_training_config(cls, training_config: Config, get_sdxl_conditioning_images, device):
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
            get_sdxl_conditioning_images=get_sdxl_conditioning_images,
        )

    def __init__(
        self,
        device,
        train_unet,
        get_sdxl_conditioning_images,
        train_unet_up_blocks=False,
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
                self.controlnet = controlnet_cls.from_unet(self.unet)
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

        self.get_sdxl_conditioning_images = get_sdxl_conditioning_images

        self.train_unet = train_unet
        self.train_unet_up_blocks = train_unet_up_blocks

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

                conditioning_images = self.get_sdxl_conditioning_images(validation_image)

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

    def parameters(self):
        if self.train_unet:
            return self.unet.parameters()

        if self.controlnet is not None and self.train_unet_up_blocks:
            return itertools.chain(self.controlnet.parameters(), self.unet.up_blocks.parameters())

        if self.controlnet is not None:
            return self.controlnet.parameters()

        if self.adapter is not None:
            return self.adapter.parameters()

        assert False

    def save(self, save_to):
        if self.train_unet:
            safetensors.torch.save_file(self.unet.module.state_dict(), os.path.join(save_to, "unet.safetensors"))

        if self.controlnet is not None and self.train_unet_up_blocks:
            safetensors.torch.save_file(self.controlnet.module.state_dict(), os.path.join(save_to, "controlnet.safetensors"))
            safetensors.torch.save_file(self.unet.module.up_blocks.state_dict(), os.path.join(save_to, "unet.safetensors"))

        if self.controlnet is not None:
            safetensors.torch.save_file(self.controlnet.module.state_dict(), os.path.join(save_to, "controlnet.safetensors"))

        if self.adapter is not None:
            safetensors.torch.save_file(self.adapter.module.state_dict(), os.path.join(save_to, "adapter.safetensors"))


def get_sdxl_dataset(train_shards: str, shuffle_buffer_size: int, batch_size: int, proportion_empty_prompts: float, get_sdxl_conditioning_images=None):
    dataset = (
        wds.WebDataset(
            train_shards,
            resampled=True,
            handler=wds.ignore_and_continue,
        )
        .shuffle(shuffle_buffer_size)
        .decode("pil", handler=wds.ignore_and_continue)
        .rename(
            image="jpg;png;jpeg;webp",
            text="text;txt;caption",
            metadata="json",
            handler=wds.warn_and_continue,
        )
        .map(lambda d: make_sample(d, proportion_empty_prompts=proportion_empty_prompts, get_sdxl_conditioning_images=get_sdxl_conditioning_images))
        .select(lambda sample: "conditioning_image" not in sample or sample["conditioning_image"] is not None)
    )

    dataset = dataset.batched(batch_size, partial=False, collation_fn=default_collate)

    return dataset


@torch.no_grad()
def make_sample(d, proportion_empty_prompts, get_sdxl_conditioning_images=None):
    image = d["image"]
    metadata = d["metadata"]

    if random.random() < proportion_empty_prompts:
        text = ""
    else:
        text = d["text"]

    c_top, c_left, _, _ = get_random_crop_params([image.height, image.width], [1024, 1024])

    original_width = int(metadata.get("original_width", 0.0))
    original_height = int(metadata.get("original_height", 0.0))

    micro_conditioning = torch.tensor([original_width, original_height, c_top, c_left, 1024, 1024])

    text_input_ids_one = sdxl_tokenize_one(text)

    text_input_ids_two = sdxl_tokenize_two(text)

    image = image.convert("RGB")

    image = TF.resize(
        image,
        1024,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )

    image = TF.crop(
        image,
        c_top,
        c_left,
        1024,
        1024,
    )

    sample = {
        "micro_conditioning": micro_conditioning,
        "text_input_ids_one": text_input_ids_one,
        "text_input_ids_two": text_input_ids_two,
        "image": TF.normalize(TF.to_tensor(image), [0.5], [0.5]),
    }

    if get_sdxl_conditioning_images is not None:
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


class GetSDXLConditioningImages:
    training: Literal["sdxl_adapter", "sdxl_unet", "sdxl_controlnet"]
    adapter_type: Optional[Literal["openpose"]]
    controlnet_type: Optional[Literal["canny", "inpainting"]]
    controlnet_variant: Literal["default", "full", "pre_encoded_controlnet_cond"]

    @classmethod
    def from_training_config(cls, training_config: Config):
        return cls(training=training_config.training, controlnet_type=training_config.controlnet_type, controlnet_variant=training_config.controlnet_variant, adapter_type=training_config.adapter_type)

    def __init__(
        self,
        training,
        controlnet_type,
        controlnet_variant,
        adapter_type,
    ):
        self.training = training
        self.controlnet_type = controlnet_type
        self.controlnet_variant = controlnet_variant
        self.adapter_type = adapter_type

        if training == "sdxl_adapter" and self.adapter_type == "openpose":
            from controlnet_aux import OpenposeDetector

            self.open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    def __call__(self, image):
        resolution = image.width

        if self.training == "sdxl_adapter" and self.adapter_type == "openpose":
            conditioning_image = self.open_pose(image, detect_resolution=resolution, image_resolution=resolution, return_pil=False)

            if (conditioning_image == 0).all():
                return None, None

            conditioning_image_as_pil = Image.fromarray(conditioning_image)

            conditioning_image = TF.to_tensor(conditioning_image)

        if self.training == "sdxl_controlnet" and self.controlnet_type == "canny":
            import cv2

            conditioning_image = np.array(image)
            conditioning_image = cv2.Canny(conditioning_image, 100, 200)
            conditioning_image = conditioning_image[:, :, None]
            conditioning_image = np.concatenate([conditioning_image, conditioning_image, conditioning_image], axis=2)

            conditioning_image_as_pil = Image.fromarray(conditioning_image)

            conditioning_image = TF.to_tensor(conditioning_image)

        if self.training == "sdxl_controlnet" and self.controlnet_type == "inpainting" and self.controlnet_variant == "pre_encoded_controlnet_cond":
            if random.random() <= 0.25:
                conditioning_image_mask = np.ones((resolution, resolution), np.float32)
            else:
                conditioning_image_mask = random.choice([make_random_rectangle_mask, make_random_irregular_mask, make_outpainting_mask])(resolution, resolution)

            conditioning_image_mask = torch.from_numpy(conditioning_image_mask)

            conditioning_image_mask = conditioning_image_mask[None, :, :]

            conditioning_image = TF.to_tensor(image)

            if self.controlnet_variant == "pre_encoded_controlnet_cond":
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


# TODO: would be nice to just call a function from a tokenizers https://github.com/huggingface/tokenizers
# i.e. afaik tokenizing shouldn't require holding any state

tokenizer_one = CLIPTokenizerFast.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")

tokenizer_two = CLIPTokenizerFast.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")


def sdxl_tokenize_one(prompts):
    return tokenizer_one(
        prompts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]


def sdxl_tokenize_two(prompts):
    return tokenizer_two(
        prompts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]


def sdxl_text_conditioning(text_encoder_one, text_encoder_two, text_input_ids_one, text_input_ids_two):
    prompt_embeds_1 = text_encoder_one(
        text_input_ids_one,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_1 = prompt_embeds_1.view(prompt_embeds_1.shape[0], prompt_embeds_1.shape[1], -1)

    prompt_embeds_2 = text_encoder_two(
        text_input_ids_two,
        output_hidden_states=True,
    )

    pooled_encoder_hidden_states = prompt_embeds_2[0]

    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    prompt_embeds_2 = prompt_embeds_2.view(prompt_embeds_2.shape[0], prompt_embeds_2.shape[1], -1)

    encoder_hidden_states = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)

    return encoder_hidden_states, pooled_encoder_hidden_states


def make_random_rectangle_mask(
    height,
    width,
    margin=10,
    bbox_min_size=100,
    bbox_max_size=512,
    min_times=1,
    max_times=2,
):
    mask = np.zeros((height, width), np.float32)

    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)

        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)

        mask[start_y : start_y + box_height, start_x : start_x + box_width] = 1

    return mask


def make_random_irregular_mask(height, width, max_angle=4, max_len=60, max_width=256, min_times=1, max_times=2):
    import cv2

    mask = np.zeros((height, width), np.float32)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)

        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)

            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle

            length = 10 + np.random.randint(max_len)

            brush_w = 5 + np.random.randint(max_width)

            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)

            choice = random.randint(0, 2)

            if choice == 0:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif choice == 1:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1.0, thickness=-1)
            elif choice == 2:
                radius = brush_w // 2
                mask[
                    start_y - radius : start_y + radius,
                    start_x - radius : start_x + radius,
                ] = 1
            else:
                assert False

            start_x, start_y = end_x, end_y

    return mask


def make_outpainting_mask(height, width, probs=[0.5, 0.5, 0.5, 0.5]):
    mask = np.zeros((height, width), np.float32)
    at_least_one_mask_applied = False

    coords = [
        [(0, 0), (1, get_padding(height))],
        [(0, 0), (get_padding(width), 1)],
        [(0, 1 - get_padding(height)), (1, 1)],
        [(1 - get_padding(width), 0), (1, 1)],
    ]

    for pp, coord in zip(probs, coords):
        if np.random.random() < pp:
            at_least_one_mask_applied = True
            mask = apply_padding(mask=mask, coord=coord)

    if not at_least_one_mask_applied:
        idx = np.random.choice(range(len(coords)), p=np.array(probs) / sum(probs))
        mask = apply_padding(mask=mask, coord=coords[idx])

    return mask


def get_padding(size, min_padding_percent=0.04, max_padding_percent=0.5):
    n1 = int(min_padding_percent * size)
    n2 = int(max_padding_percent * size)
    return np.random.randint(n1, n2) / size


def apply_padding(mask, coord):
    height, width = mask.shape

    mask[
        int(coord[0][0] * height) : int(coord[1][0] * height),
        int(coord[0][1] * width) : int(coord[1][1] * width),
    ] = 1

    return mask


@torch.no_grad()
def sdxl_diffusion_loop(
    prompts, images, unet, text_encoder_one, text_encoder_two, controlnet=None, adapter=None, sigmas=None, timesteps=None, x_T=None, micro_conditioning=None, guidance_scale=5.0, generator=None
):
    encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
        text_encoder_one,
        text_encoder_two,
        sdxl_tokenize_one(prompts).to(text_encoder_one.device),
        sdxl_tokenize_two(prompts).to(text_encoder_two.device),
    )

    if x_T is None:
        x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=torch.float32, device=unet.device, generator=generator)
        x_T = x_T * ((sigmas.max() ** 2 + 1) ** 0.5)

    if sigmas is None:
        sigmas = make_sigmas()

    if timesteps is None:
        timesteps = torch.linspace(0, sigmas.numel(), 50, dtype=torch.long, device=unet.device)

    if micro_conditioning is None:
        micro_conditioning = torch.tensor([1024, 1024, 0, 0, 1024, 1024], dtype=torch.long, device=unet.device)

    if adapter is not None:
        down_block_additional_residuals = adapter(images)
    else:
        down_block_additional_residuals = None

    if controlnet is not None:
        controlnet_cond = images
    else:
        controlnet_cond = None

    eps_theta = lambda x_t, t, sigma: sdxl_eps_theta(
        x_t=x_t,
        t=t,
        sigma=sigma,
        unet=unet,
        encoder_hidden_states=encoder_hidden_states,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        guidance_scale=guidance_scale,
        controlnet=controlnet,
        controlnet_cond=controlnet_cond,
        down_block_additional_residuals=down_block_additional_residuals,
    )

    x_0 = ode_solver_diffusion_loop(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

    return x_0


@torch.no_grad()
def sdxl_eps_theta(
    x_t,
    t,
    sigma,
    unet,
    encoder_hidden_states,
    pooled_encoder_hidden_states,
    micro_conditioning,
    guidance_scale,
    controlnet=None,
    controlnet_cond=None,
    down_block_additional_residuals=None,
):
    # TODO - how does this not effect the ode we are solving
    scaled_x_t = x_t / ((sigma**2 + 1) ** 0.5)

    if guidance_scale > 1.0:
        scaled_x_t = torch.concat([scaled_x_t, scaled_x_t])

    if controlnet is not None:
        controlnet_out = controlnet(
            x_t=scaled_x_t,
            t=t,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )

        down_block_additional_residuals = controlnet_out["down_block_res_samples"]
        mid_block_additional_residual = controlnet_out["mid_block_res_sample"]
        add_to_down_block_inputs = controlnet_out.get("add_to_down_block_inputs", None)
        add_to_output = controlnet_out.get("add_to_output", None)
    else:
        mid_block_additional_residual = None
        add_to_down_block_inputs = None
        add_to_output = None

    eps_hat = unet(
        x_t=scaled_x_t,
        t=t,
        encoder_hidden_states=encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        down_block_additional_residuals=down_block_additional_residuals,
        mid_block_additional_residual=mid_block_additional_residual,
        add_to_down_block_inputs=add_to_down_block_inputs,
        add_to_output=add_to_output,
    )

    if guidance_scale > 1.0:
        eps_hat_uncond, eps_hat = eps_hat.chunk(2)

        eps_hat = eps_hat_uncond + guidance_scale * (eps_hat - eps_hat_uncond)

    return eps_hat
