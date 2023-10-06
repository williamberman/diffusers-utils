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
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizerFast)

import wandb
from diffusion import (default_num_train_timesteps,
                       euler_ode_solver_diffusion_loop, make_sigmas)
from sdxl_models import (SDXLAdapter, SDXLControlNet, SDXLControlNetFull,
                         SDXLControlNetPreEncodedControlnetCond, SDXLUNet,
                         SDXLVae)


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
            if isinstance(self.unet, DDP):
                unet_dtype = self.unet.module.dtype
                unet_device = self.unet.module.device
            else:
                unet_dtype = self.unet.dtype
                unet_device = self.unet.device

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
                controlnet_device = self.controlnet.module.device
                controlnet_dtype = self.controlnet.module.dtype
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
        if isinstance(self.unet, DDP):
            unet = self.unet.module
            unet.eval()
            unet_set_to_eval = True
        else:
            unet = self.unet
            unet_set_to_eval = False

        if self.adapter is not None:
            adapter = self.adapter.module
            adapter.eval()
        else:
            adapter = None

        if self.controlnet is not None:
            controlnet = self.controlnet.module
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

                    x_0 = self.vae.decode(x_0)
                    x_0 = self.vae.output_tensor_to_pil(x_0)[0]

                    output_validation_images.append(wandb.Image(x_0, caption=validation_prompt))

        wandb.log({"validation": output_validation_images}, step=step)

        if unet_set_to_eval:
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
    import webdataset as wds

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

    text_input_ids_one = sdxl_tokenize_one(text)[0]

    text_input_ids_two = sdxl_tokenize_two(text)[0]

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
        "image": SDXLVae.input_pil_to_tensor(image),
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


def get_adapter_openpose_conditioning_image(image, open_pose):
    resolution = image.width

    conditioning_image = open_pose(image, detect_resolution=resolution, image_resolution=resolution, return_pil=False)

    if (conditioning_image == 0).all():
        return None, None

    conditioning_image_as_pil = Image.fromarray(conditioning_image)

    conditioning_image = TF.to_tensor(conditioning_image)

    return dict(conditioning_image=conditioning_image, conditioning_image_as_pil=conditioning_image_as_pil)


def get_controlnet_canny_conditioning_image(image):
    import cv2

    conditioning_image = np.array(image)
    conditioning_image = cv2.Canny(conditioning_image, 100, 200)
    conditioning_image = conditioning_image[:, :, None]
    conditioning_image = np.concatenate([conditioning_image, conditioning_image, conditioning_image], axis=2)

    conditioning_image_as_pil = Image.fromarray(conditioning_image)

    conditioning_image = TF.to_tensor(conditioning_image)

    return dict(conditioning_image=conditioning_image, conditioning_image_as_pil=conditioning_image_as_pil)


def get_controlnet_pre_encoded_controlnet_inpainting_conditioning_image(image, conditioning_image_mask):
    resolution = image.width

    if conditioning_image_mask is None:
        if random.random() <= 0.25:
            conditioning_image_mask = np.ones((resolution, resolution), np.float32)
        else:
            conditioning_image_mask = random.choice([make_random_rectangle_mask, make_random_irregular_mask, make_outpainting_mask])(resolution, resolution)

        conditioning_image_mask = torch.from_numpy(conditioning_image_mask)

        conditioning_image_mask = conditioning_image_mask[None, :, :]

    conditioning_image = TF.to_tensor(image)

    # where mask is 1, zero out the pixels. Note that this requires mask to be concattenated
    # with the mask so that the network knows the zeroed out pixels are from the mask and
    # are not just zero in the original image
    conditioning_image = conditioning_image * (conditioning_image_mask < 0.5)

    conditioning_image_as_pil = TF.to_pil_image(conditioning_image)

    conditioning_image = TF.normalize(conditioning_image, [0.5], [0.5])

    return dict(conditioning_image=conditioning_image, conditioning_image_mask=conditioning_image_mask, conditioning_image_as_pil=conditioning_image_as_pil)


def get_controlnet_inpainting_conditioning_image(image, conditioning_image_mask):
    resolution = image.width

    if conditioning_image_mask is None:
        if random.random() <= 0.25:
            conditioning_image_mask = np.ones((resolution, resolution), np.float32)
        else:
            conditioning_image_mask = random.choice([make_random_rectangle_mask, make_random_irregular_mask, make_outpainting_mask])(resolution, resolution)

        conditioning_image_mask = torch.from_numpy(conditioning_image_mask)

        conditioning_image_mask = conditioning_image_mask[None, :, :]

    conditioning_image = TF.to_tensor(image)

    # Just zero out the pixels which will be masked
    conditioning_image_as_pil = TF.to_pil_image(conditioning_image * (conditioning_image_mask < 0.5))

    # where mask is set to 1, set to -1 "special" masked image pixel.
    # -1 is outside of the 0-1 range that the controlnet normalized
    # input is in.
    conditioning_image = conditioning_image * (conditioning_image_mask < 0.5) + -1.0 * (conditioning_image_mask >= 0.5)

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
    ).input_ids


def sdxl_tokenize_two(prompts):
    return tokenizer_two(
        prompts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


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
    prompts: Union[str, List[str]],
    unet,
    text_encoder_one,
    text_encoder_two,
    images=None,
    controlnet=None,
    adapter=None,
    sigmas=None,
    timesteps=None,
    x_T=None,
    micro_conditioning=None,
    guidance_scale=5.0,
    generator=None,
    negative_prompts=None,
    diffusion_loop=euler_ode_solver_diffusion_loop,
):
    if isinstance(prompts, str):
        prompts = [prompts]

    batch_size = len(prompts)

    if negative_prompts is not None and guidance_scale > 1.0:
        prompts += negative_prompts

    encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
        text_encoder_one,
        text_encoder_two,
        sdxl_tokenize_one(prompts).to(text_encoder_one.device),
        sdxl_tokenize_two(prompts).to(text_encoder_two.device),
    )
    encoder_hidden_states = encoder_hidden_states.to(unet.dtype)
    pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(unet.dtype)

    if guidance_scale > 1.0:
        if negative_prompts is None:
            negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            negative_pooled_encoder_hidden_states = torch.zeros_like(pooled_encoder_hidden_states)
        else:
            encoder_hidden_states, negative_encoder_hidden_states = torch.chunk(encoder_hidden_states, 2)
            pooled_encoder_hidden_states, negative_pooled_encoder_hidden_states = torch.chunk(pooled_encoder_hidden_states, 2)
    else:
        negative_encoder_hidden_states = None
        negative_pooled_encoder_hidden_states = None

    if sigmas is None:
        sigmas = make_sigmas(device=unet.device)

    if timesteps is None:
        timesteps = torch.linspace(0, sigmas.numel() - 1, 50, dtype=torch.long, device=unet.device)

    if x_T is None:
        x_T = torch.randn((batch_size, 4, 1024 // 8, 1024 // 8), dtype=unet.dtype, device=unet.device, generator=generator)
        x_T = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

    if micro_conditioning is None:
        micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet.device)
        micro_conditioning = micro_conditioning.expand(batch_size, -1)

    if adapter is not None:
        down_block_additional_residuals = adapter(images.to(dtype=adapter.dtype, device=adapter.device))
    else:
        down_block_additional_residuals = None

    if controlnet is not None:
        controlnet_cond = images.to(dtype=controlnet.dtype, device=controlnet.device)
    else:
        controlnet_cond = None

    eps_theta = lambda *args, **kwargs: sdxl_eps_theta(
        *args,
        **kwargs,
        unet=unet,
        encoder_hidden_states=encoder_hidden_states,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        negative_encoder_hidden_states=negative_encoder_hidden_states,
        negative_pooled_encoder_hidden_states=negative_pooled_encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        guidance_scale=guidance_scale,
        controlnet=controlnet,
        controlnet_cond=controlnet_cond,
        down_block_additional_residuals=down_block_additional_residuals,
    )

    x_0 = diffusion_loop(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

    return x_0


@torch.no_grad()
def sdxl_eps_theta(
    x_t,
    t,
    sigma,
    unet,
    encoder_hidden_states,
    pooled_encoder_hidden_states,
    negative_encoder_hidden_states,
    negative_pooled_encoder_hidden_states,
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

        encoder_hidden_states = torch.concat((encoder_hidden_states, negative_encoder_hidden_states))
        pooled_encoder_hidden_states = torch.concat((pooled_encoder_hidden_states, negative_pooled_encoder_hidden_states))

        micro_conditioning = torch.concat([micro_conditioning, micro_conditioning])

        if controlnet_cond is not None:
            controlnet_cond = torch.concat([controlnet_cond, controlnet_cond])

    if controlnet is not None:
        controlnet_out = controlnet(
            x_t=scaled_x_t.to(controlnet.dtype),
            t=t,
            encoder_hidden_states=encoder_hidden_states.to(controlnet.dtype),
            micro_conditioning=micro_conditioning.to(controlnet.dtype),
            pooled_encoder_hidden_states=pooled_encoder_hidden_states.to(controlnet.dtype),
            controlnet_cond=controlnet_cond,
        )

        down_block_additional_residuals = [x.to(unet.dtype) for x in controlnet_out["down_block_res_samples"]]
        mid_block_additional_residual = controlnet_out["mid_block_res_sample"].to(unet.dtype)
        add_to_down_block_inputs = controlnet_out.get("add_to_down_block_inputs", None)
        if add_to_down_block_inputs is not None:
            add_to_down_block_inputs = [x.to(unet.dtype) for x in add_to_down_block_inputs]
        add_to_output = controlnet_out.get("add_to_output", None)
        if add_to_output is not None:
            add_to_output = add_to_output.to(unet.dtype)
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
        eps_hat, eps_hat_uncond = eps_hat.chunk(2)

        eps_hat = eps_hat_uncond + guidance_scale * (eps_hat - eps_hat_uncond)

    return eps_hat


known_negative_prompt = "text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, distortion, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, symmetrical, duplicate, error, pattern, beginner, pixelated, fake, hyper, glitch, overexposed, high-contrast, bad-contrast"

if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument("--prompts", required=True, type=str, nargs="+")
    args.add_argument("--negative_prompts", required=False, type=str, nargs="+")
    args.add_argument("--use_known_negative_prompt", action="store_true")
    args.add_argument("--num_images_per_prompt", required=True, type=int, default=1)
    args.add_argument("--num_inference_steps", required=False, type=int, default=50)
    args.add_argument("--images", required=False, type=str, default=None, nargs="+")
    args.add_argument("--masks", required=False, type=str, default=None, nargs="+")
    args.add_argument("--controlnet_checkpoint", required=False, type=str, default=None)
    args.add_argument("--controlnet", required=False, choices=["SDXLControlNet", "SDXLControlNetFull", "SDXLControNetPreEncodedControlnetCond"], default=None)
    args.add_argument("--adapter_checkpoint", required=False, type=str, default=None)
    args.add_argument("--device", required=False, default=None)
    args.add_argument("--dtype", required=False, default="fp16", choices=["fp16", "fp32"])
    args.add_argument("--guidance_scale", required=False, default=5.0, type=float)
    args.add_argument("--seed", required=False, type=int)
    args = args.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    if args.dtype == "fp16":
        dtype = torch.float16

        text_encoder_one = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
        text_encoder_one.to(device=device)

        text_encoder_two = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16)
        text_encoder_two.to(device=device)

        vae = SDXLVae.load_fp16_fix(device=device)
        vae.to(torch.float16)

        unet = SDXLUNet.load_fp16(device=device)
    elif args.dtype == "fp32":
        dtype = torch.float32

        text_encoder_one = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
        text_encoder_one.to(device=device)

        text_encoder_two = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2")
        text_encoder_two.to(device=device)

        vae = SDXLVae.load_fp16_fix(device=device)

        unet = SDXLUNet.load_fp32(device=device)
    else:
        assert False

    if args.controlnet == "SDXLControlNet":
        controlnet = SDXLControlNet.load(args.controlnet_checkpoint, device=device)
        controlnet.to(dtype)
    elif args.controlnet == "SDXLControlNetFull":
        controlnet = SDXLControlNetFull.load(args.controlnet_checkpoint, device=device)
        controlnet.to(dtype)
    elif args.controlnet == "SDXLControlNetPreEncodedControlnetCond":
        controlnet = SDXLControlNetPreEncodedControlnetCond.load(args.controlnet_checkpoint, device=device)
        controlnet.to(dtype)
    else:
        controlnet = None

    if args.adapter_checkpoint is not None:
        adapter = SDXLAdapter.load(args.adapter_checkpoint, device=device)
        adapter.to(dtype)
    else:
        adapter = None

    sigmas = make_sigmas(device=device).to(unet.dtype)

    timesteps = torch.linspace(0, sigmas.numel() - 1, args.num_inference_steps, dtype=torch.long, device=unet.device)

    prompts = []
    for prompt in args.prompts:
        prompts += [prompt] * args.num_images_per_prompt

    if args.use_known_negative_prompt:
        args.negative_prompts = [known_negative_prompt]

    if args.negative_prompts is None:
        negative_prompts = None
    elif len(args.negative_prompts) == 1:
        negative_prompts = args.negative_prompts * len(prompts)
    elif len(args.negative_prompts) == len(args.prompts):
        negative_prompts = []
        for negative_prompt in args.negative_prompts:
            negative_prompts += [negative_prompt] * args.num_images_per_prompt
    else:
        assert False

    if args.images is not None:
        images = []

        for image_idx, image in enumerate(args.images):
            image = Image.open(image)
            image = image.convert("RGB")
            image = image.resize((1024, 1024))
            image = TF.to_tensor(image)

            if args.masks is not None:
                mask = args.masks[image_idx]
                mask = Image.open(mask)
                mask = mask.convert("L")
                mask = mask.resize((1024, 1024))
                mask = TF.to_tensor(mask)

                if isinstance(controlnet, SDXLControlNetPreEncodedControlnetCond):
                    image = image * (mask < 0.5)
                    image = TF.normalize(image, [0.5], [0.5])
                    image = vae.encode(image[None, :, :, :].to(dtype=vae.dtype, device=vae.device)).to(dtype=controlnet.dtype, device=controlnet.device)
                    mask = TF.resize(mask, (1024 // 8, 1024 // 8))[None, :, :, :].to(dtype=image.dtype, device=image.device)
                    image = torch.concat((image, mask), dim=1)
                else:
                    image = (image * (mask < 0.5) + -1.0 * (mask >= 0.5)).to(dtype=dtype, device=device)
                    image = image[None, :, :, :]

            images += [image] * args.num_images_per_prompt

        images = torch.concat(images)
    else:
        images = None

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device).manual_seed(args.seed)

    images = sdxl_diffusion_loop(
        prompts=prompts,
        unet=unet,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        images=images,
        controlnet=controlnet,
        adapter=adapter,
        sigmas=sigmas,
        timesteps=timesteps,
        guidance_scale=args.guidance_scale,
        negative_prompts=negative_prompts,
        generator=generator,
    )

    images = vae.output_tensor_to_pil(vae.decode(images))

    for i, image in enumerate(images):
        image.save(f"out_{i}.png")
