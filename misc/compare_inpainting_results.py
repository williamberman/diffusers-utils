import os
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
import wandb
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from sdxl import (
    get_controlnet_inpainting_conditioning_image,
    get_controlnet_pre_encoded_controlnet_inpainting_conditioning_image,
    known_negative_prompt, sdxl_diffusion_loop)
from sdxl_models import (SDXLControlNet, SDXLControlNetFull,
                         SDXLControlNetPreEncodedControlnetCond, SDXLUNet,
                         SDXLVae)

torch.set_grad_enabled(False)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    wandb.init(project="sdxl_controlnet_inpaint_compare_results")

    vae = SDXLVae.load_fp16_fix("cuda")

    text_encoder_one = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
    text_encoder_one.to(device="cuda")

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16)
    text_encoder_two.to(device="cuda")

    unet = SDXLUNet.load_fp16("cuda")

    unet_train_base_unet = SDXLUNet.load_fp16("cuda", os.path.join(checkpoint_train_base_unet, "unet.safetensors"))
    unet_train_base_unet.up_blocks.to(dtype=torch.float16)

    unet_pre_encoded_controlnet_cond_train_base_unet = SDXLUNet.load_fp16("cuda", os.path.join(checkpoint_pre_encoded_controlnet_cond_train_base_unet, "unet.safetensors"))
    unet_pre_encoded_controlnet_cond_train_base_unet.up_blocks.to(dtype=torch.float16)

    unet_diffusers_finetuned_sdxl_inpainting = SDXLUNet.load(hf_hub_download("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "unet.safetensors"), "cuda")
    unet_diffusers_finetuned_sdxl_inpainting.to(torch.float16)

    controlnet_inpaint = SDXLControlNet.load("output/sdxl_controlnet_inpaint_9_16_resume_from_80000_batch_size_168/checkpoint-160000/controlnet", device="cuda")
    controlnet_inpaint.to(dtype=torch.float16)

    controlnet_full = SDXLControlNetFull.load("output/sdxl_controlnet_inpaint_full/checkpoint-52000/controlnet", device="cuda")
    controlnet_full.to(dtype=torch.float16)

    checkpoint_pre_encoded_controlnet_cond = get_latest_checkpoint("output/sdxl_controlnet_inpaint_pre_encoded_controlnet_cond")
    controlnet_pre_encoded_controlnet_cond = SDXLControlNetPreEncodedControlnetCond.load(os.path.join(checkpoint_pre_encoded_controlnet_cond, "controlnet"), device="cuda")
    controlnet_pre_encoded_controlnet_cond.to(dtype=torch.float16)

    checkpoint_train_base_unet = get_latest_checkpoint("output/sdxl_controlnet_inpaint_train_base_unet")
    controlnet_train_base_unet = SDXLControlNet.load(os.path.join(checkpoint_train_base_unet, "controlnet"), device="cuda")
    controlnet_train_base_unet.to(dtype=torch.float16)

    checkpoint_pre_encoded_controlnet_cond_train_base_unet = get_latest_checkpoint("output/sdxl_controlnet_inpaint_pre_encoded_controlnet_cond_train_base_unet")
    controlnet_pre_encoded_controlnet_cond_train_base_unet = SDXLControlNetPreEncodedControlnetCond.load(os.path.join(checkpoint_pre_encoded_controlnet_cond_train_base_unet, "controlnet"), "cuda")
    controlnet_pre_encoded_controlnet_cond_train_base_unet.to(dtype=torch.float16)

    num_images_per_validation = 4

    chunk_size = num_images_per_validation * 2

    suffix = ", high quality, 4k"

    validation_data = [
        ["./validation_data/person_jumping.png", "./validation_data/person_jumping_mask.png", "superman" + suffix],
        ["./validation_data/dog_sitting_on_bench.png", "./validation_data/dog_sitting_on_bench_mask.png", "a cat sitting on a bench" + suffix],
        ["./validation_data/dog_sitting_on_bench.png", "./validation_data/dog_sitting_on_bench_mask.png", "a lion sitting on a bench" + suffix],
        ["./validation_data/dog_sitting_on_bench.png", "./validation_data/dog_sitting_on_bench_mask.png", "a green lion sitting on a bench" + suffix],
        ["./validation_data/two_birds_on_branch.png", "./validation_data/two_birds_on_branch_mask.png", "two birds on a branch" + suffix],
        ["./validation_data/couple_sitting_on_bench_infront_of_lake.png", "./validation_data/couple_sitting_on_bench_infront_of_lake_mask.png", "couple sitting on bench infront of lake" + suffix],
        ["./validation_data/house_in_snowy_mountains.png", "./validation_data/house_in_snowy_mountains_mask.png", "a house in the snowy mountains" + suffix],
    ]

    tables = [wandb.Table(columns=["model"] + [f"image {i}" for i in range(num_images_per_validation)]) for _ in range(len(validation_data))]

    validation_data_table = wandb.Table(columns=["prompt", "image", "mask", "masked_image"])

    prompts = []
    conditioning_images = []
    conditioning_images_pre_encoded_controlnet_cond = []
    x_T_only_finetuned_base_model = []  # TODO - add to these

    for image, mask_image, prompt in validation_data:
        prompts += [prompt] * num_images_per_validation

        image = Image.open(image).convert("RGB").resize((1024, 1024))

        mask_as_pil = Image.open(mask_image).convert("L").resize((1024, 1024))
        mask = torch.from_numpy(np.array(mask_as_pil) / 255)[None, :, :]

        out = get_controlnet_inpainting_conditioning_image(image, conditioning_image_mask=mask)
        conditioning_images += [out["conditioning_image"]] * num_images_per_validation
        masked_image_as_pil = out["masked_image_as_pil"]

        out = get_controlnet_pre_encoded_controlnet_inpainting_conditioning_image(image, conditioning_image_mask=mask)
        conditioning_image = vae.encode(out["conditioning_image"][None, :, :, :].to(device=vae.device, dtype=vae.dtype)).to(torch.float16)
        mask = TF.resize(mask, (1024 // 8, 1024 // 8))[None, :, :, :].to(dtype=conditioning_image.dtype)
        conditioning_image = torch.concat((conditioning_image, mask))
        conditioning_images_pre_encoded_controlnet_cond += [conditioning_image] * num_images_per_validation

        row = [prompt]
        row.append(wandb.Image(image))
        row.append(wandb.Image(mask_as_pil))
        row.append(wandb.Image(masked_image_as_pil))
        validation_data_table.add_data(*row)

    wandb.log({"validation data": validation_data_table})

    conditioning_images = torch.concat(conditioning_images)
    conditioning_images_pre_encoded_controlnet_cond = torch.concat(conditioning_images_pre_encoded_controlnet_cond)
    x_T_only_finetuned_base_model = torch.concat(x_T_only_finetuned_base_model)

    for conditioning_images_, unet_, controlnet, log_name in [
        [conditioning_images, unet, controlnet_inpaint, "regular controlnet architecture"],
        [conditioning_images, unet, controlnet_full, '"full" controlnet architecture'],
        [conditioning_images_pre_encoded_controlnet_cond, unet, controlnet_pre_encoded_controlnet_cond, "regular controlnet architecture with vae encoding control"],
        # last training run degenerated:
        # [conditioning_images, unet_train_base_unet, controlnet_train_base_unet, "regular controlnet architecture + train unet up blocks"],
        [
            conditioning_images_pre_encoded_controlnet_cond,
            unet_pre_encoded_controlnet_cond_train_base_unet,
            controlnet_pre_encoded_controlnet_cond_train_base_unet,
            "regular controlnet architecture with vae encoding control + train unet up blocks",
        ],
        [],
    ]:
        idx = 0

        for prompts_, conditioning_images__ in zip(split_list(prompts, chunk_size), conditioning_images_.split(chunk_size)):
            images = sdxl_diffusion_loop(
                prompts=prompts_,
                negative_prompts=[known_negative_prompt] * len(prompts_),
                unet=unet_,
                text_encoder_one=text_encoder_one,
                text_encoder_two=text_encoder_two,
                controlnet=controlnet,
                images=conditioning_images__,
            )

            images = vae.output_tensor_to_pil(vae.decode(images))

            log_images(log_name, images[0:4], idx)
            idx += 1

            if len(images) > 4:
                log_images(log_name, images[4:], idx)
                idx += 1

    idx = 0

    for prompts_, x_T in zip(split_list(prompts, chunk_size), x_T_only_finetuned_base_model.chunk(chunk_size)):
        images = sdxl_diffusion_loop(
            prompts=prompts_,
            negative_prompts=[known_negative_prompt] * len(prompts_),
            unet=unet_diffusers_finetuned_sdxl_inpainting,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            x_T=x_T,
        )

        log_images("diffusers finetuned sdxl inpaint 0.1", images[0:4], idx)
        idx += 1

        if len(images) > 4:
            log_images("diffusers finetuned sdxl inpaint 0.1", images[4:], idx)
            idx += 1

    def log_images(model_name, images, idx):
        row = [model_name]
        row += [wandb.Image(x) for x in images]
        tables[idx].add_data(*row)

    for i in range(len(validation_data)):
        prompt = validation_data[i][2]
        table = tables[i]
        wandb.log({prompt: table})


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def get_latest_checkpoint(path):
    checkpoints = os.listdir(path)

    checkpoints = [x.split("-")[1] for x in checkpoints]

    checkpoints.sort()

    latest_checkpoint = checkpoints[-1]

    latest_checkpoint = os.path.join(path, f"checkpoint-{latest_checkpoint}")

    return latest_checkpoint


if __name__ == "__main__":
    main()
