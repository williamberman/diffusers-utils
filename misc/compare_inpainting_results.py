from sdxl_controlnet import SDXLControlNet
from sdxl_controlnet_full import SDXLControlNetFull
from sdxl_controlnet_pre_encoded_controlnet_cond import SDXLControlNetPreEncodedControlnetCond
from sdxl_unet import SDXLUNet
import torch
from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline, StableDiffusionXLInpaintPipeline
from PIL import Image
import numpy as np
from masking import make_masked_image, masked_image_as_pil
from typing import List
import wandb
from utils import load_safetensors_state_dict
import torchvision.transforms.functional as TF
import os


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    wandb.init(
        project="sdxl_controlnet_inpaint_compare_results"
    )

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    vae.to('cuda')

    negative_prompt = "text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, distortion, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, symmetrical, duplicate, error, pattern, beginner, pixelated, fake, hyper, glitch, overexposed, high-contrast, bad-contrast"

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

        # TODO
        # ["./validation_data/house_in_snowy_mountains.png", "./validation_data/house_in_snowy_mountains_mask.png", "a house in the snowy mountains" + suffix],
    ]


    tables = [
        wandb.Table(columns=["model"] + [f"image {i}" for i in range(num_images_per_validation)])
        for _ in range(len(validation_data))
    ]

    validation_data_table = wandb.Table(columns=["prompt", "image", "mask", "masked_image"])

    masked_images = []
    masks = []
    prompts = []

    for image, mask_image, prompt in validation_data:
        row = [prompt]

        image = Image.open(image).convert('RGB').resize((1024, 1024))
        row.append(wandb.Image(image))
        
        mask = Image.open(mask_image).convert('L').resize((1024, 1024))
        row.append(wandb.Image(mask))
        mask = np.array(mask) / 255

        masked_image, _ = make_masked_image(image, return_type="controlnet_scaled_tensor", mask=mask)
        row.append(wandb.Image(masked_image_as_pil(masked_image)))
        masked_image = masked_image[None, :, :, :]
        masked_image = masked_image.to(device='cuda', dtype=torch.float16)

        mask = torch.from_numpy(mask)[None, None, :, :].to(device='cuda', dtype=torch.float16)
        masks += [mask] * num_images_per_validation

        masked_images += [masked_image] * num_images_per_validation

        prompts += [prompt] * num_images_per_validation

        validation_data_table.add_data(*row)

    wandb.log({"validation data": validation_data_table})

    masked_images = torch.concat(masked_images)

    vae_encoded_masked_images = vae.encode(masked_images).latent_dist.sample(generator=torch.Generator().manual_seed(0))
    vae_encoded_masked_images = vae_encoded_masked_images * vae.config.scaling_factor

    masks = torch.concat(masks)
    masks = TF.resize(masks, (128, 128))

    vae_encoded_masked_images = torch.concat([vae_encoded_masked_images, masks], dim=1)

    # first, do all that use the regular unet
    unet = SDXLUNet.load_fp16('cuda')

    def sdxl_controlnet_inpaint():
        controlnet = SDXLControlNet.from_pretrained("output/sdxl_controlnet_inpaint_9_16_resume_from_80000_batch_size_168/checkpoint-160000/controlnet")
        controlnet.to(dtype=torch.float16, device='cuda')

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            unet=unet,
            vae=vae,
            controlnet=controlnet,
        )
        pipe.to("cuda")

        idx = 0

        for prompts_, masked_images_ in zip(split_list(prompts, chunk_size), masked_images.split(chunk_size)):
            masked_images_ = torch.concat([masked_images_, masked_images_])

            images = pipe(
                prompts_,
                image=masked_images_,
                negative_prompt=[negative_prompt]*len(prompts_),
                generator=torch.Generator().manual_seed(0),
                height=1024,
                width=1024,
            ).images

            log_images("regular controlnet architecture", images[0:4], idx)
            idx += 1
            log_images("regular controlnet architecture", images[4:], idx)
            idx += 1


    def sdxl_controlnet_inpaint_full():
        controlnet = SDXLControlNetFull.from_pretrained("output/sdxl_controlnet_inpaint_full/checkpoint-52000/controlnet")
        controlnet.to(dtype=torch.float16, device='cuda')

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            unet=unet,
            vae=vae,
            controlnet=controlnet,
        )
        pipe.to("cuda")

        idx = 0

        for prompts_, masked_images_ in zip(split_list(prompts, chunk_size), masked_images.split(chunk_size)):
            masked_images_ = torch.concat([masked_images_, masked_images_])

            images = pipe(
                prompts_,
                image=masked_images_,
                negative_prompt=[negative_prompt]*len(prompts_),
                generator=torch.Generator().manual_seed(0),
                height=1024,
                width=1024,
            ).images

            log_images('"full" controlnet architecture', images[0:4], idx)
            idx += 1
            log_images('"full" controlnet architecture', images[4:], idx)
            idx += 1

    def sdxl_controlnet_inpaint_pre_encoded_controlnet_cond():
        checkpoint = get_latest_checkpoint("output/sdxl_controlnet_inpaint_pre_encoded_controlnet_cond")

        controlnet = SDXLControlNetPreEncodedControlnetCond.from_pretrained(os.path.join(checkpoint, "controlnet"))

        controlnet.to(dtype=torch.float16, device='cuda')

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            unet=unet,
            vae=vae,
            controlnet=controlnet,
        )
        pipe.to("cuda")

        idx = 0

        for prompts_, vae_encoded_masked_images_ in zip(split_list(prompts, chunk_size), vae_encoded_masked_images.split(chunk_size)):
            vae_encoded_masked_images_ = torch.concat([vae_encoded_masked_images_, vae_encoded_masked_images_])

            images = pipe(
                prompts_,
                image=vae_encoded_masked_images_,
                negative_prompt=[negative_prompt]*len(prompts_),
                generator=torch.Generator().manual_seed(0),
                height=1024,
                width=1024,
            ).images

            log_images('regular controlnet architecture with vae encoding control', images[0:4], idx)
            idx += 1
            log_images('regular controlnet architecture with vae encoding control', images[4:], idx)
            idx += 1

    # from now on we can mutate the unet weights

    def sdxl_controlnet_inpaint_train_base_unet():
        checkpoint = get_latest_checkpoint("output/sdxl_controlnet_inpaint_train_base_unet")

        controlnet = SDXLControlNet.from_pretrained(os.path.join(checkpoint, "controlnet"))

        controlnet.to(dtype=torch.float16, device='cuda')

        unet_state_dict = load_safetensors_state_dict(os.path.join(checkpoint, "unet.safetensors"))
        unet.up_blocks.load_state_dict(unet_state_dict)
        unet.up_blocks.to(dtype=torch.float16, device='cuda')

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            unet=unet,
            vae=vae,
            controlnet=controlnet,
        )
        pipe.to("cuda")

        idx = 0

        for prompts_, masked_images_ in zip(split_list(prompts, chunk_size), masked_images.split(chunk_size)):
            masked_images_ = torch.concat([masked_images_, masked_images_])

            images = pipe(
                prompts_,
                image=masked_images_,
                negative_prompt=[negative_prompt]*len(prompts_),
                generator=torch.Generator().manual_seed(0),
                height=1024,
                width=1024,
            ).images

            log_images('regular controlnet architecture + train unet up blocks', images[0:4], idx)
            idx += 1
            log_images('regular controlnet architecture + train unet up blocks', images[4:], idx)
            idx += 1

    def sdxl_controlnet_inpaint_pre_encoded_controlnet_cond_train_base_unet():
        checkpoint = get_latest_checkpoint("output/sdxl_controlnet_inpaint_pre_encoded_controlnet_cond_train_base_unet")
        controlnet = SDXLControlNetPreEncodedControlnetCond.from_pretrained(os.path.join(checkpoint, "controlnet"))
        controlnet.to(dtype=torch.float16, device='cuda')

        unet_state_dict = load_safetensors_state_dict(os.path.join(checkpoint, "unet.safetensors"))
        unet.up_blocks.load_state_dict(unet_state_dict)
        unet.up_blocks.to(dtype=torch.float16, device='cuda')

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            unet=unet,
            vae=vae,
            controlnet=controlnet,
        )
        pipe.to("cuda")

        idx = 0

        for prompts_, vae_encoded_masked_images_ in zip(split_list(prompts, chunk_size), vae_encoded_masked_images.split(chunk_size)):
            vae_encoded_masked_images_ = torch.concat([vae_encoded_masked_images_, vae_encoded_masked_images_])

            images = pipe(
                prompts_,
                image=vae_encoded_masked_images_,
                negative_prompt=[negative_prompt]*len(prompts_),
                generator=torch.Generator().manual_seed(0),
                height=1024,
                width=1024,
            ).images

            log_images('regular controlnet architecture with vae encoding control + train unet up blocks', images[0:4], idx)
            idx += 1
            log_images('regular controlnet architecture with vae encoding control + train unet up blocks', images[4:], idx)
            idx += 1

    pil_images = []
    pil_mask_images = []

    for image, mask_image, prompt in validation_data:
        image = Image.open(image).convert('RGB').resize((1024, 1024))
        pil_images += [image] * num_images_per_validation

        mask = Image.open(mask_image).convert('L').resize((1024, 1024))
        pil_mask_images += [mask] * num_images_per_validation


    def base_sdxl_inpainting():
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            vae=vae,
        )
        pipe.to("cuda")

        idx = 0

        for prompts_, pil_images_, pil_mask_images_ in zip(split_list(prompts, chunk_size), split_list(pil_images, chunk_size), split_list(pil_mask_images, chunk_size)):
            images: List[Image.Image] = pipe(
                prompts_,
                image=pil_images_,
                mask_image=pil_mask_images_,
                negative_prompt=[negative_prompt]*len(prompts_),
                generator=torch.Generator().manual_seed(0),
                height=1024,
                width=1024,
            ).images

            log_images("base sdxl inpainting", images[0:4], idx)
            idx += 1
            log_images("base sdxl inpainting", images[4:], idx)
            idx += 1

    def diffusers_finetuned_sdxl_inpainting_01():
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            vae=vae,
        )
        pipe.to("cuda")

        idx = 0

        for prompts_, pil_images_, pil_mask_images_ in zip(split_list(prompts, chunk_size), split_list(pil_images, chunk_size), split_list(pil_mask_images, chunk_size)):
            images: List[Image.Image] = pipe(
                prompts_,
                image=pil_images_,
                mask_image=pil_mask_images_,
                negative_prompt=[negative_prompt]*len(prompts_),
                generator=torch.Generator().manual_seed(0),
                height=1024,
                width=1024,
            ).images

            log_images("diffusers finetuned sdxl inpaint 0.1", images[0:4], idx)
            idx += 1
            log_images("diffusers finetuned sdxl inpaint 0.1", images[4:], idx)
            idx += 1


    def log_images(model_name, images, idx):
        row = [model_name]
        row += [wandb.Image(x) for x in images]
        tables[idx].add_data(*row)


    sdxl_controlnet_inpaint()
    sdxl_controlnet_inpaint_full()
    sdxl_controlnet_inpaint_pre_encoded_controlnet_cond()
    sdxl_controlnet_inpaint_train_base_unet()
    sdxl_controlnet_inpaint_pre_encoded_controlnet_cond_train_base_unet()
    base_sdxl_inpainting()
    diffusers_finetuned_sdxl_inpainting_01()

    for i in range(len(validation_data)):
        prompt = validation_data[i][2]
        table = tables[i]
        wandb.log({prompt: table})

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def get_latest_checkpoint(path):
    checkpoints = os.listdir(path)

    checkpoints = [x.split('-')[1] for x in checkpoints]

    checkpoints.sort()

    latest_checkpoint = checkpoints[-1]

    latest_checkpoint = os.path.join(path, f"checkpoint-{latest_checkpoint}")

    return latest_checkpoint

if __name__ == "__main__":
    main()