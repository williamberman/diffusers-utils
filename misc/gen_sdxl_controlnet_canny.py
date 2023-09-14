import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    ControlNetModel,
)
from diffusers.utils import load_image
from argparse import ArgumentParser
from PIL import Image
from typing import List
import numpy as np
import cv2


def main():
    args = ArgumentParser()
    args.add_argument("--prompt", required=True, type=str)
    args.add_argument("--image", required=True, type=str)
    args.add_argument("--num_images", required=True, type=int, default=1)
    args.add_argument("--controlnet_path", required=True, type=str)
    args = args.parse_args()

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path, torch_dtype=torch.float16
    )

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        vae=vae,
        controlnet=controlnet,
    )
    pipe.to("cuda")

    image = load_image(args.image).resize((1024, 1024)).convert("RGB")
    image = make_canny_conditioning(image)

    negative_prompt = "text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, distortion, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, symmetrical, duplicate, error, pattern, beginner, pixelated, fake, hyper, glitch, overexposed, high-contrast, bad-contrast"

    images: List[Image.Image] = pipe(
        args.prompt,
        image=image,
        num_images_per_prompt=args.num_images,
        negative_prompt=negative_prompt,
        callback=lambda i, t, latents: save_intermediate_image(vae, i, t, latents),
    ).images

    for i, image in enumerate(images):
        image.save(f"out_{i}.png")


def make_canny_conditioning(image):
    controlnet_image = np.array(image)
    controlnet_image = cv2.Canny(controlnet_image, 100, 200)
    controlnet_image = controlnet_image[:, :, None]
    controlnet_image = np.concatenate(
        [controlnet_image, controlnet_image, controlnet_image], axis=2
    )
    controlnet_image = Image.fromarray(controlnet_image)
    return controlnet_image


def save_intermediate_image(vae, i, t, latents):
    latents = latents[0:1, :, :, :]

    latents = latents / vae.config.scaling_factor
    image: torch.Tensor = vae.decode(latents).sample
    image = (image * 0.5 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = image.permute(0, 2, 3, 1)
    image = image.cpu().numpy()[0]
    image = Image.fromarray(image)
    image.save(f"./out_step_{t.item()}.png")


if __name__ == "__main__":
    main()
