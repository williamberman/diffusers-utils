import torch
from diffusers import StableDiffusionXLPipeline
from argparse import ArgumentParser
from PIL import Image
from typing import List


def main():
    args = ArgumentParser()
    args.add_argument("--prompt", required=True, type=str)
    args.add_argument("--num_images", required=True, type=int, default=1)
    args = args.parse_args()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    negative_prompt = "text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, distortion, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, symmetrical, duplicate, error, pattern, beginner, pixelated, fake, hyper, glitch, overexposed, high-contrast, bad-contrast"

    images: List[Image.Image] = pipe(
        args.prompt,
        num_images_per_prompt=args.num_images,
        negative_prompt=negative_prompt,
    ).images

    for i, image in enumerate(images):
        image.save(f"out_{i}.png")


if __name__ == "__main__":
    main()
