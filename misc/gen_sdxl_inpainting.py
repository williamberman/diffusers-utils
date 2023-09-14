import torch
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
from diffusers.utils import load_image
from argparse import ArgumentParser
from PIL import Image
from typing import List


def main():
    args = ArgumentParser()
    args.add_argument("--prompt", required=True, type=str)
    args.add_argument("--image", required=True, type=str)
    args.add_argument("--mask", required=True, type=str)
    args.add_argument("--num_images", required=True, type=int, default=1)
    args.add_argument("--num_inference_steps", required=False, type=int, default=None)
    args = args.parse_args()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        vae=vae,
    )
    pipe.to("cuda")

    init_image = load_image(args.image).resize((1024, 1024)).convert("RGB")
    mask_image = load_image(args.mask).resize((1024, 1024)).convert("RGB")

    negative_prompt = "text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, distortion, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, symmetrical, duplicate, error, pattern, beginner, pixelated, fake, hyper, glitch, overexposed, high-contrast, bad-contrast"

    kwargs = {}

    if args.num_inference_steps is not None:
        kwargs["num_inference_steps"] = args.num_inference_steps

    images: List[Image.Image] = pipe(
        args.prompt,
        image=init_image,
        mask_image=mask_image,
        num_images_per_prompt=args.num_images,
        negative_prompt=negative_prompt,
        **kwargs,
    ).images

    for i, image in enumerate(images):
        image.save(f"out_{i}.png")


if __name__ == "__main__":
    main()
