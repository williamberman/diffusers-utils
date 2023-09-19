import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from argparse import ArgumentParser


def main():
    args = ArgumentParser()
    args.add_argument("--image", required=True, type=str)
    args.add_argument("--prompt", required=True, type=str)
    args = args.parse_args()

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe = pipe.to("cuda")

    init_image = load_image(args.image).convert("RGB")
    image = pipe(args.prompt, image=init_image).images[0]
    image.save('./out.png')

if __name__ == "__main__":
    main()