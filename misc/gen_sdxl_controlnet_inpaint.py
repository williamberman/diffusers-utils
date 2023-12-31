import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    UNet2DConditionModel
)
from argparse import ArgumentParser
from PIL import Image
from typing import List
from masking import make_masked_image, masked_image_as_pil
from blocks import SDXLControlNet, SDXLUNet
from utils import load_safetensors_state_dict


def main():
    args = ArgumentParser()
    args.add_argument("--controlnet_path", required=True, type=str)
    args = args.parse_args()

    unet_ = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")

    unet = SDXLUNet()
    unet.load_state_dict(unet_.state_dict())
    unet.to(dtype=torch.float16)

    del unet_

    # controlnet = ControlNetModel.from_pretrained(
    #     args.controlnet_path, torch_dtype=torch.float16
    # )
    controlnet = SDXLControlNet()
    controlnet.load_state_dict(load_safetensors_state_dict(args.controlnet_path))
    controlnet.to(dtype=torch.float16)

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

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

    validation_images = [
        # "./validation_data/person_jumping.png",
        # "./validation_data/bright_room_with_chair.png",
        # "./validation_data/couple_sitting_on_bench_infront_of_lake.png",
        # "./validation_data/house_in_snowy_mountains.png",
        # "./validation_data/hq_woman.png",
        # "./validation_data/man_skating.png",
        # "./validation_data/painting_of_rice_paddies.png",
        # "./validation_data/tornado_at_sea.png",
        "./validation_data/two_birds_on_branch.png",
    ]

    validation_prompts = [
        # "superman",
        # "bright room with chair",
        # "couple sitting on bench infront of lake",
        # "house in snowy mountains",
        # "a beautiful woman",
        # "a man skating in brooklyn",
        # "a painting of people working the rice paddies",
        # "a tornado in ohio",
        "two birds on a branch",
    ]

    negative_prompt = "text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, distortion, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, symmetrical, duplicate, error, pattern, beginner, pixelated, fake, hyper, glitch, overexposed, high-contrast, bad-contrast"

    formatted_validation_images = []

    for validation_image in validation_images:
        validation_image = Image.open(validation_image)
        validation_image = validation_image.convert("RGB")
        validation_image = validation_image.resize(
            (1024, 1024)
        )

        # TODO - because we can't get a PIL back here to pass to both the
        # wandb lob and the pipeline, this is messy+redundant. Is there
        # a better way to do this?

        validation_image = make_masked_image(
            validation_image, return_type="controlnet_scaled_tensor"
        )

        masked_image_as_pil(validation_image).save('./masked.png')

        validation_image = validation_image[None, :, :, :]

        validation_image = validation_image.to(device="cuda", dtype=torch.float16)

        formatted_validation_images.append(validation_image)

    formatted_validation_images = torch.concat(formatted_validation_images)
    formatted_validation_images = torch.concat([formatted_validation_images, formatted_validation_images])

    images: List[Image.Image] = pipe(
        validation_prompts,
        image=formatted_validation_images,
        negative_prompt=[negative_prompt]*len(validation_prompts),
        callback=lambda i, t, latents: save_intermediate_image(vae, i, t, latents),
    ).images

    for i, image in enumerate(images):
        image.save(f"out_{i}.png")


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
