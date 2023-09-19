import torch
from diffusers import (
    AutoencoderKL,
)
from PIL import Image
import torchvision.transforms.functional as TF


def main():
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    vae.to('cuda')
    
    image = Image.open("./validation_data/two_birds_on_branch.png")
    image = image.resize((1024, 1024))
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])
    image = image[None, :, :, :]
    image = image.to('cuda', torch.float16)

    image = vae(image, sample_posterior=True).sample

    image = (image * 0.5 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = image.permute(0, 2, 3, 1)
    image = image.cpu().numpy()[0]
    image = Image.fromarray(image)

    image.save('./out.png')

if __name__ == "__main__":
    main()
