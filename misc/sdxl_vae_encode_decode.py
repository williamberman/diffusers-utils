from PIL import Image
from sdxl_models import SDXLVae

def main():
    vae = SDXLVae.load_fp16_fix(device='cuda')
    
    image = Image.open("./validation_data/two_birds_on_branch.png")
    image = image.convert('RGB')
    image = image.resize((1024, 1024))
    image = vae.input_pil_to_tensor(image)
    image = image.to(device=vae.device, dtype=vae.dtype)

    image = vae(image)
    image = vae.output_tensor_to_pil(image)[0]

    image.save('./out.png')

if __name__ == "__main__":
    main()
