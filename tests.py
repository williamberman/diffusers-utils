import torch
from diffusers import AutoencoderKL
from PIL import Image

from sdxl_models import SDXLVae

device = "cuda"


def test_training():
    ...


def test_save_checkpoint():
    ...


def test_sdxl_dataset():
    ...


def test_sdxl_diffusion_loop():
    ...


def test_gen_sdxl_simplified_interface():
    ...


def test_sdxl_vae():
    vae = SDXLVae.load_fp16_fix()
    vae.to(device=device, dtype=torch.float16)

    vae_ = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae_.to(device=device)

    image = Image.open("./validation_data/two_birds_on_branch.png")
    image = image.convert("RGB")
    image = image.resize((1024, 1024))
    image = vae.input_pil_to_tensor(image)
    image = image.to(dtype=vae.dtype, device=vae.device)

    encoder_output = vae.encode(image, generator=torch.Generator(device).manual_seed(0))

    expected_encoder_output = vae_.encode(image).latent_dist.sample(generator=torch.Generator(device).manual_seed(0))
    expected_encoder_output = expected_encoder_output * vae_.config.scaling_factor

    total_diff = (expected_encoder_output.float() - encoder_output.float()).abs().sum()
    assert total_diff == 0

    decoder_output = vae.decode(encoder_output)
    expected_decoder_output = vae_.decode(expected_encoder_output / vae_.config.scaling_factor).sample

    total_diff = (expected_decoder_output.float() - decoder_output.float()).abs().sum()
    assert total_diff < 650


def test_sdxl_unet():
    ...


def test_sdxl_controlnet():
    ...


def test_sdxl_pre_encoded_controlnet_cond():
    ...


def test_sdxl_controlnet_full():
    ...


def test_sdxl_adapter():
    ...


def test_ode_solver_diffusion_loop():
    ...


if __name__ == "__main__":
    test_sdxl_vae()
