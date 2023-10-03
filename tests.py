import torch
from diffusers import (AutoencoderKL, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from diffusion import make_sigmas
from sdxl import (sdxl_diffusion_loop, sdxl_text_conditioning,
                  sdxl_tokenize_one, sdxl_tokenize_two)
from sdxl_models import AttentionMixin, SDXLUNet, SDXLVae

AttentionMixin.attention_implementation = "torch_2.0_scaled_dot_product"

device = "cuda"

text_encoder_one = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
text_encoder_one.to(device=device)

text_encoder_two = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16)
text_encoder_two.to(device=device)

vae = SDXLVae.load_fp16_fix(device=device)
vae.to(dtype=torch.float16)

vae_ = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
vae_.to(device=device)

unet = SDXLUNet.load_fp16(device=device)

unet_ = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16, variant="fp16")
unet_.to(device=device)

sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, unet=unet_, vae=vae_, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two
)
sdxl_pipe.to(device)
sdxl_pipe.set_progress_bar_config(disable=True)


def test_sdxl_vae():
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
    assert total_diff < 650  # TODO


def test_sdxl_unet():
    prompts = ["horse"]

    encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
        text_encoder_one,
        text_encoder_two,
        sdxl_tokenize_one(prompts).to(text_encoder_one.device),
        sdxl_tokenize_two(prompts).to(text_encoder_two.device),
    )
    encoder_hidden_states = encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)
    pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=unet.dtype, device=unet.device)

    micro_conditioning = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.long, device=unet.device)

    x_t = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet.dtype, device=unet.device, generator=torch.Generator(device).manual_seed(0))

    t = torch.tensor(500, dtype=torch.long, device=device)

    unet_output = unet(
        x_t=x_t,
        t=t,
        encoder_hidden_states=encoder_hidden_states,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        micro_conditioning=micro_conditioning,
    )

    expected_unet_output = unet_(
        sample=x_t, timestep=t, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs={"time_ids": micro_conditioning, "text_embeds": pooled_encoder_hidden_states}
    ).sample

    total_diff = (expected_unet_output.float() - unet_output.float()).abs().sum()

    assert total_diff < 12


def test_text_to_image():
    sigmas = make_sigmas(device=unet.device)

    x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=unet.dtype, device=unet.device, generator=torch.Generator(device).manual_seed(0))
    x_T_ = x_T * ((sigmas.max() ** 2 + 1) ** 0.5)

    out = sdxl_diffusion_loop(["horse"], unet=unet, text_encoder_one=text_encoder_one, text_encoder_two=text_encoder_two, generator=torch.Generator(device).manual_seed(0), x_T=x_T_)
    vae.output_tensor_to_pil(vae.decode(out))[0]
    sdxl_pipe(prompt="horse", latents=x_T)


if __name__ == "__main__":
    test_sdxl_vae()
    test_sdxl_unet()
    test_text_to_image()
