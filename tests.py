import numpy as np
import scipy
import torch
import torchvision.transforms.functional as TF
from diffusers import (AutoencoderKL, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from diffusion import make_sigmas
from sdxl import (sdxl_diffusion_loop, sdxl_text_conditioning,
                  sdxl_tokenize_one, sdxl_tokenize_two)
from sdxl_models import (SDXLControlNet,
                         SDXLControlNetPreEncodedControlnetCond, SDXLUNet,
                         SDXLVae, set_attention_implementation)

set_attention_implementation("torch_2.0_scaled_dot_product")
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

device = "cuda"
dtype = torch.float32

if dtype == torch.float32:
    text_encoder_one = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
    text_encoder_one.to(device=device)

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2")
    text_encoder_two.to(device=device)

    vae = SDXLVae.load_fp16_fix(device=device)

    vae_ = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    vae_.to(device=device)

    unet = SDXLUNet.load_fp32(device=device)

    unet_ = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
    unet_.to(device=device)

    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet_, vae=vae_, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two)
    sdxl_pipe.to(device)
    sdxl_pipe.set_progress_bar_config(disable=True)
elif dtype == torch.float16:
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
        "stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16, unet=unet_, vae=vae_, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two
    )
    sdxl_pipe.to(device)
    sdxl_pipe.set_progress_bar_config(disable=True)
else:
    assert False


@torch.no_grad()
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
    assert total_diff == 0


@torch.no_grad()
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

    t = torch.tensor([500], dtype=torch.long, device=device)

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

    assert total_diff == 0


@torch.no_grad()
def test_text_to_image():
    sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
    # fmt: off
    timesteps = torch.tensor([1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501, 521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761, 781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981], dtype=torch.long, device=device)
    # fmt: on

    x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=dtype, device=unet_.device, generator=torch.Generator(device).manual_seed(0))
    x_T_ = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

    out = sdxl_diffusion_loop(
        ["horse"],
        unet=unet,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        generator=torch.Generator(device).manual_seed(0),
        x_T=x_T_,
        timesteps=timesteps,
        sigmas=sigmas,
    )
    out = vae.output_tensor_to_pil(vae.decode(out))[0]
    out = np.array(out).astype(np.int32)

    expected_out = sdxl_pipe(prompt="horse", latents=x_T).images[0]
    expected_out = np.array(expected_out).astype(np.int32)

    diff = np.abs(out - expected_out).flatten()
    diff.sort()

    assert scipy.stats.mode(diff).mode == 1
    assert diff.mean() < 1


@torch.no_grad()
def test_controlnet():
    import cv2

    if dtype == torch.float32:
        controlnet = SDXLControlNet.load("./weights/sdxl_controlnet_canny.safetensors", device=device)
    elif dtype == torch.float16:
        controlnet = SDXLControlNet.load("./weights/sdxl_controlnet_canny.fp16.safetensors", device=device)
    else:
        assert False

    sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
    # fmt: off
    timesteps = torch.tensor([1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501, 521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761, 781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981], dtype=torch.long, device=device)
    # fmt: on

    x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=dtype, device=unet_.device, generator=torch.Generator(device).manual_seed(0))
    x_T_ = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

    image = Image.open("./validation_data/bright_room_with_chair.png").convert("RGB").resize((1024, 1024))
    image = cv2.Canny(np.array(image), 100, 200)[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = TF.to_tensor(image)[None, :, :, :].to(device=device, dtype=controlnet.dtype)

    out = sdxl_diffusion_loop(
        ["a beautiful room"],
        unet=unet,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        generator=torch.Generator(device).manual_seed(0),
        x_T=x_T_,
        timesteps=timesteps,
        sigmas=sigmas,
        controlnet=controlnet,
        images=image,
    )
    out = vae.output_tensor_to_pil(vae.decode(out))[0]


def test_controlnet_pre_encoded_controlnet_cond():
    controlnet = SDXLControlNetPreEncodedControlnetCond.load("./weights/sdxl_controlnet_inpaint_pre_encoded_controlnet_cond_checkpoint_200000.safetensors", device=device)
    controlnet.to(dtype=dtype)

    sigmas = make_sigmas(device=unet.device).to(dtype=unet.dtype)
    # fmt: off
    timesteps = torch.tensor([1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501, 521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761, 781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981], dtype=torch.long, device=device)
    # fmt: on

    x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=dtype, device=unet_.device, generator=torch.Generator(device).manual_seed(0))
    x_T_ = x_T * ((sigmas[timesteps[-1]] ** 2 + 1) ** 0.5)

    image = TF.to_tensor(Image.open("./validation_data/dog_sitting_on_bench.png").convert("RGB").resize((1024, 1024)))
    mask = TF.to_tensor(Image.open("./validation_data/dog_sitting_on_bench_mask.png").convert("L").resize((1024, 1024)))
    image = image * (mask < 0.5)
    image = TF.normalize(image, [0.5], [0.5])
    image = vae.encode(image[None, :, :, :].to(dtype=vae.dtype, device=vae.device)).to(dtype=controlnet.dtype, device=controlnet.device)
    mask = TF.resize(mask, (1024 // 8, 1024 // 8))[None, :, :, :].to(dtype=image.dtype, device=image.device)
    image = torch.concat((image, mask), dim=1)

    out = sdxl_diffusion_loop(
        ["a green lion sitting on a bench"],
        unet=unet,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        generator=torch.Generator(device).manual_seed(0),
        x_T=x_T_,
        timesteps=timesteps,
        sigmas=sigmas,
        controlnet=controlnet,
        images=image,
    )
    out = vae.output_tensor_to_pil(vae.decode(out))[0]


if __name__ == "__main__":
    test_sdxl_vae()
    test_sdxl_unet()
    test_text_to_image()
    test_controlnet()
    test_controlnet_pre_encoded_controlnet_cond()
