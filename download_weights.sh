#! /bin/bash

set -e


wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors -O weights/sdxl_unet.fp16.safetensors
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors -O weights/sdxl_unet.safetensors
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -O weights/sdxl_vae.fp16.safetensors
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors -O weights/sdxl_vae.safetensors
wget https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors -O weights/sdxl_vae_fp16_fix.safetensors

wget https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors -O weights/sdxl_controlnet_canny.safetensors
wget https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors -O weights/sdxl_controlnet_canny.fp16.safetensors

wget https://huggingface.co/williamberman/sdxl_controlnet_inpainting/resolve/main/sdxl_controlnet_inpaint_pre_encoded_controlnet_cond_checkpoint_200000.safetensors -O weights/sdxl_controlnet_inpaint_pre_encoded_controlnet_cond_checkpoint_200000.safetensors