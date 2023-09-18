import os

import safetensors.torch
import torch
from diffusers import ControlNetModel
from torch import nn

from blocks import ResnetBlock2D, Transformer2DModel, get_sinusoidal_embedding
from sdxl_controlnet import ControlNetOutput
from utils import load_safetensors_state_dict, maybe_ddp_module, zero_module


class SDXLControlNetPreEncodedControlnetCond(ControlNetModel):
    def __init__(self):
        super().__init__()

        # fmt: off

        encoder_hidden_states_dim = 2048

        # timesteps embedding:

        time_sinusoidal_embedding_dim = 320
        time_embedding_dim = 1280

        self.get_sinusoidal_timestep_embedding = lambda timesteps: get_sinusoidal_embedding(timesteps, time_sinusoidal_embedding_dim)

        self.time_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(time_sinusoidal_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # image size and crop coordinates conditioning embedding (i.e. micro conditioning):

        num_micro_conditioning_values = 6
        micro_conditioning_embedding_dim = 256
        additional_embedding_encoder_dim = 1280
        self.get_sinusoidal_micro_conditioning_embedding = lambda micro_conditioning: get_sinusoidal_embedding(micro_conditioning, micro_conditioning_embedding_dim)

        self.add_embedding = nn.ModuleDict(dict(
            linear_1=nn.Linear(additional_embedding_encoder_dim + num_micro_conditioning_values * micro_conditioning_embedding_dim, time_embedding_dim),
            act=nn.SiLU(),
            linear_2=nn.Linear(time_embedding_dim, time_embedding_dim),
        ))

        # actual unet blocks:

        # unet latents: 4 + 
        # control image latents: 4 + 
        # controlnet_mask: 1
        # = 9 channels
        self.conv_in = nn.Conv2d(9, 320, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            # 320 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 320, time_embedding_dim),
                    ResnetBlock2D(320, 320, time_embedding_dim),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 320 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(320, 640, time_embedding_dim),
                    ResnetBlock2D(640, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))]),
            )),
            # 640 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640, 1280, time_embedding_dim),
                    ResnetBlock2D(1280, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
            )),
        ])

        self.controlnet_down_blocks = nn.ModuleList([
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.controlnet_mid_block = zero_module(nn.Conv2d(1280, 1280, kernel_size=1))

        # fmt: on

    def forward(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale=1.0,
        added_cond_kwargs=None,
        return_dict=True,
        # for compatibility with diffusers
        guess_mode=None,
    ):
        batch_size = sample.shape[0]

        if len(timesteps.shape) == 0:
            timesteps = timesteps[None]

        if timesteps.shape[0] == 1 and timesteps.shape[0] != batch_size:
            timesteps = timesteps.expand(batch_size)

        timesteps = self.get_sinusoidal_timestep_embedding(timesteps)
        timesteps = timesteps.to(dtype=sample.dtype)
        timesteps = self.time_embedding["linear_1"](timesteps)
        timesteps = self.time_embedding["act"](timesteps)
        timesteps = self.time_embedding["linear_2"](timesteps)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(added_cond_kwargs["time_ids"])
        additional_conditioning = additional_conditioning.to(dtype=sample.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([added_cond_kwargs["text_embeds"], additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        timesteps = timesteps + additional_conditioning

        # NOTE: hack for dealing with CFG when called from diffusers pipeline
        if sample.shape[0] == 2 * controlnet_cond.shape[0]:
            controlnet_cond = torch.concat((controlnet_cond, controlnet_cond), dim=0)

        sample = torch.concat((sample, controlnet_cond), dim=1)

        sample = self.conv_in(sample)

        down_block_res_sample = conditioning_scale * self.controlnet_down_blocks[0](sample)
        down_block_res_samples = [down_block_res_sample]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                sample = resnet(sample, timesteps)

                if "attentions" in down_block:
                    sample = down_block["attentions"][i](sample, encoder_hidden_states)

                down_block_res_sample = conditioning_scale * self.controlnet_down_blocks[len(down_block_res_samples)](sample)
                down_block_res_samples.append(down_block_res_sample)

            if "downsamplers" in down_block:
                sample = down_block["downsamplers"][0]["conv"](sample)

                down_block_res_sample = conditioning_scale * self.controlnet_down_blocks[len(down_block_res_samples)](sample)
                down_block_res_samples.append(down_block_res_sample)

        sample = self.mid_block["resnets"][0](sample, timesteps)
        sample = self.mid_block["attentions"][0](sample, encoder_hidden_states)
        sample = self.mid_block["resnets"][1](sample, timesteps)

        mid_block_res_sample = conditioning_scale * self.controlnet_mid_block(sample)

        down_block_res_samples = tuple(down_block_res_samples)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample, None, None)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            add_to_down_block_inputs=None,
            add_to_output=None,
        )

    # methods to mimic diffusers

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, load_path):
        load_path = os.path.join(load_path, "diffusion_pytorch_model.safetensors")
        sd = load_safetensors_state_dict(load_path)
        controlnet = cls()
        controlnet.load_state_dict(sd)
        return controlnet

    def save_pretrained(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "diffusion_pytorch_model.safetensors")
        sd = {k: v.to("cpu") for k, v in self.state_dict().items()}
        safetensors.torch.save_file(sd, save_path)

    @classmethod
    def from_unet(cls, unet):
        unet = maybe_ddp_module(unet)

        controlnet = cls()

        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
        controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        conv_in_weight = unet.conv_in.state_dict()["weight"]
        padding = torch.zeros((320, 5, 3, 3), device=conv_in_weight.device, dtype=conv_in_weight.dtype)
        conv_in_weight = torch.concat((conv_in_weight, padding), dim=1)

        conv_in_bias = unet.conv_in.state_dict()["bias"]

        controlnet.conv_in.load_state_dict({"weight": conv_in_weight, "bias": conv_in_bias})

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet
