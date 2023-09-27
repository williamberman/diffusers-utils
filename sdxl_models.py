from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from utils import (Attention, ModelUtils, ResnetBlock2D, Transformer2DModel,
                   get_sinusoidal_embedding, maybe_ddp_module, zero_module)

vae_scaling_factor = 0.13025


class SDXLVae(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        self.encoder = nn.ModuleDict(dict(
            # 3 -> 128
            conv_in=nn.Conv2d(3, 128, kernel_size=3, padding=1),

            down_blocks=nn.ModuleList([
                # 128 -> 128
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(128, 128, eps=1e-6), ResnetBlock2D(128, 128, eps=1e-6)]),
                    downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)))]),
                )),
                # 128 -> 256
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(128, 256, eps=1e-6), ResnetBlock2D(256, 256, eps=1e-6)]),
                    downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)))]),
                )),
                # 256 -> 512
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(256, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
                    downsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)))]),
                )),
                # 512 -> 512
                nn.ModuleDict(dict(resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]))),
            ]),

            # 512 -> 512
            mid_block=nn.ModuleDict(dict(
                attentions=nn.ModuleList([Attention(512, 512, qkv_bias=True)]),
                resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
            )),

            # 512 -> 8
            conv_norm_out=nn.GroupNorm(32, 512, eps=1e-06),
            conv_act=nn.SiLU(),
            conv_out=nn.Conv2d(512, 8, kernel_size=3, padding=1)
        ))

        # 8 -> 8
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)

        # 8 -> 4 from sampling mean and std

        # 4 -> 4
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

        self.decoder = nn.ModuleDict(dict(
            # 4 -> 512
            conv_in=nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # 512 -> 512
            mid_block=nn.ModuleDict(dict(
                attentions=nn.ModuleList([Attention(512, 512, qkv_bias=True)]),
                resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
            )),

            up_blocks=nn.ModuleList([
                # 512 -> 512
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
                    upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)))]),
                )),

                # 512 -> 512
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6), ResnetBlock2D(512, 512, eps=1e-6)]),
                    upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)))]),
                )),

                # 512 -> 256
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(512, 256, eps=1e-6), ResnetBlock2D(256, 256, eps=1e-6), ResnetBlock2D(256, 256, eps=1e-6)]),
                    upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)))]),
                )),

                # 256 -> 128
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList([ResnetBlock2D(256, 128, eps=1e-6), ResnetBlock2D(128, 128, eps=1e-6), ResnetBlock2D(128, 128, eps=1e-6)]),
                )),
            ]),

            # 128 -> 3
            conv_norm_out=nn.GroupNorm(32, 128, eps=1e-06),
            conv_act=nn.SiLU(),
            conv_out=nn.Conv2d(128, 3, kernel_size=3, padding=1)
        ))

        # fmt: on

    def encode(self, x, generator=None):
        h = x

        h = self.encoder["conv_in"](h)

        for down_block in self.encoder["down_blocks"]:
            for resnet in down_block["resnets"]:
                h = resnet(h)

            if "downsamplers" in down_block:
                h = down_block["downsamplers"][0]["conv"](h)

        h = self.encoder["mid_block"]["resnets"][0](h)
        h = self.encoder["mid_block"]["attentions"][0](h)
        h = self.encoder["mid_block"]["resnets"][1](h)

        h = self.encoder["conv_norm_out"](h)
        h = self.encoder["conv_act"](h)
        h = self.encoder["conv_out"](h)

        mean, logvar = self.quant_conv(h).chunk(2, dim=1)

        logvar = torch.clamp(logvar, -30.0, 20.0)

        std = torch.exp(0.5 * logvar)

        z = mean + torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator) * std

        z = z * vae_scaling_factor

        return z

    def decode(self, z):
        z = z / vae_scaling_factor

        h = z

        h = self.post_quant_conv(h)

        h = self.decoder["mid_block"]["resnets"][0](h)
        h = self.decoder["mid_block"]["attentions"][0](h)
        h = self.decoder["mid_block"]["resnets"][1](h)

        for up_block in self.encoder["up_blocks"]:
            for resnet in up_block["resnets"]:
                h = resnet(h)

            if "upsamplers" in up_block:
                h = up_block["upsamplers"][0]["conv"](h)

        x_pred = ((x_pred * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1)

        x_pred = x_pred.permute(0, 2, 3, 1).cpu().numpy()

        x_pred = [Image.fromarray(x) for x in x_pred]

        return x_pred

    @classmethod
    def load_fp32(cls, device=None):
        return cls.load("./weights/sdxl_vae.safetensors", device=device)

    @classmethod
    def load_fp16(cls, device=None):
        return cls.load("./weights/sdxl_vae.fp16.safetensors", device=device)

    @classmethod
    def load_fp16_fix(cls, device=None):
        return cls.load("./weights/sdxl_vae_fp16_fix.safetensors", device=device)


class SDXLUNet(nn.Module, ModelUtils):
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

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

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

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.up_blocks = nn.ModuleList([
            # 1280 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 640, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(1280, 1280, kernel_size=3, padding=1)))]),
            )),
            # 1280 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 320, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, padding=1)))]),
            )),
            # 640 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                ]),
            ))
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        add_to_down_block_inputs: Optional[List[torch.Tensor]] = None,
        add_to_output: Optional[torch.Tensor] = None,
    ):
        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        hidden_state = self.conv_in(hidden_state)

        residuals = [hidden_state]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                if add_to_down_block_inputs is not None:
                    hidden_state = hidden_state + add_to_down_block_inputs.pop(0)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][i](hidden_state, encoder_hidden_states)

                residuals.append(hidden_state)

            if "downsamplers" in down_block:
                if add_to_down_block_inputs is not None:
                    hidden_state = hidden_state + add_to_down_block_inputs.pop(0)

                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                residuals.append(hidden_state)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        if mid_block_additional_residual is not None:
            hidden_state = hidden_state + mid_block_additional_residual

        for up_block in self.up_blocks:
            for i, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                if down_block_additional_residuals is not None:
                    residual = residual + down_block_additional_residuals.pop()

                hidden_state = torch.concat([hidden_state, residual], dim=1)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in up_block:
                    hidden_state = up_block["attentions"][i](hidden_state, encoder_hidden_states)

            if "upsamplers" in up_block:
                hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
                hidden_state = up_block["upsamplers"][0]["conv"](hidden_state)

        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)

        if add_to_output is not None:
            hidden_state = hidden_state + add_to_output

        eps_hat = hidden_state

        return eps_hat

    @classmethod
    def load_fp32(cls, device=None):
        return cls.load("./weights/sdxl_unet.safetensors", device=device)

    @classmethod
    def load_fp16(cls, device=None):
        return cls.load("./weights/sdxl_unet.fp16.safetensors", device=device)


class SDXLControlNet(nn.Module, ModelUtils):
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

        # controlnet cond embedding:
        self.controlnet_cond_embedding = nn.ModuleDict(dict(
            conv_in=nn.Conv2d(3, 16, kernel_size=3, padding=1),
            blocks=nn.ModuleList([
                # 16 -> 32
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
                # 32 -> 96
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 96, kernel_size=3, padding=1, stride=2),
                # 96 -> 256
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.Conv2d(96, 256, kernel_size=3, padding=1, stride=2),
            ]),
            conv_out=zero_module(nn.Conv2d(256, 320, kernel_size=3, padding=1)),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

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
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        controlnet_cond,
    ):
        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        controlnet_cond = self.controlnet_cond_embedding["conv_in"](controlnet_cond)
        controlnet_cond = F.silu(controlnet_cond)

        for block in self.controlnet_cond_embedding["blocks"]:
            controlnet_cond = F.silu(block(controlnet_cond))

        controlnet_cond = self.controlnet_cond_embedding["conv_out"](controlnet_cond)

        hidden_state = self.conv_in(hidden_state)

        hidden_state = hidden_state + controlnet_cond

        down_block_res_sample = self.controlnet_down_blocks[0](hidden_state)
        down_block_res_samples = [down_block_res_sample]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][i](hidden_state, encoder_hidden_states)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        mid_block_res_sample = self.controlnet_mid_block(hidden_state)

        return dict(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )

    @classmethod
    def from_unet(cls, unet):
        unet = maybe_ddp_module(unet)

        controlnet = cls()

        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
        controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet


class SDXLControlNetPreEncodedControlnetCond(nn.Module, ModelUtils):
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
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        controlnet_cond,
    ):
        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        hidden_state = torch.concat((hidden_state, controlnet_cond), dim=1)

        hidden_state = self.conv_in(hidden_state)

        down_block_res_sample = self.controlnet_down_blocks[0](hidden_state)
        down_block_res_samples = [down_block_res_sample]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][i](hidden_state, encoder_hidden_states)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                down_block_res_sample = self.controlnet_down_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.append(down_block_res_sample)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        mid_block_res_sample = self.controlnet_mid_block(hidden_state)

        return dict(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )

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


class SDXLControlNetFull(nn.Module, ModelUtils):
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

        # controlnet cond embedding:
        self.controlnet_cond_embedding = nn.ModuleDict(dict(
            conv_in=nn.Conv2d(3, 16, kernel_size=3, padding=1),
            blocks=nn.ModuleList([
                # 16 -> 32
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
                # 32 -> 96
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 96, kernel_size=3, padding=1, stride=2),
                # 96 -> 256
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.Conv2d(96, 256, kernel_size=3, padding=1, stride=2),
            ]),
            conv_out=zero_module(nn.Conv2d(256, 320, kernel_size=3, padding=1)),
        ))

        # actual unet blocks:

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)

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
        ])

        self.mid_block = nn.ModuleDict(dict(
            resnets=nn.ModuleList([
                ResnetBlock2D(1280, 1280, time_embedding_dim),
                ResnetBlock2D(1280, 1280, time_embedding_dim),
            ]),
            attentions=nn.ModuleList([Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10)]),
        ))

        self.controlnet_mid_block = zero_module(nn.Conv2d(1280, 1280, kernel_size=1))

        self.up_blocks = nn.ModuleList([
            # 1280 -> 1280
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 1280, 1280, time_embedding_dim),
                    ResnetBlock2D(1280 + 640, 1280, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                    Transformer2DModel(1280, encoder_hidden_states_dim, num_transformer_blocks=10),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(1280, 1280, kernel_size=3, padding=1)))]),
            )),
            # 1280 -> 640
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(1280 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 640, 640, time_embedding_dim),
                    ResnetBlock2D(640 + 320, 640, time_embedding_dim),
                ]),
                attentions=nn.ModuleList([
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                    Transformer2DModel(640, encoder_hidden_states_dim, num_transformer_blocks=2),
                ]),
                upsamplers=nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv2d(640, 640, kernel_size=3, padding=1)))]),
            )),
            # 640 -> 320
            nn.ModuleDict(dict(
                resnets=nn.ModuleList([
                    ResnetBlock2D(640 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                    ResnetBlock2D(320 + 320, 320, time_embedding_dim),
                ]),
            ))
        ])

        # take the output of transformer(resnet(hidden_states)) and project it to
        # the number of residual channels for the same block
        self.controlnet_up_blocks = nn.ModuleList([
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 1280, kernel_size=1)),
            zero_module(nn.Conv2d(1280, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 640, kernel_size=1)),
            zero_module(nn.Conv2d(640, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
            zero_module(nn.Conv2d(320, 320, kernel_size=1)),
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

        self.controlnet_conv_out = zero_module(nn.Conv2d(4, 4, kernel_size=1))

        # fmt: on

    def forward(
        self,
        x_t,
        t,
        encoder_hidden_states,
        micro_conditioning,
        pooled_encoder_hidden_states,
        controlnet_cond,
    ):
        hidden_state = x_t

        t = self.get_sinusoidal_timestep_embedding(t)
        t = t.to(dtype=hidden_state.dtype)
        t = self.time_embedding["linear_1"](t)
        t = self.time_embedding["act"](t)
        t = self.time_embedding["linear_2"](t)

        additional_conditioning = self.get_sinusoidal_micro_conditioning_embedding(micro_conditioning)
        additional_conditioning = additional_conditioning.to(dtype=hidden_state.dtype)
        additional_conditioning = additional_conditioning.flatten(1)
        additional_conditioning = torch.concat([pooled_encoder_hidden_states, additional_conditioning], dim=-1)
        additional_conditioning = self.add_embedding["linear_1"](additional_conditioning)
        additional_conditioning = self.add_embedding["act"](additional_conditioning)
        additional_conditioning = self.add_embedding["linear_2"](additional_conditioning)

        t = t + additional_conditioning

        controlnet_cond = self.controlnet_cond_embedding["conv_in"](controlnet_cond)
        controlnet_cond = F.silu(controlnet_cond)

        for block in self.controlnet_cond_embedding["blocks"]:
            controlnet_cond = F.silu(block(controlnet_cond))

        controlnet_cond = self.controlnet_cond_embedding["conv_out"](controlnet_cond)

        hidden_state = self.conv_in(hidden_state)

        hidden_state = hidden_state + controlnet_cond

        residuals = [hidden_state]

        add_to_down_block_input = self.controlnet_down_blocks[0](hidden_state)
        add_to_down_block_inputs = [add_to_down_block_input]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                hidden_state = resnet(hidden_state, t)

                if "attentions" in down_block:
                    hidden_state = down_block["attentions"][i](hidden_state, encoder_hidden_states)

                if len(add_to_down_block_inputs) < len(self.controlnet_down_blocks):
                    add_to_down_block_input = self.controlnet_down_blocks[len(add_to_down_block_inputs)](hidden_state)
                    add_to_down_block_inputs.append(add_to_down_block_input)

                residuals.append(hidden_state)

            if "downsamplers" in down_block:
                hidden_state = down_block["downsamplers"][0]["conv"](hidden_state)

                if len(add_to_down_block_inputs) < len(self.controlnet_down_blocks):
                    add_to_down_block_input = self.controlnet_down_blocks[len(add_to_down_block_inputs)](hidden_state)
                    add_to_down_block_inputs.append(add_to_down_block_input)

                residuals.append(hidden_state)

        hidden_state = self.mid_block["resnets"][0](hidden_state, t)
        hidden_state = self.mid_block["attentions"][0](hidden_state, encoder_hidden_states)
        hidden_state = self.mid_block["resnets"][1](hidden_state, t)

        mid_block_res_sample = self.controlnet_mid_block(hidden_state)

        down_block_res_samples = []

        for up_block in self.up_blocks:
            for i, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                hidden_state = torch.concat([hidden_state, residual], dim=1)

                hidden_state = resnet(hidden_state, t)

                if "attentions" in up_block:
                    hidden_state = up_block["attentions"][i](hidden_state, encoder_hidden_states)

                down_block_res_sample = self.controlnet_up_blocks[len(down_block_res_samples)](hidden_state)
                down_block_res_samples.insert(0, down_block_res_sample)

            if "upsamplers" in up_block:
                hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
                hidden_state = up_block["upsamplers"][0]["conv"](hidden_state)

        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)

        add_to_output = self.controlnet_conv_out(hidden_state)

        return dict(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            add_to_down_block_inputs=add_to_down_block_inputs,
            add_to_output=add_to_output,
        )

    @classmethod
    def from_unet(cls, unet):
        unet = maybe_ddp_module(unet)

        controlnet = cls()

        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
        controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())
        controlnet.up_blocks.load_state_dict(unet.up_blocks.state_dict())

        controlnet.conv_norm_out.load_state_dict(unet.conv_norm_out.state_dict())
        controlnet.conv_out.load_state_dict(unet.conv_out.state_dict())

        return controlnet


class SDXLAdapter(nn.Module, ModelUtils):
    def __init__(self):
        super().__init__()

        # fmt: off

        self.adapter = nn.ModuleDict(dict(
            # 3 -> 768
            unshuffle=nn.PixelUnshuffle(16),

            # 768 -> 320
            conv_in=nn.Conv2d(768, 320, kernel_size=3, padding=1),

            body=nn.ModuleList([
                # 320 -> 320
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList(
                        nn.ModuleDict(dict(block1=nn.Conv2d(320, 320, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(320, 320, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(320, 320, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(320, 320, kernel_size=1))),
                    )
                )),
                # 320 -> 640
                nn.ModuleDict(dict(
                    in_conv=nn.Conv2d(320, 640, kernel_size=1),
                    resnets=nn.ModuleList(
                        nn.ModuleDict(dict(block1=nn.Conv2d(640, 640, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(640, 640, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(640, 640, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(640, 640, kernel_size=1))),
                    )
                )),
                # 640 -> 1280
                nn.ModuleDict(dict(
                    downsample=nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                    in_conv=nn.Conv2d(640, 1280, kernel_size=1),
                    resnets=nn.ModuleList(
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                    )
                )),
                # 1280 -> 1280
                nn.ModuleDict(dict(
                    resnets=nn.ModuleList(
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                        nn.ModuleDict(dict(block1=nn.Conv2d(1280, 1280, kernel_size=3, padding=1), act=nn.ReLU(), block2=nn.Conv2d(1280, 1280, kernel_size=1))),
                    )
                )),
            ])
        ))

        # fmt: on

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            if "downsample" in block:
                x = block["downsample"](x)

            if "in_conv" in block:
                x = block["in_conv"](x)

            for resnet in block["resnets"]:
                residual = x
                x = resnet["block1"](x)
                x = resnet["act"](x)
                x = resnet["block2"](x)
                x = residual + x

            features.append(x)

        return features