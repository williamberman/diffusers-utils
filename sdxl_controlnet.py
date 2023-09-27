import torch
import torch.nn.functional as F
from torch import nn

from utils import (ModelUtils, ResnetBlock2D, Transformer2DModel,
                   get_sinusoidal_embedding, maybe_ddp_module, zero_module)


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
