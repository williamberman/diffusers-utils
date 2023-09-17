from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import ControlNetModel
from diffusers.utils.outputs import BaseOutput
from torch import nn

from blocks import ResnetBlock2D, Transformer2DModel, get_sinusoidal_embedding
from utils import zero_module


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor
    add_to_down_block_inputs: Optional[Tuple[torch.Tensor]]
    add_to_output: Optional[torch.Tensor]


class SDXLControlNet(ControlNetModel):
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

        controlnet_cond = self.controlnet_cond_embedding["conv_in"](controlnet_cond)
        controlnet_cond = F.silu(controlnet_cond)

        for block in self.controlnet_cond_embedding["blocks"]:
            controlnet_cond = F.silu(block(controlnet_cond))

        controlnet_cond = self.controlnet_cond_embedding["conv_out"](controlnet_cond)

        sample = self.conv_in(sample)

        sample = sample + controlnet_cond

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
    def from_pretrained(cls, *args, **kwargs):
        diffusers_controlnet = ControlNetModel.from_pretrained(*args, **kwargs)
        controlnet = cls()
        controlnet.load_state_dict(diffusers_controlnet.state_dict())
        return controlnet

    @classmethod
    def from_unet(cls, unet):
        controlnet = cls()

        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
        controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet
