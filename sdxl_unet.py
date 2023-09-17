from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.utils.outputs import BaseOutput
from torch import nn

from blocks import ResnetBlock2D, Transformer2DModel, get_sinusoidal_embedding


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class SDXLUNet(UNet2DConditionModel):
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
        sample,
        timesteps,
        encoder_hidden_states,
        added_cond_kwargs=None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        add_to_down_block_inputs: Optional[Tuple[torch.Tensor]] = None,
        add_to_output: Optional[Tuple[torch.Tensor]] = None,
        return_dict=True,
        # for compatibility with diffusers
        cross_attention_kwargs=None,
    ):
        if isinstance(add_to_down_block_inputs, tuple):
            add_to_down_block_inputs = list(add_to_down_block_inputs)

        if isinstance(down_block_additional_residuals, tuple):
            down_block_additional_residuals = list(down_block_additional_residuals)

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

        sample = self.conv_in(sample)

        residuals = [sample]

        for down_block in self.down_blocks:
            for i, resnet in enumerate(down_block["resnets"]):
                if add_to_down_block_inputs is not None:
                    sample = sample + add_to_down_block_inputs.pop(0)

                sample = resnet(sample, timesteps)

                if "attentions" in down_block:
                    sample = down_block["attentions"][i](sample, encoder_hidden_states)

                residuals.append(sample)

            if "downsamplers" in down_block:
                if add_to_down_block_inputs is not None:
                    sample = sample + add_to_down_block_inputs.pop(0)

                sample = down_block["downsamplers"][0]["conv"](sample)

                residuals.append(sample)

        sample = self.mid_block["resnets"][0](sample, timesteps)
        sample = self.mid_block["attentions"][0](sample, encoder_hidden_states)
        sample = self.mid_block["resnets"][1](sample, timesteps)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        for up_block in self.up_blocks:
            for i, resnet in enumerate(up_block["resnets"]):
                residual = residuals.pop()

                if down_block_additional_residuals is not None:
                    residual = residual + down_block_additional_residuals.pop()

                sample = torch.concat([sample, residual], dim=1)

                sample = resnet(sample, timesteps)

                if "attentions" in up_block:
                    sample = up_block["attentions"][i](sample, encoder_hidden_states)

            if "upsamplers" in up_block:
                sample = F.interpolate(sample, scale_factor=2.0, mode="nearest")
                sample = up_block["upsamplers"][0]["conv"](sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if add_to_output is not None:
            sample = sample + add_to_output

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    # methods to mimic diffusers

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def config(self):
        @dataclass
        class SDXLUnetConfig:
            addition_time_embed_dim: int = 256
            in_channels: int = 4

        config = SDXLUnetConfig()

        return config

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        diffusers_unet = UNet2DConditionModel.from_pretrained(*args, **kwargs)
        unet = cls()
        unet.load_state_dict(diffusers_unet.state_dict())
        return unet
