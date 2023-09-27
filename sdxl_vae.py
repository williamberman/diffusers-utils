import torch
from PIL import Image
from torch import nn

from utils import Attention, ModelUtils, ResnetBlock2D

scaling_factor = 0.13025


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

        z = z * scaling_factor

        return z

    def decode(self, z):
        z = z / scaling_factor

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
