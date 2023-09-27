from torch import nn

from utils import ModelUtils


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
