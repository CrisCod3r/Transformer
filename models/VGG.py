import torch
import torch.nn as nn  
from torch import flatten

# Everything has been taken from the original VGG paper
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, vgg_type="VGG16"):

        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[vgg_type])

        self.fullblock = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),

            # Added dropout to combat overfitting
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),

            # Added dropout to combat overfitting
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = flatten(x , 1)

        x = self.fullblock(x)

        return x

    def create_conv_layers(self, vgg_type):
        layers = []
        in_channels = self.in_channels

        for x in vgg_type:

            # "M" means MaxPool
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # Unpack layers and make them sequential
        return nn.Sequential(*layers)

