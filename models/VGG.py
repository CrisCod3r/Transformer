import torch
import torch.nn as nn  
from torch import flatten

# Every architecture has been taken from the original VGG paper (except the classifiers)
VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, vgg_type="VGG19"):

        super(VGG, self).__init__()

        self.name = vgg_type

        self.in_channels = in_channels

        self.conv_layers = self.create_conv_layers(VGG_types[vgg_type])

        # Linear layer for classification
        self.classifier = nn.Linear(512, num_classes)
        

    def forward(self, x):
        x = self.conv_layers(x)

        # Flatten tensor to enter linear layer
        x = flatten(x, 1)

        x = self.classifier(x)

        return x

    def create_conv_layers(self, vgg_type):

        layers = []
        in_channels = self.in_channels

        for param in vgg_type:

            # "M" means MaxPool
            if param == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            
            else:

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=param,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(param),
                    nn.ReLU(inplace=True),
                ]

                in_channels = param
            

        # Unpack layers and make them sequential
        return nn.Sequential(*layers)

