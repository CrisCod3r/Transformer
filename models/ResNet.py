import torch
import torch.nn as nn
from torch import flatten

class ConvLayerBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):

        super(ConvLayerBlock, self).__init__()

        # Parameter used to expand the number of channels, as proposed in the ResNet paper
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv2 = nn.Conv2d( intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv3 = nn.Conv2d( intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)

        self.batchnorm = nn.BatchNorm2d(intermediate_channels)
        self.batchnorm2 = nn.BatchNorm2d(intermediate_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):

        # Tensors share references, clone(), creates a completely new one that does not share memory
        identity = x.clone()

        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batchnorm2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x = x + identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, convlayerblock, layers, image_channels, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines
        self.layer1 = self.make_layer(
            ConvLayerBlock, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self.make_layer(
            ConvLayerBlock, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self.make_layer(
            ConvLayerBlock, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self.make_layer(
            ConvLayerBlock, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.full = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = flatten(x , 1)
        x = self.full(x)

        return x

    def make_layer(self, layerblock, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space (56x56 -> 28x28,  stride=2 ), or channels changes
        # we need to adapt the Identity (skip connection) so that it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            ConvLayerBlock(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first ResNet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.

        # One residual block has already been computed
        for i in range(num_residual_blocks - 1):
            layers.append(ConvLayerBlock(self.in_channels, intermediate_channels))

        # Unpack layers
        return nn.Sequential(*layers)

# All numbers in the second parameter have been taken directly from the proposed 
# channels in the ResNet paper

def ResNet50(img_channel=3, num_classes=2):
    return ResNet(ConvLayerBlock, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=2):
    return ResNet(ConvLayerBlock, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=2):
    return ResNet(ConvLayerBlock, [3, 8, 36, 3], img_channel, num_classes)


def test():
    net = ResNet50(img_channel=3, num_classes=2)
    y = net(torch.randn(4, 3, 224, 224)).to("cpu")
    print(y.size())


test()