import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):

        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )
        
        self.batchnorm = nn.BatchNorm2d(planes)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.batchnorm(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.batchnorm(out)
        out = F.relu(out)

        # Residual connection
        out += self.shortcut(x)
        out = F.relu(out)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels, self.expansion * channels, kernel_size=1, bias=False)

        self.batchnorm1 = nn.BatchNorm2d(channels)
        self.batchnorm2 = nn.BatchNorm2d(self.expansion*channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.batchnorm1(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.batchnorm2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()

        self.name = "ResNet50"
        self.in_channels = 64

        # First convolutional layer, to transform 3 channels to 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(64)


        self.blocklayer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        self.blocklayer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.blocklayer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.blocklayer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion

        # Unpack and return as a sequential layer
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = F.relu(x)

        x = self.blocklayer1(x)
        x = self.blocklayer2(x)
        x = self.blocklayer3(x)
        x = self.blocklayer4(x)
        x = F.avg_pool2d(x, 4)

        # Flatten to enter linear layer
        x = torch.flatten(x, 1)

        x = self.linear(x)
        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    # Original [3, 4, 6, 3]
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
