import torch
from torch import nn
from torch import flatten
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self, use_auxiliar=True, num_classes=2):
        super(GoogLeNet, self).__init__()

        assert use_auxiliar == True or use_auxiliar == False
        self.use_auxiliar = use_auxiliar

        self.conv1 = ConvolutionalBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvolutionalBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Parameter order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)

        # Classification layer
        self.full = nn.Linear(1024, num_classes)

        if self.use_auxiliar:
            self.aux1 = AuxiliarClassifier(512, num_classes)
            self.aux2 = AuxiliarClassifier(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)

        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.use_auxiliar and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.use_auxiliar and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)

        # Flatten tensor to enter linear layer
        x = flatten(x , 1)

        # Dropout, to combat overfitting
        x = self.dropout(x)
        x = self.full(x)

        if self.use_auxiliar and self.training:
            return aux1, aux2, x
        else:
            return x

# An InceptionBlock consists on reducing the input channels for the 3x3 kernel and the 5x5 kernel and concatenating the outputs
# of the four branches
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):

        super(InceptionBlock, self).__init__()

        # 1x1 Kernel (no reduction)
        self.branch1 = ConvolutionalBlock(in_channels, out_1x1, kernel_size=1)

        # 3x3 Kernel, reduction is applied
        self.branch2 = nn.Sequential(

            # Input channels to reducted
            ConvolutionalBlock(in_channels, red_3x3, kernel_size=1),

            # Reducted channels as input
            ConvolutionalBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 5x5 Kernel, reduction is applied
        self.branch3 = nn.Sequential(

            # Input channels to reducted
            ConvolutionalBlock(in_channels, red_5x5, kernel_size=1),
            
            # Reducted channels as input
            ConvolutionalBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # In this branch the reduction is applied after the MaxPooling
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvolutionalBlock(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )

# Auxiliar classifier proposed, only used in traning
class AuxiliarClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliarClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvolutionalBlock(in_channels, 128, kernel_size=1)
        self.full1 = nn.Linear(2048, 1024)
        self.full2 = nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.pool(x)

        x = self.conv(x)

        # Flatten tensor to enter linear layer
        x = flatten(x , 1)

        x = self.full1(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.full2(x)

        return x

# Classic convolutional blocks (grouped in a class)
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)

        return x



