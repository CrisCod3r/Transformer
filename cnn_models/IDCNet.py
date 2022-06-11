import torch
import torch.nn as nn
from torch import flatten
import torch.nn.functional as F
from DepthwiseSeparableConvolution import depthwise_separable_conv as DepthSepConv

from torchvision.ops import SqueezeExcitation
# ---------------=| Global Variables |=------------------

# (Depth of layer,  Input channels)
models = {
    'tiny': [(1,32), (1,32), (2,64), (3,128)],
    'small': [(1,32), (2,64), (2,64), (3,128), (3,128)],
    'base': [(1,32), (2,64), (2,64), (3,128), (3,128), (3,256), (3,256), (3,512)],
    'large': [(1,32), (2,64), (2,64), (3,128), (3,128), (3,128), (3,256), (3,256), (3,256), (3,512)]
}

class IDCNet(nn.Module):
    def __init__(self, structure: list, in_channels: int = 3, num_classes: int = 2, use_downsample: bool = False ):
        super(IDCNet,self).__init__()
        
        features = []
        img_size = 50

        for block in structure:

            depth, out_channels = block[0], block[1]

            if in_channels != out_channels:

                
                features += [
                    # Add a convolution to transform from in_channels to out_channels
                    DepthSepConv(nin = in_channels, nout = out_channels, kernel_size = 3),

                    DepthSepConvBlock(n_channels = out_channels, kernel_size = 3, depth = depth, ratio = 4, use_downsample = use_downsample),
                    nn.MaxPool2d(kernel_size = 2)
                ]

                img_size = img_size // 2
            else:

                features += [
                    DepthSepConvBlock(n_channels = out_channels, kernel_size = 3, depth = depth, ratio = 4, use_downsample = use_downsample)
                ]

            in_channels = out_channels


        self.features = nn.Sequential(*features)

        self.linear = nn.Linear(in_features= out_channels * img_size * img_size, out_features = num_classes)


    
    def forward(self, x):

        x = self.features(x)

        x = flatten(x, 1)

        x = self.linear(x)

        return x



class DepthSepConvBlock(nn.Module):
    def __init__(
            self,
            n_channels: int,
            kernel_size: int,
            depth: int, # Number of DepthSepConv layers
            ratio :int, # Redution ratio for SqueezeExcitation
            use_downsample: bool = False
        ):
        super(DepthSepConvBlock, self).__init__()
        

        # Identity downsample used for skip connections. 1 x 1 Convolution
        self.identity_downsample = DepthSepConv(nin = n_channels, nout = n_channels, kernel_size = 1, padding = 0)
        self.use_downsample = use_downsample

        # Squeeze and Excitation ratio
        self.ratio = ratio

        # Main block
        self.conv_block = self.make_block(n_channels, kernel_size, depth)


    def make_block(self, n_channels, kernel_size, depth):

        blocks = []

        for block in range(depth):

            blocks += [
                # Main block
                DepthSepConv(nin = n_channels, nout = n_channels, kernel_size = kernel_size),
                nn.BatchNorm2d(n_channels),
                nn.GELU(),

                # Squeeze and Excitation
                SqueezeExcitation(input_channels = n_channels, squeeze_channels = n_channels // self.ratio, activation = nn.GELU, scale_activation = nn.Sigmoid)
                ]

        return nn.Sequential(*blocks)

    def forward(self, x):

        identity = x

        out = self.conv_block(x)

        if self.use_downsample:
            identity = self.identity_downsample(x)

        # Skip connection. (Residual)
        out += identity
        return out




def IDCNet_tiny(in_channels: int = 3,num_classes: int = 2, use_downsample: bool = False):
    return IDCNet(models['tiny'], in_channels, num_classes, use_downsample )

def IDCNet_small( in_channels: int = 3,num_classes: int = 2, use_downsample: bool = False):
    return IDCNet(models['small'], in_channels, num_classes, use_downsample )

def IDCNet_base( in_channels: int = 3,num_classes: int = 2, use_downsample: bool = False):

    return IDCNet(models['base'], in_channels, num_classes, use_downsample )

def IDCNet_large( in_channels: int = 3,num_classes: int = 2, use_downsample: bool = False):
    return IDCNet(models['large'], in_channels, num_classes, use_downsample )