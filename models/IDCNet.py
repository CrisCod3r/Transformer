import torch.nn as nn
from torch import flatten
import torch.nn.functional as F
from DepthwiseSeparableConvolution import depthwise_separable_conv as DepthSepConv

class IDCNet(nn.Module):
    def __init__(self):
        super(IDCNet,self).__init__()

        # --------=| Block #1 |=-------------
        self.block1 = nn.Sequential(

            DepthSepConv(nin = 3, nout = 64, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(64),

            DepthSepConv(nin = 64, nout = 64, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size = 2)
        )

        # --------=| Block #2 |=-------------
        self.block2 = nn.Sequential(

            DepthSepConv(nin = 64, nout = 128, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(128),

            DepthSepConv(nin = 128, nout = 128, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(kernel_size = 2)
        )

        # --------=| Block #3|=-------------
        self.block3 = nn.Sequential(

            DepthSepConv(nin = 128, nout = 256, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(256),

            DepthSepConv(nin = 256, nout = 256, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(256),

            DepthSepConv(nin = 256, nout = 256, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size = 2)      
        )

        # --------=| Block #4|=-------------
        self.block4 = nn.Sequential(

            DepthSepConv(nin = 256, nout = 512, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(512),

            DepthSepConv(nin = 512, nout = 512, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(512),

            DepthSepConv(nin = 512, nout = 512, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(512),

            nn.MaxPool2d(kernel_size = 2)      
        )

        # --------=| Block #5 |=-------------
        self.block5 = nn.Sequential(

            DepthSepConv(nin = 512, nout = 512, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(512),

            DepthSepConv(nin = 512, nout = 512, kernel_size = 3),
            nn.SELU(),
            nn.BatchNorm2d(512),

            nn.MaxPool2d(kernel_size = 2)
        )

        self.linear = nn.Linear(in_features= 512, out_features = 2)


    
    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = flatten(x, 1)

        x = self.linear(x)

        return x