import torch.nn as nn
from torch import flatten
import torch.nn.functional as F
from DepthwiseSeparableConvolution import depthwise_separable_conv as DepthSepConv

class IDCNet(nn.Module):
    def __init__(self):
        super(IDCNet,self).__init__()

        # --------=| Block #1 |=-------------
        self.block1 = nn.Sequential(

            DepthSepConv(nin = 3, nout = 32, kernel_size = 3),
            nn.PReLU(num_parameters = 32),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(kernel_size = 2)
        )

        # --------=| Block #2 |=-------------
        self.block2 = nn.Sequential(

            DepthSepConv(nin = 32, nout = 64, kernel_size = 3),
            nn.PReLU(num_parameters = 64),
            nn.BatchNorm2d(64),

            DepthSepConv(nin = 64, nout = 64, kernel_size = 3),
            nn.PReLU(num_parameters = 64),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size = 2)
        )

        # --------=| Block #3|=-------------
        self.block3 = nn.Sequential(

            DepthSepConv(nin = 64, nout = 128, kernel_size = 3),
            nn.PReLU(num_parameters = 128),
            nn.BatchNorm2d(128),

            DepthSepConv(nin = 128, nout = 128, kernel_size = 3),
            nn.PReLU(num_parameters = 128),
            nn.BatchNorm2d(128),

            DepthSepConv(nin = 128, nout = 128, kernel_size = 3),
            nn.PReLU(num_parameters = 128),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(kernel_size = 2)      
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features= 128 * 6 * 6, out_features = 512),
            nn.PReLU(num_parameters = 512),

            # This raises a ValueError if batch size == 1
            nn.BatchNorm1d(512),

            nn.Linear(in_features = 512, out_features = 2)
        )

    
    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = flatten(x, 1)

        x = self.linear(x)

        return x