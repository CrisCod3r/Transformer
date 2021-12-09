# Flatten tensor to enter linear layer
#x = flatten(x , 1)

import torch
import torch.nn as nn
from math import ceil
from torch import flatten

# EfficientNet does not propose a new architecture, it takes a base model (or base architecture) and scales it
base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# Values of alpha, beta and gamma 
phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2), 
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

# Classic convolutional block, nothing new here
class ConvolutionalBlock(nn.Module):
    # Groups is a parameter used to apply depthwise convolutions, it is set to 1 if it is not applied
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)

        # Batch Normalization 
        self.batchnorm = nn.BatchNorm2d(out_channels)

        # SiLU function as proposed in EfficientNet paper
        self.silu = nn.SiLU()



    def forward(self, x):
        x = self.conv(x)

        x = self.batchnorm(x)
        
        x = self.silu(x)

        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):

        super(SqueezeExcitation, self).__init__()

        self.squeeze = nn.Sequential(

            # Convert to a single value with the exact same number of channels (in_channels)
            nn.AdaptiveAvgPool2d(1), 

            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),

            # Return to same output channels as in_channels
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        # Return the value of each channel weighted by the value before the excitation
        return x * self.squeeze(x)

class InvertedResidualBlock(nn.Module):
    # reduction is used for squeeze excitaction and survival prob for stochastic depth
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):        

        super(InvertedResidualBlock, self).__init__()

        self.survival_prob = survival_prob

        # If in_channels != out_channels we can not use the skip connection on our base model
        self.use_residual = in_channels == out_channels and stride == 1

        # Hidden dimention inside the IRB, as proposed
        hidden_dim = in_channels * expand_ratio

        # If we are going to expand
        self.expand = in_channels != hidden_dim
        reduced_dim = in_channels // reduction

        if self.expand:
            self.expanded_conv = ConvolutionalBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            ConvolutionalBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),

            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):

        # Stochastic depth is only used in training
        if not self.training:
            return x

        # Generate a random value to choose which layers are ignored (only the skip connection is used)
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expanded_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):

    # alpha ==> layers,  beta ==> channels, values are chosen as proposed in the EfficientNet paper
    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        
        # In this implementation, we do not rescale the resolution of the images
        phi, res, drop_rate = phi_values[version]

        depth_factor = alpha ** phi
        width_factor = beta ** phi
        
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [ConvolutionalBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:

            # The output channels have to dividible by 4 (because of the reduced dimension used earlier)
            # Width increase
            out_channels = 4 * ceil(int(channels*width_factor) / 4)

            # Depth increase
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,

                        # Downsample always at the first layer
                        stride = stride if layer == 0 else 1,
                        kernel_size=kernel_size,

                        # if K = 1x1 -> padding = 0, K = 3x3 -> padding = 1, K = 5x5 -> padding = 2
                        padding=kernel_size // 2, 
                    )
                )
                in_channels = out_channels

        features.append( ConvolutionalBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0) )

        return nn.Sequential(*features)

    def __init__(self, version, num_classes=2):

        super(EfficientNet, self).__init__()

        # We need to calculate the factors depending on the version
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)

        # Last output channels
        last_channels = ceil(1280 * width_factor)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )



    

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)

        x = flatten(x , 1)
        x = self.classifier(x)

        return x


#     model = EfficientNet( version=version, num_classes=num_classes).to(device)