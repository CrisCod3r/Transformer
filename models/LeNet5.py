import torch.nn as nn
from torch import flatten
import torch.nn.functional as F

# LeNet5 CNN architecture (slightly modified)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()

        self.name = "LeNet5"
        # Convolutional layers (RGB input, 3 input channels)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1)

        # Linear layers (out_features=2 because there are only 2 classes)
        self.full1 = nn.Linear(in_features = 120 * 5 * 5, out_features = 84)
        self.full2 = nn.Linear(in_features = 84, out_features = 2)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size = 2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)

        # Flatten tensor to enter linear layer
        x = flatten(x , 1)

        x = self.full1(x)
        x = F.relu(x)

        x = self.full2(x)

        return x