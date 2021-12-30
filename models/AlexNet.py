import torch.nn as nn
from torch import flatten
import torch.nn.functional as F

# AlexNet CNN architecture (slightly modified)
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.name = "AlexNet"
        # Convolutional layers (RGB input, 3 input channels)        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Linear layers (out_features=2 because there are only 2 classes)
        self.full1  = nn.Linear(in_features= 256, out_features= 4096)
        self.full2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.full3 = nn.Linear(in_features=4096 , out_features=2)

        # Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)



    def forward(self,x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.maxpool2(x)
        
        # Flatten tensor to enter linear layer
        x = flatten(x , 1)

        x = self.full1(x)
        x = F.relu(x)     
           
        x = self.full2(x)
        x = F.relu(x)     

        x = self.full3(x)

        return x