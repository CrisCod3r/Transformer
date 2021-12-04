import torch.optim as optim
import torch.nn as nn
from torch import flatten
import torch.nn.functional as F
#torch.set_printoptions(precision=3)

# LeNet5 CNN architecture (slightly modified)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()

        # Extraction layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6,kernel_size = 5,stride = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,kernel_size = 5,stride = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5,stride = 1)

        self.full1 = nn.Linear(in_features = 120 * 5 * 5, out_features = 84)
        self.full2 = nn.Linear(in_features = 84, out_features = 2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        # Flatten tensor
        x = flatten(x , 1)

        x = self.full1(x)

        x = F.relu(x)

        x = self.full2(x)

        probs = F.softmax(x, dim=1)
        return probs

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 256, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=2)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool2(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    """def __init__(self):
        super(AlexNet,self).__init__()



        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels= 64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels= 192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels= 384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        # Extraction layers
        self.dropout = nn.Dropout(p=0.5)
        self.full1 = nn.Linear(in_features = 256 * 6 * 6, out_features = 4096)
        self.full2 = nn.Linear(in_features= 4096, out_features= 4096)
        self.full3 = nn.Linear(in_features= 4096, out_features= 2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x,inplace=True)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x,inplace=True)

        x = self.conv4(x)
        x = F.relu(x,inplace=True)

        x = self.conv5(x)
        x = F.relu(x,inplace=True)
        x = self.pool2(x)


        x = self.avgpool(x)

        # Flatten tensor
        x = flatten(x , 1)

        x = self.dropout(x)
        x = self.full1(x)
        x = F.relu(x,inplace=True)

        x = self.dropout(x)
        x = self.full2(x)
        x = F.relu(x,inplace=True)

        x = self.full3(x)

        return x"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)

        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 16 * 9 * 9, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        self.fc3 = nn.Linear(in_features = 84, out_features = 2)

    def forward(self,x):    
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        #Flatten to enter fully connected layer
        x = x.view(-1, 16 * 9 *  9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=512, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        x = self.maxpool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        x = self.maxpool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))

        x = self.maxpool(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))

        x = self.maxpool(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        #dropout was included to combat overfitting
        x = F.dropout(x, p=0.5) 

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)

        x = self.fc3(x)
        return x


# My own Convolutional Neural Network. For testing other parameters and atchitectures
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
