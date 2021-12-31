import torch
from torch import nn
from torch import flatten
import torch.nn.functional as F


# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
 
class WeightedNet(nn.Module):
    def __init__(self, in_channels = 3,num_classes=2):
        super(WeightedNet, self).__init__()
        
        self.name = "WeightedNet"
        # ----------=| ExtractionBlock 1 |=-------------
        # Parameter order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception1_B1 = nn.Sequential(

            InceptionBlock(in_channels, 24, 6, 8, 6, 8, 24),  # Output channels = 64 (24 + 8 + 8 + 24)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(64)
        )

        self.inception2_B1 = nn.Sequential(

            InceptionBlock(64, 24, 6, 8, 6, 8, 24), # Output channels = 64

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(64)
        )

        # ----------=| ExtractionBlock 2 |=-------------
        self.inception1_B2 = nn.Sequential(

            InceptionBlock(64, 48, 8, 16, 8, 16, 48),  # Output channels = 128 (48 + 16 + 16 + 48)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(128)
        )

        self.inception2_B2 = nn.Sequential(

            InceptionBlock(128, 48, 8, 16, 8, 16, 48),  # Output channels = 128 (48 + 16 + 16 + 48)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(128)
        )

        # ----------=| ExtractionBlock 3 |=-------------
        self.inception1_B3 = nn.Sequential(

            InceptionBlock(128, 96, 24, 32, 24, 32, 96),  # Output channels = 256 (96 + 32 + 32 + 96)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(256)
        )

        self.inception2_B3 = nn.Sequential(

            InceptionBlock(256, 96, 24, 32, 24, 32, 96),  # Output channels = 256 (96 + 32 + 32 + 96)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(256)
        )

        self.inception3_B3 = nn.Sequential(

            InceptionBlock(256, 96, 24, 32, 24, 32, 96),  # Output channels = 256 (96 + 32 + 32 + 96)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(256)
        )

        # ----------=| ExtractionBlock 4 |=-------------
        self.inception1_B4 = nn.Sequential(

            InceptionBlock(256, 192, 48, 64, 48, 64, 192),  # Output channels = 512 (192 + 64 + 64 + 192)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(512)
        )

        self.inception2_B4 = nn.Sequential(

            InceptionBlock(512, 192, 48, 64, 48, 64, 192),  # Output channels = 512 (192 + 64 + 64 + 192)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(512)
        )

        self.inception3_B4 = nn.Sequential(

            InceptionBlock(512, 192, 48, 64, 48, 64, 192),  # Output channels = 512 (192 + 64 + 64 + 192)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(512)
        )

        # ----------=| ExtractionBlock 5 |=-------------
        self.inception1_B5 = nn.Sequential(

            InceptionBlock(512, 192, 48, 64, 48, 64, 192),  # Output channels = 512 (192 + 64 + 64 + 192)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(512)
        )

        self.inception2_B5 = nn.Sequential(

            InceptionBlock(512, 192, 48, 64, 48, 64, 192),  # Output channels = 512 (192 + 64 + 64 + 192)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(512)
        )

        self.inception3_B5 = nn.Sequential(

            InceptionBlock(512, 192, 48, 64, 48, 64, 192),  # Output channels = 512 (192 + 64 + 64 + 192)

            # BatchNorm to combat overfitting
            nn.BatchNorm2d(512)
        )

        self.weightedClassifier1 = WeightedClassifier(in_features=64 * 25 * 25, initial_weight = 1 / 5, num_classes=2)
        self.weightedClassifier2 = WeightedClassifier(in_features=128 * 12 * 12, initial_weight = 1 / 5, num_classes=2)
        self.weightedClassifier3 = WeightedClassifier(in_features=256 * 6 * 6, initial_weight = 1 / 5, num_classes=2)
        self.weightedClassifier4 = WeightedClassifier(in_features=512 * 3 * 3, initial_weight = 1 / 5, num_classes=2)
        self.weightedClassifier5 = WeightedClassifier(in_features=512 * 1 * 1, initial_weight = 1 / 5, num_classes=2)

        # MaxPool has no trainable paramaters, and thus, we can share them between blocks
        # self.maxpool = nn.MaxPool2d(kernel_size= 2)
        # AvgPool has no trainable paramaters, and thus, we can share them between blocks
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.accuracies = [0, 0, 0, 0, 0]


    def forward(self,x):
        self.outputs = []

        x = self.inception1_B1(x)
        x = self.inception2_B1(x)

        x = self.avgpool(x)

        # Get the estimated output up until this point, also, flatten tensor to enter linear layer
        self.outputs.append( self.weightedClassifier1(flatten(x , 1)) )
        
        x = self.inception1_B2(x)
        x = self.inception2_B2(x)

        x = self.avgpool(x)

        # Get the estimated output up until this point, also, flatten tensor to enter linear layer
        self.outputs.append( self.weightedClassifier2(flatten(x , 1)) )

        x = self.inception1_B3(x)
        x = self.inception2_B3(x)
        x = self.inception3_B3(x)

        x = self.avgpool(x)

        # Get the estimated output up until this point, also, flatten tensor to enter linear layer
        self.outputs.append( self.weightedClassifier3(flatten(x , 1)) )

        x = self.inception1_B4(x)
        x = self.inception2_B4(x)
        x = self.inception3_B4(x)

        x = self.avgpool(x)

        # Get the estimated output up until this point, also, flatten tensor to enter linear layer
        self.outputs.append( self.weightedClassifier4(flatten(x , 1)) )

        x = self.inception1_B5(x)
        x = self.inception2_B5(x)
        x = self.inception3_B5(x)

        x = self.avgpool(x)

        # Get the estimated output up until this point, also, flatten tensor to enter linear layer
        self.outputs.append( self.weightedClassifier5(flatten(x , 1)) )

        return sum(self.outputs)

    def update_weights(self, labels):

        for idx in range(len(self.outputs)):
            _, predicted = self.outputs[idx].max(1)
            self.accuracies[idx] += ( predicted.eq(labels).sum().item() )

        total_sum = sum(self.accuracies)

        self.weightedClassifier1.weight = self.accuracies[0] / total_sum
        self.weightedClassifier2.weight = self.accuracies[1] / total_sum
        self.weightedClassifier3.weight = self.accuracies[2] / total_sum
        self.weightedClassifier4.weight = self.accuracies[3] / total_sum
        self.weightedClassifier5.weight = self.accuracies[4] / total_sum

    def weights(self):
        return [self.weightedClassifier1.weight,self.weightedClassifier2.weight,self.weightedClassifier3.weight,self.weightedClassifier4.weight,self.weightedClassifier5.weight]

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




class WeightedClassifier(nn.Module):
    def __init__(self, in_features, initial_weight, num_classes=2):

        super(WeightedClassifier, self).__init__()

        self.weight = initial_weight

        # Classification layer
        self.classifier = nn.Linear(in_features= in_features, out_features= num_classes)

    def forward(self, x):

        x = self.classifier(x)

        # Return the weighted output
        return x * self.weight