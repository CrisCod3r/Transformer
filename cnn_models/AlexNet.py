import torch.nn as nn
from torch import flatten
import torch.nn.functional as F

# AlexNet CNN architecture (slightly modified)
class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # Minimum input size for orginal AlexNet is 63 x 63, this has been modified to accepted 50 x 50 images
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))


        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # 2 Classes
            nn.Linear(4096, 2),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        # Flatten tensor to enter linear layer
        x = flatten(x, 1)

        x = self.classifier(x)

        return x