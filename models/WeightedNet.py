import torch
from torch import nn
from torch import flatten
import torch.nn.functional as F

class WeightedNet(nn.Module):
    def __init__(self, num_classes=2):
        super(WeightedNet, self).__init__()
