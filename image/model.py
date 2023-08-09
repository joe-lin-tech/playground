import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import List
from params import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class Classifier(nn.Module):
    def __init__(self, num_channels: List[int], num_features: List[int]):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels[0], 5)
        self.conv2 = nn.Conv2d(num_channels[0], num_channels[1], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(num_channels[1] * 5 * 5, num_features[0])
        self.fc2 = nn.Linear(num_features[0], num_features[1])
        self.out = nn.Linear(num_features[1], len(CLASSES))
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)