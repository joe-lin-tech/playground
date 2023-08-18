import torch.nn as nn
import torch

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        # (448, 448, 3)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3) # (224, 224, 64)
        self.pool1 = nn.MaxPool2d(2, 2) # (112, 112, 64)
        self.conv2 = nn.Conv2d(64, 192, 3, padding=1) # (112, 112, 192)
        self.pool2 = nn.MaxPool2d(2, 2) # (56, 56, 192)

        self.block1 = nn.Sequential(
            nn.Conv2d(192, 128, 1), # (56, 56, 128)
            nn.Conv2d(128, 256, 3, padding=1), # (56, 56, 256)
            nn.Conv2d(256, 256, 1), # (56, 56, 256)
            nn.Conv2d(256, 512, 3, padding=1), # (56, 56, 512)
            nn.MaxPool2d(2, 2) # (28, 28, 512)
        )

        self.block2 = []
        for _ in range(4):
            self.block2.append(nn.Conv2d(512, 256, 1)) # (28, 28, 256)
            self.block2.append(nn.Conv2d(256, 512, 3, padding=1)) # (28, 28, 512)
        self.block2.append(nn.Conv2d(512, 512, 1)) # (28, 28, 512)
        self.block2.append(nn.Conv2d(512, 1024, 3, padding=1)) # (28, 28, 1024)
        self.block2.append(nn.MaxPool2d(2, 2)) # (14, 14, 1024)
        self.block2 = nn.Sequential(*self.block2)
        
        self.block3 = []
        for _ in range(2):
            self.block3.append(nn.Conv2d(1024, 512, 1)) # (14, 14, 512)
            self.block3.append(nn.Conv2d(512, 1024, 3, padding=1)) # (14, 14, 1024)
        self.block3.append(nn.Conv2d(1024, 1024, 3, padding=1)) # (14, 14, 1024)
        self.block3.append(nn.Conv2d(1024, 1024, 3, stride=2, padding=1)) # (7, 7, 1024)
        self.block3 = nn.Sequential(*self.block3)

        self.conv3 = nn.Conv2d(1024, 1024, 3, padding=1) # (7, 7, 1024)
        self.conv4 = nn.Conv2d(1024, 1024, 3, padding=1) # (7, 7, 1024)

    def forward(self, x):
        x = self.pool2(self.conv2(self.pool1(self.conv1(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.conv4(self.conv3(x))