import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.nn import Module

class SimpleCNN(Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16*5*5, 128) # easy way to find: run the model with debugger and check x size before
        self.fc2 = nn.Linear(128,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, img):
        # downsample with max
        x = f.max_pool2d(f.relu(self.conv1(img)), (2,2))
        # downsample with avg
        x = f.max_pool2d(f.relu(self.conv2(x)), (2,2))
        # flatten starting at the channel depth dim
        x = torch.flatten(x, -3)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim=-1)
        return x