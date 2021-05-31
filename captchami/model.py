from abc import ABC

import torch.nn as nn
import torch.nn.functional as F


class NetModel(nn.Module, ABC):
    def __init__(self, in_channels: int, classes: int, batch_size: int, kernel_size: int = 3, linear_input: int = 720):
        super(NetModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(linear_input, batch_size)
        self.fc2 = nn.Linear(batch_size, 120)
        self.fc3 = nn.Linear(in_features=120, out_features=classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=1)
