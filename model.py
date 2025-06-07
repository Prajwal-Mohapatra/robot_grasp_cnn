# ===================== model.py =====================
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspCNN(nn.Module):
    def __init__(self, input_size=(480, 640)):
        super(GraspCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Dynamically compute flattened feature size
        dummy_input = torch.zeros(1, 4, *input_size)
        with torch.no_grad():
            x = self._forward_conv(dummy_input)
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 4 * 2)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        return x

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)  # Shape: [B, 4, H, W]
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 4, 2)
