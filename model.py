# ===================== model.py ===================== #
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspCNN(nn.Module):
    def __init__(self, input_size=(480, 640)):
        super(GraspCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        # Compute flattened size dynamically
        dummy_input = torch.zeros(1, 4, *input_size)
        with torch.no_grad():
            x = self._forward_conv(dummy_input)
            self.flattened_size = x.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 6)  # Predict (x, y, theta, w, h, q)

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)  # [B, 4, H, W]
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x  # [B, 6]
