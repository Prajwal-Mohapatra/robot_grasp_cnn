# ===================== model.py =====================

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspCNN(nn.Module):
    def __init__(self, input_size=(224, 224)):  # Smaller input for efficiency
        super(GraspCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        # Calculate the flattened feature size dynamically
        dummy = torch.zeros(1, 4, *input_size)
        with torch.no_grad():
            flat_features = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flat_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 8)  # 4 grasp points with (x, y)
        )

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)  # Concatenate along channel axis
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, 4, 2)
