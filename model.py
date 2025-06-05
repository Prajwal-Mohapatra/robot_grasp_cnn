import torch
import torch.nn as nn

class GraspCNN(nn.Module):
    def __init__(self):
        super(GraspCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 60 * 60, 128),  # Assuming input 480x480, pooled 3x: 480/8=60
            nn.ReLU(),
            nn.Linear(128, 5)  # x, y, theta, height, width
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
