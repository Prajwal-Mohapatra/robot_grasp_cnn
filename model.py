# ===================== model.py =====================
import torch
import torch.nn as nn
from torchvision import models

class GraspCNN(nn.Module):
    def __init__(self, output_dim=6, pretrained=True):
        """
        Model with a deeper regression head for increased precision.
        """
        super(GraspCNN, self).__init__()

        # 1. Load a pretrained ResNet-34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        # 2. Adapt the first convolutional layer for 4 input channels (RGB-D)
        conv1_weight = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, :3, :, :] = conv1_weight
        resnet.conv1.weight.data[:, 3, :, :] = conv1_weight.mean(dim=1)

        # 3. Use the ResNet backbone for feature extraction
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Define a deeper regression head for more complex feature interpretation
        # This increased capacity is crucial for improving angle prediction.
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256), # Added an extra hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.regressor(x)
        return x
