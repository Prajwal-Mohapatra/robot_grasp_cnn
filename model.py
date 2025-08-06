import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GraspCNN(nn.Module):
    def __init__(self, pretrained=True):
        """
        A CNN model for grasp prediction that outputs dense grasp maps.
        V6: Added Dropout layers in the decoder to combat overfitting.
        """
        super(GraspCNN, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        conv1_weight = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, :3, :, :] = conv1_weight
        resnet.conv1.weight.data[:, 3, :, :] = conv1_weight.mean(dim=1)

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1), # Added Dropout
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1), # Added Dropout
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.quality_head = nn.Conv2d(8, 1, kernel_size=1)
        self.angle_head = nn.Conv2d(8, 2, kernel_size=1)
        self.width_head = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        x = self.features(x)
        x = self.decoder(x)
        
        quality = torch.sigmoid(self.quality_head(x))
        angle = self.angle_head(x)
        width = torch.relu(self.width_head(x))

        return quality, angle, width
