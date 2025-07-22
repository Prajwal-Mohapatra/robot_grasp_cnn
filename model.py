# ===================== model.py =====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GraspCNN(nn.Module):
    def __init__(self, output_dim=6, pretrained=True):
        """
        output_dim=6 corresponds to [cx, cy, sin(θ), cos(θ), w, h]
        """
        super(GraspCNN, self).__init__()

        # 1) Load a pretrained ResNet‑34 and adapt its first conv to 4 channels
        resnet = models.resnet34(pretrained=pretrained)
        # Replace the first conv (3→4 input channels)
        w = resnet.conv1.weight.data
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize the new channel by averaging the RGB weights
        resnet.conv1.weight.data[:, :3, :, :] = w
        resnet.conv1.weight.data[:, 3:4, :, :] = w.mean(dim=1, keepdim=True)

        # Remove ResNet's classifier; we only want the feature extractor
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            # global average pooling to 512‑dim
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2) Regression head: small MLP to predict grasp parameters
        self.regressor = nn.Sequential(
            nn.Flatten(),                    # ⇒ [B, 512]
            nn.Linear(512, 256),             # hidden dimension
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)       # [cx, cy, sinθ, cosθ, w, h]
        )

    def forward(self, rgb, depth):
        # rgb: [B,3,H,W], depth: [B,1,H,W] ⇒ concat to [B,4,H,W]
        x = torch.cat([rgb, depth], dim=1)
        x = self.features(x)                # [B,512,1,1]
        x = self.regressor(x)               # [B,6]
        # split and re‐scale:
        #   cx, cy in [–1,1] (tanh), sin/cos already in [–1,1], w,h in [0,1] (sigmoid)
        cxcy, sc, wh = x[:, :2], x[:, 2:4], x[:, 4:6]
        cxcy = torch.tanh(cxcy)             # center
        sc   = torch.tanh(sc)               # sinθ, cosθ
        wh   = torch.sigmoid(wh)            # width & height normalized [0,1]
        return torch.cat([cxcy, sc, wh], dim=1)  # [B,6]
