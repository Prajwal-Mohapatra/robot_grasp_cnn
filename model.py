# ===================== model.py =====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GraspCNN(nn.Module):
    """
    A Fully Convolutional Grasp-Prediction Network (Adapted for the pipeline).

    This model uses a pretrained ResNet-34 as an encoder to extract features
    from the input RGB-D image. It then uses a series of up-sampling layers
    (a decoder) to generate pixel-wise prediction maps for grasp quality,
    angle (as cos and sin), and width.
    """
    def __init__(self, pretrained=True):
        super(GraspCNN, self).__init__()

        # 1. Encoder: Load a pretrained ResNet-34
        # We use the weights from ImageNet to leverage learned features.
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        # 2. Adapt the first convolutional layer for 4 input channels (RGB-D)
        # We copy the weights for the RGB channels and use the mean for the new depth channel.
        conv1_weight = resnet.conv1.weight.data
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data[:, :3, :, :] = conv1_weight
        # Corrected line: Removed keepdim=True to match tensor shapes for assignment
        self.conv1.weight.data[:, 3, :, :] = conv1_weight.mean(dim=1)

        # 3. Use the rest of the ResNet layers as the feature extractor (encoder)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 4. Decoder: A series of up-sampling layers to generate the output maps
        # Each block consists of an up-sampling layer followed by a convolution.
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 5. Final Output Convolutions
        # These produce the final 4 maps: quality, cos(2a), sin(2a), width
        self.conv_q = nn.Conv2d(32, 1, kernel_size=1)
        self.conv_cos = nn.Conv2d(32, 1, kernel_size=1)
        self.conv_sin = nn.Conv2d(32, 1, kernel_size=1)
        self.conv_width = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the network.
        Accepts a 4-channel RGB-D tensor.
        """
        # --- Encoder ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # --- Decoder ---
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = F.relu(self.upconv5(x))

        # --- Output Maps ---
        # Generate each map and apply the appropriate activation function
        # to constrain the output to the correct range.
        q_map = torch.sigmoid(self.conv_q(x))      # Quality: [0, 1]
        cos_map = torch.tanh(self.conv_cos(x))     # Cosine: [-1, 1]
        sin_map = torch.tanh(self.conv_sin(x))     # Sine: [-1, 1]
        width_map = torch.sigmoid(self.conv_width(x)) # Width: [0, 1] (normalized)

        # Concatenate the maps to form the final output tensor
        # Shape: (batch_size, 4, height, width)
        return torch.cat([q_map, cos_map, sin_map, width_map], dim=1)

if __name__ == '__main__':
    # Test the model with a dummy input to verify shapes
    print("Testing model architecture...")
    # Batch size = 2, Channels = 4 (RGB-D), Height = 224, Width = 224
    dummy_input = torch.randn(2, 4, 224, 224)

    model = GraspCNN()
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # The output shape should be (2, 4, 224, 224) to match the pipeline's expectations
    assert output.shape == (2, 4, 224, 224), "Output shape is incorrect!"
    print("\n✅ Model architecture is compatible with the pipeline.")
