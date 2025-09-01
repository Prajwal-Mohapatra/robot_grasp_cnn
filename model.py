import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers. (Unchanged from your original code)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, 1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match input/output channels
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu_out(out)

class AttentionComplementaryModule(nn.Module):
    """
    An Attention Complementary Module (ACM) for fusing RGB and Depth feature maps.
    It learns a spatial attention map to weigh the importance of each modality at every pixel.
    """
    def __init__(self, in_channels):
        super(AttentionComplementaryModule, self).__init__()
        # This network learns the attention map C
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # Ensures attention weights are between 0 and 1
        )

    def forward(self, rgb_features, depth_features):
        """ Fused = C * RGB + (1 - C) * Depth """
        combined = torch.cat((rgb_features, depth_features), dim=1)
        attention_map = self.attention_net(combined)
        fused_features = attention_map * rgb_features + (1 - attention_map) * depth_features
        return fused_features

class AC_GRConvNet(nn.Module):
    """
    GR-ConvNet modified with a two-stream encoder and an Attention Complementary Module.
    """
    def __init__(self, output_channels=4):
        super(AC_GRConvNet, self).__init__()

        # -- RGB Stream Encoder --
        self.rgb_conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.rgb_relu1 = nn.ReLU(inplace=True)
        self.rgb_res1 = ResidualBlock(32, 32, stride=2)
        self.rgb_res2 = ResidualBlock(32, 64, stride=2)
        self.rgb_res3 = ResidualBlock(64, 128, stride=2)

        # -- Depth Stream Encoder --
        self.depth_conv1 = nn.Conv2d(1, 32, kernel_size=9, stride=1, padding=4)
        self.depth_relu1 = nn.ReLU(inplace=True)
        self.depth_res1 = ResidualBlock(32, 32, stride=2)
        self.depth_res2 = ResidualBlock(32, 64, stride=2)
        self.depth_res3 = ResidualBlock(64, 128, stride=2)

        # -- Fusion and Backbone --
        self.acm_fusion = AttentionComplementaryModule(in_channels=128)
        self.backbone_res1 = ResidualBlock(128, 128)
        self.backbone_res2 = ResidualBlock(128, 128)

        # -- Decoder -- (Structure remains the same)
        self.decoder_res1 = ResidualBlock(128, 128)
        self.decoder_up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_relu1 = nn.ReLU(inplace=True)

        self.decoder_res2 = ResidualBlock(64 + 64, 64) # Skip connection from encoder_res2
        self.decoder_up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_relu2 = nn.ReLU(inplace=True)

        self.decoder_res3 = ResidualBlock(32 + 32, 32) # Skip connection from encoder_res1
        self.decoder_up3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_relu3 = nn.ReLU(inplace=True)

        # -- Output Head -- (Structure remains the same)
        self.output_head = nn.Sequential(
            nn.Conv2d(32 + 4, 32, kernel_size=3, padding=1), # Skip connection from input
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # Input 'x' has 4 channels. Split into RGB (3) and Depth (1).
        rgb_input = x[:, 0:3, :, :]
        depth_input = x[:, 3:4, :, :]

        # == Encoder Pass ==
        # RGB Stream
        rgb_c1_out = self.rgb_relu1(self.rgb_conv1(rgb_input))
        rgb_r1_out = self.rgb_res1(rgb_c1_out)
        rgb_r2_out = self.rgb_res2(rgb_r1_out)
        rgb_r3_out = self.rgb_res3(rgb_r2_out)

        # Depth Stream
        depth_c1_out = self.depth_relu1(self.depth_conv1(depth_input))
        depth_r1_out = self.depth_res1(depth_c1_out)
        depth_r2_out = self.depth_res2(depth_r1_out)
        depth_r3_out = self.depth_res3(depth_r2_out)

        # == Fusion before Backbone ==
        fused_bottleneck = self.acm_fusion(rgb_r3_out, depth_r3_out)
        backbone_out = self.backbone_res2(self.backbone_res1(fused_bottleneck))

        # == Decoder Pass with Fused Skip Connections ==
        # Fuse skip connections using simple element-wise addition for efficiency
        fused_skip_r1 = rgb_r1_out + depth_r1_out
        fused_skip_r2 = rgb_r2_out + depth_r2_out
        
        dec_r1_out = self.decoder_res1(backbone_out)
        dec_u1_out = self.decoder_relu1(self.decoder_up1(dec_r1_out))

        skip1 = torch.cat((dec_u1_out, fused_skip_r2), 1)
        dec_r2_out = self.decoder_res2(skip1)
        dec_u2_out = self.decoder_relu2(self.decoder_up2(dec_r2_out))

        skip2 = torch.cat((dec_u2_out, fused_skip_r1), 1)
        dec_r3_out = self.decoder_res3(skip2)
        dec_u3_out = self.decoder_relu3(self.decoder_up3(dec_r3_out))

        skip3 = torch.cat((dec_u3_out, x), 1) # Final skip from original input
        
        # == Final Output ==
        output = self.output_head(skip3)

        # Apply appropriate activations to each channel (Unchanged)
        q_pred = torch.sigmoid(output[:, 0:1, :, :])      # Quality map [0, 1]
        cos_pred = torch.tanh(output[:, 1:2, :, :])       # cos(2θ) map [-1, 1]
        sin_pred = torch.tanh(output[:, 2:3, :, :])       # sin(2θ) map [-1, 1]
        width_pred = torch.sigmoid(output[:, 3:4, :, :])  # Width map [0, 1]

        return torch.cat([q_pred, cos_pred, sin_pred, width_pred], 1)

if __name__ == '__main__':
    # Test the model with a dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The model name is now AC_GRConvNet
    model = AC_GRConvNet().to(device)
    # The input shape remains the same (Batch, 4 Channels for RGB-D, Height, Width)
    dummy_input = torch.randn(2, 4, 224, 224).to(device)
    output = model(dummy_input)
    
    print("AC-GRConvNet Test")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check output shapes for each component
    q, cos_t, sin_t, w = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    print(f"  - Quality map shape: {q.shape}")
    print(f"  - Cos(2θ) map shape: {cos_t.shape}")
    print(f"  - Sin(2θ) map shape: {sin_t.shape}")
    print(f"  - Width map shape: {w.shape}")
    
    # Check value ranges
    print(f"\nValue ranges:")
    print(f"  - Quality min/max: {q.min().item():.2f}/{q.max().item():.2f} (Expected: ~0-1)")
    print(f"  - Cos(2θ) min/max: {cos_t.min().item():.2f}/{cos_t.max().item():.2f} (Expected: ~-1-1)")
    print(f"  - Width min/max: {w.min().item():.2f}/{w.max().item():.2f} (Expected: ~0-1)")
