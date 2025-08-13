import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
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

class GRConvNet(nn.Module):
    """
    GR-ConvNet: A Generative Residual Convolutional Neural Network for Grasp Prediction.
    This architecture is an encoder-decoder network with a residual backbone.
    """
    def __init__(self, input_channels=4, output_channels=4):
        super(GRConvNet, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        self.encoder_relu1 = nn.ReLU(inplace=True)

        self.encoder_res1 = ResidualBlock(32, 32, stride=2) # Downsample
        self.encoder_res2 = ResidualBlock(32, 64, stride=2) # Downsample
        self.encoder_res3 = ResidualBlock(64, 128, stride=2) # Downsample

        # Backbone (Residual Core)
        self.backbone_res1 = ResidualBlock(128, 128)
        self.backbone_res2 = ResidualBlock(128, 128)

        # Decoder
        self.decoder_res1 = ResidualBlock(128, 128)
        self.decoder_up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_relu1 = nn.ReLU(inplace=True)

        self.decoder_res2 = ResidualBlock(64 + 64, 64) # Skip connection from encoder_res2
        self.decoder_up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_relu2 = nn.ReLU(inplace=True)

        self.decoder_res3 = ResidualBlock(32 + 32, 32) # Skip connection from encoder_res1
        self.decoder_up3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_relu3 = nn.ReLU(inplace=True)

        # Output Head
        # Produces 4 maps: Quality, cos(2θ), sin(2θ), Width
        self.output_head = nn.Sequential(
            nn.Conv2d(32 + input_channels, 32, kernel_size=3, padding=1), # Skip connection from input
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder Path
        enc_c1_out = self.encoder_relu1(self.encoder_conv1(x))
        enc_r1_out = self.encoder_res1(enc_c1_out)
        enc_r2_out = self.encoder_res2(enc_r1_out)
        enc_r3_out = self.encoder_res3(enc_r2_out)

        # Backbone
        backbone_out = self.backbone_res2(self.backbone_res1(enc_r3_out))

        # Decoder Path with Skip Connections
        dec_r1_out = self.decoder_res1(backbone_out)
        dec_u1_out = self.decoder_relu1(self.decoder_up1(dec_r1_out))

        # Skip connection from encoder_res2 (64 channels)
        skip1 = torch.cat((dec_u1_out, enc_r2_out), 1)
        dec_r2_out = self.decoder_res2(skip1)
        dec_u2_out = self.decoder_relu2(self.decoder_up2(dec_r2_out))

        # Skip connection from encoder_res1 (32 channels)
        skip2 = torch.cat((dec_u2_out, enc_r1_out), 1)
        dec_r3_out = self.decoder_res3(skip2)
        dec_u3_out = self.decoder_relu3(self.decoder_up3(dec_r3_out))

        # Skip connection from original input image
        skip3 = torch.cat((dec_u3_out, x), 1)
        
        # Final Output
        output = self.output_head(skip3)

        # Apply appropriate activations to each channel
        q_pred = torch.sigmoid(output[:, 0:1, :, :])      # Quality map [0, 1]
        cos_pred = torch.tanh(output[:, 1:2, :, :])       # cos(2θ) map [-1, 1]
        sin_pred = torch.tanh(output[:, 2:3, :, :])       # sin(2θ) map [-1, 1]
        width_pred = torch.sigmoid(output[:, 3:4, :, :])  # Width map [0, 1]

        return torch.cat([q_pred, cos_pred, sin_pred, width_pred], 1)

if __name__ == '__main__':
    # Test the model with a dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRConvNet(input_channels=4).to(device)
    dummy_input = torch.randn(2, 4, 224, 224).to(device) # Batch size 2, 4 channels, 224x224
    output = model(dummy_input)
    
    print("GR-ConvNet Test")
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

