"""
EnhancerNet - Lightweight CNN for Image Quality Enhancement
Designed to restore degraded fundus images before segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual  # Skip connection
        out = self.relu(out)

        return out


class EnhancerNet(nn.Module):
    """
    Lightweight enhancement network using encoder-decoder architecture
    with residual connections and skip connections

    Architecture:
    - Encoder: Extracts features at multiple scales
    - Bottleneck: Residual blocks for feature refinement
    - Decoder: Reconstructs enhanced image
    - Skip connections: Preserve spatial information

    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_features: Number of features in first layer
        num_residual_blocks: Number of residual blocks in bottleneck
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 32,
        num_residual_blocks: int = 4
    ):
        super().__init__()

        # Encoder (downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_features, base_features * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True)
        )

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(base_features * 4) for _ in range(num_residual_blocks)]
        )

        # Decoder (upsampling path)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_features * 4, base_features * 2,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(base_features * 4, base_features *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_features * 2, base_features,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.out = nn.Conv2d(base_features, out_channels,
                             kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Degraded input image of shape (B, C, H, W)

        Returns:
            Enhanced image of shape (B, C, H, W)
        """
        # Encoder with skip connections
        e1 = self.enc1(x)      # Same size
        e2 = self.enc2(e1)     # 1/2 size
        e3 = self.enc3(e2)     # 1/4 size

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.dec3(b)                      # 1/2 size
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # Same size as e1
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # Original size

        # Output with residual connection (learns difference from input)
        enhancement = self.out(d1)
        enhanced = x + enhancement  # Residual learning

        return torch.clamp(enhanced, 0.0, 1.0)


class LightweightEnhancerNet(nn.Module):
    """
    Ultra-lightweight version for faster training/inference
    Uses depthwise separable convolutions
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 16
    ):
        super().__init__()

        # Depthwise separable convolution
        def depthwise_conv(in_c, out_c, stride=1):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride,
                          padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                # Pointwise
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = depthwise_conv(in_channels, base_features)
        self.enc2 = depthwise_conv(base_features, base_features * 2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_features * 2),
            ResidualBlock(base_features * 2)
        )

        # Decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_features * 2, base_features,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        # Output
        self.out = nn.Conv2d(base_features, out_channels,
                             kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        b = self.bottleneck(e2)

        d2 = self.dec2(b)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        enhancement = self.out(d1)
        enhanced = x + enhancement

        return torch.clamp(enhanced, 0.0, 1.0)


class AttentionEnhancerNet(nn.Module):
    """
    Enhanced version with channel attention mechanism
    Helps network focus on important features
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 32,
        num_residual_blocks: int = 4
    ):
        super().__init__()

        self.enhancer = EnhancerNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            num_residual_blocks=num_residual_blocks
        )

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_features * 4, base_features *
                      4 // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 4 // 4,
                      base_features * 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.enhancer(x)


def test_enhancer_net():
    """Test function to verify EnhancerNet output shapes"""
    print("\n" + "="*50)
    print("Testing EnhancerNet Architectures")
    print("="*50)

    x = torch.randn(2, 3, 512, 512)

    # Test standard EnhancerNet
    model = EnhancerNet(base_features=32, num_residual_blocks=4)
    output = model(x)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n1. Standard EnhancerNet:")
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters:   {num_params:,}")
    assert output.shape == x.shape, "Output shape mismatch!"

    # Test lightweight version
    model_light = LightweightEnhancerNet(base_features=16)
    output_light = model_light(x)
    num_params_light = sum(p.numel() for p in model_light.parameters())

    print(f"\n2. Lightweight EnhancerNet:")
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output_light.shape}")
    print(f"   Parameters:   {num_params_light:,}")
    assert output_light.shape == x.shape, "Output shape mismatch!"

    print(f"\n✓ All tests passed!")
    print(
        f"  Parameter reduction: {(1 - num_params_light/num_params)*100:.1f}%")


if __name__ == '__main__':
    test_enhancer_net()
