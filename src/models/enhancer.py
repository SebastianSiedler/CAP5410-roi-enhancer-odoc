"""
Lightweight Encoder-Decoder Enhancer for image preprocessing.

This model enhances fundus images before they are passed to UNet for segmentation.
It uses a shallower architecture (3-4 levels) compared to UNet and includes
residual connections to preserve original content.
"""

import torch
import torch.nn as nn


class EnhancerBlock(nn.Module):
    """Lightweight convolution block for enhancer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EnhancerDown(nn.Module):
    """Downscaling block for enhancer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            EnhancerBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class EnhancerUp(nn.Module):
    """Upscaling block for enhancer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = EnhancerBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ImageEnhancer(nn.Module):
    """
    Lightweight encoder-decoder network for image enhancement.

    Features:
    - Shallower architecture (3-4 levels vs UNet's 5 levels)
    - Residual connection from input to output
    - Produces enhanced images with same dimensions as input

    Args:
        n_channels: Number of input/output channels (3 for RGB)
        base_channels: Number of channels in first layer (default: 32, lighter than UNet's 64)
        num_levels: Number of encoder-decoder levels (3 or 4)
    """

    def __init__(self, n_channels=3, base_channels=32, num_levels=3):
        super().__init__()
        assert num_levels in [3, 4], "num_levels must be 3 or 4"

        self.n_channels = n_channels
        self.num_levels = num_levels

        # Initial convolution
        self.inc = EnhancerBlock(n_channels, base_channels)

        # Encoder (3 or 4 levels)
        self.down1 = EnhancerDown(base_channels, base_channels * 2)
        self.down2 = EnhancerDown(base_channels * 2, base_channels * 4)

        if num_levels == 3:
            # 3-level architecture
            # Bottleneck
            self.bottleneck = EnhancerDown(
                base_channels * 4, base_channels * 8)

            # Decoder
            self.up1 = EnhancerUp(base_channels * 8, base_channels * 4)
            self.up2 = EnhancerUp(base_channels * 4, base_channels * 2)
            self.up3 = EnhancerUp(base_channels * 2, base_channels)

        else:  # num_levels == 4
            # 4-level architecture
            self.down3 = EnhancerDown(base_channels * 4, base_channels * 8)

            # Bottleneck
            self.bottleneck = EnhancerDown(
                base_channels * 8, base_channels * 16)

            # Decoder
            self.up1 = EnhancerUp(base_channels * 16, base_channels * 8)
            self.up2 = EnhancerUp(base_channels * 8, base_channels * 4)
            self.up3 = EnhancerUp(base_channels * 4, base_channels * 2)
            self.up4 = EnhancerUp(base_channels * 2, base_channels)

        # Output convolution (produces RGB image)
        self.outc = nn.Conv2d(base_channels, n_channels, kernel_size=1)

        # Tanh to constrain output range, then we'll add residual
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Enhanced image [B, 3, H, W]
        """
        # Store input for residual connection
        identity = x

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        if self.num_levels == 3:
            # 3-level architecture
            x4 = self.bottleneck(x3)

            # Decoder with skip connections
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)

        else:  # num_levels == 4
            # 4-level architecture
            x4 = self.down3(x3)
            x5 = self.bottleneck(x4)

            # Decoder with skip connections
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        # Output layer
        enhancement = self.outc(x)
        enhancement = self.tanh(enhancement)  # Range: [-1, 1]

        # Add residual connection (weighted)
        # Scale enhancement to be additive correction
        enhanced = identity + 0.3 * enhancement

        # Clip to valid range [keep normalized range for now]
        # If input is normalized with ImageNet stats, output will be too

        return enhanced


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the enhancer
    print("Testing ImageEnhancer...")

    # Test 3-level enhancer
    enhancer_3 = ImageEnhancer(n_channels=3, base_channels=32, num_levels=3)
    print(f"\n3-Level Enhancer:")
    print(f"  Parameters: {count_parameters(enhancer_3):,}")

    # Test 4-level enhancer
    enhancer_4 = ImageEnhancer(n_channels=3, base_channels=32, num_levels=4)
    print(f"\n4-Level Enhancer:")
    print(f"  Parameters: {count_parameters(enhancer_4):,}")

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)

    print(f"\nInput shape: {x.shape}")

    y3 = enhancer_3(x)
    print(f"3-Level output shape: {y3.shape}")

    y4 = enhancer_4(x)
    print(f"4-Level output shape: {y4.shape}")

    # Compare with UNet
    from unet import UNet
    unet = UNet(n_channels=3, n_classes=3, base_channels=64)
    print(f"\nUNet (for comparison):")
    print(f"  Parameters: {count_parameters(unet):,}")

    print("\n✅ All tests passed!")
