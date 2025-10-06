"""
U-Net Architecture for Optic Disc and Cup Segmentation
Standard U-Net with encoder-decoder structure and skip connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # Use bilinear upsampling or transposed conv
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: upsampled feature map from previous layer
            x2: skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution for output"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Optic Disc/Cup Segmentation

    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (2 for disc and cup)
        bilinear: Use bilinear upsampling instead of transposed convolution
        base_features: Number of features in first layer (default 64)
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        bilinear: bool = True,
        base_features: int = 64
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_features * 8, base_features * 16 // factor)

        # Decoder
        self.up1 = Up(base_features * 16, base_features *
                      8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)

        # Output layer
        self.outc = OutConv(base_features, n_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, n_classes, H, W)
        """
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits


def test_unet():
    """Test function to verify U-Net output shapes"""
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    x = torch.randn(2, 3, 512, 512)  # Batch of 2 RGB images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert output.shape == (2, 2, 512, 512), "Output shape mismatch!"
    print("U-Net test passed!")


if __name__ == '__main__':
    test_unet()
