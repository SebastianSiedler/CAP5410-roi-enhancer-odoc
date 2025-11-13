"""
ASPP-UNet: UNet with Atrous Spatial Pyramid Pooling for Optic Disc/Cup Segmentation

Replaces the standard UNet bottleneck with an ASPP module to capture multi-scale
context, which is crucial for accurate cup boundary detection within the disc.

Key improvements over standard UNet:
- Multi-scale feature extraction at the bottleneck
- Larger receptive field without losing resolution
- Better context aggregation for varying cup sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block used in UNet encoder/decoder"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module

    Replaces the standard bottleneck with parallel dilated convolutions
    at different rates to capture multi-scale context.
    """

    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super().__init__()

        self.dilation_rates = dilation_rates

        # Parallel atrous convolutions
        self.aspp_blocks = nn.ModuleList()
        for rate in dilation_rates:
            if rate == 1:
                # 1x1 convolution for local features
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # 3x3 dilated convolution for multi-scale context
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3,
                              padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.aspp_blocks.append(conv)

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final projection to combine all branches
        total_channels = out_channels * (len(dilation_rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # Apply all ASPP branches
        aspp_outs = []
        for block in self.aspp_blocks:
            aspp_outs.append(block(x))

        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w),
                                    mode='bilinear', align_corners=False)
        aspp_outs.append(global_feat)

        # Concatenate all branches
        out = torch.cat(aspp_outs, dim=1)

        # Project to output channels
        out = self.project(out)
        return out


class ASPPUNet(nn.Module):
    """
    UNet with ASPP bottleneck for multi-scale segmentation.

    Architecture:
    - Encoder: Standard UNet encoder (4 downsampling stages)
    - Bottleneck: ASPP module instead of regular DoubleConv
    - Decoder: Standard UNet decoder with skip connections

    The ASPP bottleneck captures features at multiple scales simultaneously,
    which is crucial for segmenting the optic cup within the disc.

    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (3 for background/disc/cup)
        base_channels: Number of channels in first layer (default: 64)
        dilation_rates: Dilation rates for ASPP (default: [1, 6, 12, 18])
        bilinear: Use bilinear upsampling (default: True)
    """

    def __init__(self, n_channels=3, n_classes=3, base_channels=64,
                 dilation_rates=[1, 6, 12, 18], bilinear=True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial convolution
        self.inc = DoubleConv(n_channels, base_channels)

        # Encoder path
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # Calculate bottleneck channels
        factor = 2 if bilinear else 1
        bottleneck_in = base_channels * 8
        bottleneck_out = base_channels * 16 // factor

        # ASPP Bottleneck (replaces standard Down layer)
        self.down4 = nn.MaxPool2d(2)
        self.aspp = ASPP(
            bottleneck_in,
            bottleneck_out,
            dilation_rates=dilation_rates
        )

        # Decoder path with skip connections
        self.up1 = Up(base_channels * 16, base_channels *
                      8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # Output layer
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # ASPP Bottleneck
        x5_pooled = self.down4(x4)
        x5 = self.aspp(x5_pooled)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits


class LightweightASPPUNet(nn.Module):
    """
    Lightweight version of ASPP-UNet with fewer channels and smaller ASPP.

    Useful for faster training and inference while still maintaining
    multi-scale feature extraction capability.
    """

    def __init__(self, n_channels=3, n_classes=3, base_channels=32,
                 dilation_rates=[1, 6, 12], bilinear=True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # ASPP Bottleneck
        factor = 2 if bilinear else 1
        bottleneck_in = base_channels * 8
        bottleneck_out = base_channels * 16 // factor

        self.down4 = nn.MaxPool2d(2)
        self.aspp = ASPP(
            bottleneck_in,
            bottleneck_out,
            dilation_rates=dilation_rates  # Fewer dilation rates
        )

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels *
                      8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # Output
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # ASPP Bottleneck
        x5_pooled = self.down4(x4)
        x5 = self.aspp(x5_pooled)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits
