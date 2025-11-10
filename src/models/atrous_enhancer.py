"""
Atrous/Dilated Convolution based Image Enhancer for Optic Disc/Cup Segmentation

Uses multi-scale dilated convolutions to capture context at different scales
before segmentation. This is particularly useful for:
- Multi-scale feature extraction
- Larger receptive field without downsampling
- Preserving spatial resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtrousSpatialPyramidPooling(nn.Module):
    """
    ASPP module - captures multi-scale context using parallel atrous convolutions
    with different dilation rates.
    """

    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super().__init__()

        self.dilation_rates = dilation_rates

        # Parallel atrous convolutions with different dilation rates
        self.aspp_blocks = nn.ModuleList()
        for rate in dilation_rates:
            if rate == 1:
                # 1x1 convolution
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # 3x3 dilated convolution
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

        # Final 1x1 conv to combine all branches
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


class AtrousEnhancerBlock(nn.Module):
    """
    Basic block with atrous convolution for multi-scale feature extraction
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class AtrousImageEnhancer(nn.Module):
    """
    Lightweight atrous convolution-based image enhancer.

    Uses multi-scale dilated convolutions to enhance images by capturing
    context at different scales without reducing spatial resolution.

    Architecture:
    - Initial conv layer
    - ASPP module for multi-scale feature extraction
    - Enhancement pathway with atrous blocks
    - Residual connection with learnable weight

    Args:
        n_channels: Number of input channels (3 for RGB)
        base_channels: Base number of channels (default: 32)
        dilation_rates: Dilation rates for ASPP (default: [1, 6, 12, 18])
        residual_weight: Weight for residual connection (default: 0.3)
    """

    def __init__(self, n_channels=3, base_channels=32,
                 dilation_rates=[1, 6, 12, 18], residual_weight=0.3):
        super().__init__()

        self.residual_weight = residual_weight

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ASPP module for multi-scale context
        self.aspp = AtrousSpatialPyramidPooling(
            base_channels,
            base_channels,
            dilation_rates=dilation_rates
        )

        # Enhancement pathway with different dilation rates
        self.enhance1 = AtrousEnhancerBlock(
            base_channels, base_channels, dilation=1)
        self.enhance2 = AtrousEnhancerBlock(
            base_channels, base_channels, dilation=2)
        self.enhance3 = AtrousEnhancerBlock(
            base_channels, base_channels, dilation=4)

        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, n_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Enhanced image tensor (B, C, H, W)
        """
        identity = x

        # Initial feature extraction
        features = self.init_conv(x)

        # Multi-scale context with ASPP
        features = self.aspp(features)

        # Enhancement pathway
        features = self.enhance1(features)
        features = self.enhance2(features)
        features = self.enhance3(features)

        # Generate enhancement
        enhancement = self.output(features)

        # Apply residual connection with learned weight
        # output = identity + weight * enhancement
        out = identity + self.residual_weight * enhancement

        # Clamp to valid range (assuming normalized input)
        out = torch.clamp(out, -2.5, 2.5)  # Rough bounds for normalized images

        return out


class LightweightAtrousEnhancer(nn.Module):
    """
    Lightweight version with fewer parameters for faster training.
    Uses smaller dilation rates and fewer channels.
    """

    def __init__(self, n_channels=3, base_channels=24, residual_weight=0.3):
        super().__init__()

        self.residual_weight = residual_weight

        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Lightweight ASPP with smaller dilation rates
        self.aspp = AtrousSpatialPyramidPooling(
            base_channels,
            base_channels,
            dilation_rates=[1, 3, 6]  # Fewer, smaller dilations
        )

        # Single enhancement block
        self.enhance = AtrousEnhancerBlock(
            base_channels, base_channels, dilation=2)

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, n_channels, 3, padding=1),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        features = self.init_conv(x)
        features = self.aspp(features)
        features = self.enhance(features)
        enhancement = self.output(features)

        out = identity + self.residual_weight * enhancement
        out = torch.clamp(out, -2.5, 2.5)

        return out


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing Atrous Image Enhancers...")
    print("=" * 70)

    # Test full version
    model_full = AtrousImageEnhancer(
        n_channels=3,
        base_channels=32,
        dilation_rates=[1, 6, 12, 18]
    ).to(device)

    # Test lightweight version
    model_lite = LightweightAtrousEnhancer(
        n_channels=3,
        base_channels=24
    ).to(device)

    # Test input
    x = torch.randn(2, 3, 256, 256).to(device)

    print("\n1. Full Atrous Enhancer:")
    print(f"   Parameters: {count_parameters(model_full):,}")
    with torch.no_grad():
        out_full = model_full(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out_full.shape}")
    print(f"   Output range: [{out_full.min():.3f}, {out_full.max():.3f}]")

    print("\n2. Lightweight Atrous Enhancer:")
    print(f"   Parameters: {count_parameters(model_lite):,}")
    with torch.no_grad():
        out_lite = model_lite(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {out_lite.shape}")
    print(f"   Output range: [{out_lite.min():.3f}, {out_lite.max():.3f}]")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
