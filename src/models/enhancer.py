"""
Lightweight Enhancement Network for Task-Aware ROI Enhancement.

This enhancer is designed to be trained with a frozen segmentation network,
learning to enhance fundus images specifically to improve optic disc/cup segmentation.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block for the enhancer network."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class EnhancerUNet(nn.Module):
    """
    Lightweight U-Net based enhancer for fundus images.

    This network learns to enhance images specifically for improving
    optic disc/cup segmentation. It uses residual learning to predict
    enhancement adjustments rather than reconstructing the entire image.

    Args:
        in_channels: Number of input channels (3 for RGB)
        base_channels: Number of channels in first layer (default: 16 for lightweight)
        num_residual_blocks: Number of residual blocks at bottleneck (default: 3)
        residual_learning: If True, output is added to input (learns delta)
    """

    def __init__(self, in_channels=3, base_channels=16, num_residual_blocks=3, residual_learning=True):
        super().__init__()
        self.residual_learning = residual_learning

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 8) for _ in range(num_residual_blocks)
        ])

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        # Tanh activation to constrain output range for residual learning
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Apply residual blocks
        for res_block in self.residual_blocks:
            b = res_block(b)

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.out_conv(d1)

        if self.residual_learning:
            # Learn residual: output = input + tanh(delta) * 0.05
            # Tanh constrains delta to [-1, 1], scaled by 0.05 for subtle enhancements
            # V3 FIX: Reduced from 0.5 to 0.05 (10x reduction) to prevent over-modification
            delta = self.tanh(out) * 0.05
            out = x + delta
            # Clamp to valid image range [0, 1]
            out = torch.clamp(out, 0, 1)
        else:
            # Direct reconstruction with sigmoid
            out = torch.sigmoid(out)

        return out


class SimpleCNNEnhancer(nn.Module):
    """
    Very lightweight CNN-based enhancer.

    Uses only convolutional layers without downsampling.
    Suitable for cases where computational budget is extremely limited.

    Args:
        in_channels: Number of input channels (3 for RGB)
        hidden_channels: Number of hidden channels (default: 32)
        num_layers: Number of convolutional layers (default: 5)
    """

    def __init__(self, in_channels=3, hidden_channels=32, num_layers=5):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(in_channels, hidden_channels,
                      kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels,
                          hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        layers.append(nn.Conv2d(hidden_channels,
                      in_channels, kernel_size=3, padding=1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Learn residual (V3: reduced scale from 0.5 to 0.05)
        delta = self.net(x) * 0.05  # Scale tanh output for subtle enhancements
        out = x + delta
        out = torch.clamp(out, 0, 1)
        return out


if __name__ == '__main__':
    """Test the enhancer models"""
    print("Testing EnhancerUNet...")
    model = EnhancerUNet(in_channels=3, base_channels=16,
                         num_residual_blocks=3)

    # Test with random input
    x = torch.randn(2, 3, 256, 256)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Estimate VRAM usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)

    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\n" + "="*80)
    print("\nTesting SimpleCNNEnhancer...")
    model_simple = SimpleCNNEnhancer(
        in_channels=3, hidden_channels=32, num_layers=5)

    output_simple = model_simple(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_simple.shape}")
    print(
        f"Model parameters: {sum(p.numel() for p in model_simple.parameters()):,}")

    param_size_simple = sum(p.numel() * p.element_size()
                            for p in model_simple.parameters())
    buffer_size_simple = sum(b.numel() * b.element_size()
                             for b in model_simple.buffers())
    model_size_mb_simple = (
        param_size_simple + buffer_size_simple) / (1024 ** 2)

    print(f"Model size: {model_size_mb_simple:.2f} MB")
    print(
        f"Output range: [{output_simple.min():.3f}, {output_simple.max():.3f}]")
