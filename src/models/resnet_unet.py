"""
ResNet-based UNet architecture for optic disc/cup segmentation.

Uses ResNet34 as the encoder (backbone) with pretrained ImageNet weights.
This architecture is optimized for medical imaging and fits within GTX 2080 Ti VRAM.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """ResNet encoder using pretrained ResNet34"""

    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=pretrained)

        # Extract encoder layers
        # Input: 3 -> 64 channels
        self.conv1 = resnet.conv1  # 64 channels, stride 2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride 2

        # ResNet blocks (each may downsample)
        self.layer1 = resnet.layer1  # 64 channels, no downsample
        self.layer2 = resnet.layer2  # 128 channels, stride 2
        self.layer3 = resnet.layer3  # 256 channels, stride 2
        self.layer4 = resnet.layer4  # 512 channels, stride 2

    def forward(self, x):
        """
        Returns encoder features at different scales for skip connections.

        Returns:
            List of features [x1, x2, x3, x4, x5] at different scales
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # 64 channels, H/2, W/2

        x = self.maxpool(x)
        x2 = self.layer1(x)  # 64 channels, H/4, W/4
        x3 = self.layer2(x2)  # 128 channels, H/8, W/8
        x4 = self.layer3(x3)  # 256 channels, H/16, W/16
        x5 = self.layer4(x4)  # 512 channels, H/32, W/32

        return [x1, x2, x3, x4, x5]


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        # Convolution after concatenation
        self.conv1 = nn.Conv2d(
            in_channels // 2 + skip_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        """
        Args:
            x: Input from previous layer
            skip: Skip connection from encoder
        """
        x = self.upsample(x)

        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class ResNetUNet(nn.Module):
    """
    ResNet-based UNet for semantic segmentation.

    Uses ResNet34 encoder with pretrained ImageNet weights.
    Optimized for GTX 2080 Ti (11GB VRAM) with medical imaging.

    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (3 for background, disc, cup)
        pretrained: Use pretrained ImageNet weights for encoder
    """

    def __init__(self, n_channels=3, n_classes=3, pretrained=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (ResNet34)
        self.encoder = ResNetEncoder(pretrained=pretrained)

        # Decoder
        # x5: 512 channels -> x4: 256 channels
        self.decoder4 = DecoderBlock(512, 256, 256)

        # x4: 256 channels -> x3: 128 channels
        self.decoder3 = DecoderBlock(256, 128, 128)

        # x3: 128 channels -> x2: 64 channels
        self.decoder2 = DecoderBlock(128, 64, 64)

        # x2: 64 channels -> x1: 64 channels
        self.decoder1 = DecoderBlock(64, 64, 64)

        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2
        )

        # Output layer
        self.outc = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            logits: Output tensor [B, n_classes, H, W]
        """
        # Encoder
        encoder_features = self.encoder(x)
        x1, x2, x3, x4, x5 = encoder_features

        # Decoder with skip connections
        d4 = self.decoder4(x5, x4)  # 256 channels
        d3 = self.decoder3(d4, x3)  # 128 channels
        d2 = self.decoder2(d3, x2)  # 64 channels
        d1 = self.decoder1(d2, x1)  # 64 channels

        # Final upsampling and output
        d0 = self.final_upsample(d1)  # Back to original resolution
        logits = self.outc(d0)

        return logits


class ResNetUNetLite(nn.Module):
    """
    Lightweight ResNet-based UNet for even lower VRAM usage.

    Uses ResNet18 encoder (lighter than ResNet34).
    Best for limited VRAM or larger batch sizes.

    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (3 for background, disc, cup)
        pretrained: Use pretrained ImageNet weights for encoder
    """

    def __init__(self, n_channels=3, n_classes=3, pretrained=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Extract encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Decoder (same structure as ResNetUNet)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        self.final_upsample = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2)

        self.outc = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x

        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # Decoder
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        # Output
        d0 = self.final_upsample(d1)
        logits = self.outc(d0)

        return logits


if __name__ == '__main__':
    """Test the models"""
    print("Testing ResNetUNet (ResNet34)...")
    model = ResNetUNet(n_channels=3, n_classes=3, pretrained=False)

    # Test with random input
    x = torch.randn(2, 3, 512, 512)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Estimate VRAM usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)

    # Rough estimate: model + activations + gradients + optimizer states
    estimated_vram_gb = (model_size_mb * 4) / 1024  # Rough multiplier
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Estimated VRAM (batch_size=2): ~{estimated_vram_gb:.2f} GB")

    print("\n" + "="*80)
    print("\nTesting ResNetUNetLite (ResNet18)...")
    model_lite = ResNetUNetLite(n_channels=3, n_classes=3, pretrained=False)

    output_lite = model_lite(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_lite.shape}")
    print(
        f"Model parameters: {sum(p.numel() for p in model_lite.parameters()):,}")

    param_size_lite = sum(p.numel() * p.element_size()
                          for p in model_lite.parameters())
    buffer_size_lite = sum(b.numel() * b.element_size()
                           for b in model_lite.buffers())
    model_size_mb_lite = (param_size_lite + buffer_size_lite) / (1024 ** 2)
    estimated_vram_gb_lite = (model_size_mb_lite * 4) / 1024

    print(f"Model size: {model_size_mb_lite:.2f} MB")
    print(f"Estimated VRAM (batch_size=2): ~{estimated_vram_gb_lite:.2f} GB")
