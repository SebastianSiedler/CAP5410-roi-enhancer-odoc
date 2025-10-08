"""
ResNet-UNet Architecture

Combines ResNet encoder (pretrained on ImageNet) with U-Net decoder.
Benefits:
- Pretrained features: Better initialization than random weights
- Deeper network: Can learn more complex features
- Residual connections: Easier to train deep networks
- Skip connections: Preserve spatial information (U-Net style)

This architecture is particularly effective for medical imaging where:
1. Limited training data benefits from transfer learning
2. Complex anatomical structures need deep feature extraction
3. Precise segmentation requires spatial information preservation
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class ResNetUNet(nn.Module):
    """
    ResNet-UNet for binary segmentation.
    
    Architecture:
    - Encoder: ResNet (pretrained on ImageNet)
    - Decoder: Upsampling + Conv blocks with skip connections
    - Output: 2 channels (disc, cup)
    
    Args:
        n_classes: Number of output classes (default: 2 for disc/cup)
        backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
        pretrained: Use ImageNet pretrained weights
    """
    
    def __init__(
        self,
        n_classes: int = 2,
        backbone: str = 'resnet34',
        pretrained: bool = True
    ):
        super(ResNetUNet, self).__init__()
        
        self.n_classes = n_classes
        self.backbone_name = backbone
        
        # Load pretrained ResNet
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Encoder (ResNet layers)
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels, /2
        
        self.pool1 = resnet.maxpool  # /2 (total /4)
        self.encoder2 = resnet.layer1  # 64/256 channels
        self.encoder3 = resnet.layer2  # 128/512 channels, /2 (total /8)
        self.encoder4 = resnet.layer3  # 256/1024 channels, /2 (total /16)
        self.encoder5 = resnet.layer4  # 512/2048 channels, /2 (total /32)
        
        # Decoder (Upsampling path)
        # decoder(in_channels, out_channels, skip_channels)
        self.decoder5 = DecoderBlock(encoder_channels[4], encoder_channels[3], encoder_channels[3])
        self.decoder4 = DecoderBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.decoder3 = DecoderBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.decoder2 = DecoderBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])
        
        # Last decoder doesn't have skip connection, just upsample
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0], kernel_size=2, stride=2),
            nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
        
        # Store encoder channels for reference
        self.encoder_channels = encoder_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Output tensor (B, n_classes, H, W)
        """
        # Encoder path
        enc1 = self.encoder1(x)      # 64, H/2, W/2
        enc1_pool = self.pool1(enc1)  # 64, H/4, W/4
        enc2 = self.encoder2(enc1_pool)  # 64/256, H/4, W/4
        enc3 = self.encoder3(enc2)    # 128/512, H/8, W/8
        enc4 = self.encoder4(enc3)    # 256/1024, H/16, W/16
        enc5 = self.encoder5(enc4)    # 512/2048, H/32, W/32
        
        # Decoder path with skip connections
        dec5 = self.decoder5(enc5, enc4)  # H/16, W/16
        dec4 = self.decoder4(dec5, enc3)  # H/8, W/8
        dec3 = self.decoder3(dec4, enc2)  # H/4, W/4
        dec2 = self.decoder2(dec3, enc1)  # H/2, W/2
        dec1 = self.decoder1(dec2)  # H, W (back to original size)
        
        # Final output
        out = self.final_conv(dec1)
        
        return out
    
    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning."""
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False
        for param in self.encoder5.parameters():
            param.requires_grad = False
        print("Encoder frozen. Only decoder will be trained.")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights for full training."""
        for param in self.encoder1.parameters():
            param.requires_grad = True
        for param in self.encoder2.parameters():
            param.requires_grad = True
        for param in self.encoder3.parameters():
            param.requires_grad = True
        for param in self.encoder4.parameters():
            param.requires_grad = True
        for param in self.encoder5.parameters():
            param.requires_grad = True
        print("Encoder unfrozen. Full model will be trained.")


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connections.
    
    Architecture:
    1. Upsample input (2x)
    2. Concatenate with skip connection from encoder
    3. Two conv layers with BatchNorm and ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = None):
        super(DecoderBlock, self).__init__()
        
        # If skip_channels not provided, assume same as out_channels
        if skip_channels is None:
            skip_channels = out_channels
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
        
        # Convolution layers (after concat: out_channels + skip_channels)
        self.conv1 = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder (can be None for last layer)
            
        Returns:
            Decoded features
        """
        x = self.upsample(x)
        
        if skip is not None:
            # Match spatial dimensions if needed
            if x.size()[2:] != skip.size()[2:]:
                x = nn.functional.interpolate(
                    x,
                    size=skip.size()[2:],
                    mode='bilinear',
                    align_corners=True
                )
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test the model."""
    
    print("Testing ResNet-UNet architectures...\n")
    
    # Test different backbones
    backbones = ['resnet18', 'resnet34', 'resnet50']
    
    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"{backbone.upper()} Backbone")
        print('='*60)
        
        model = ResNetUNet(
            n_classes=2,
            backbone=backbone,
            pretrained=False  # Don't download for testing
        )
        
        # Count parameters
        total_params = count_parameters(model)
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Test forward pass
        x = torch.randn(2, 3, 512, 512)
        print(f"Input shape: {x.shape}")
        
        with torch.no_grad():
            output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Test encoder freezing
        model.freeze_encoder()
        frozen_params = count_parameters(model)
        print(f"Trainable after freezing encoder: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        
        model.unfreeze_encoder()
        unfrozen_params = count_parameters(model)
        print(f"Trainable after unfreezing: {unfrozen_params:,} ({unfrozen_params/1e6:.2f}M)")
    
    print(f"\n{'='*60}")
    print("✅ All tests passed!")
    print('='*60)
