"""
Hybrid U-Net Enhancement Model for Task-Aware Image Enhancement.

This model combines U-Net architecture with Residual Blocks (inspired by CycleGAN generators)
to enhance fundus images for optimal optic disc/cup segmentation performance.

Key features:
- U-Net backbone for image-to-image reconstruction
- Bottleneck Residual Blocks for efficiency
- Leaky ReLU activation to preserve low-contrast information
- Designed for end-to-end training with downstream segmentation model
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Bottleneck Residual Block inspired by CycleGAN generators.
    Reduces parameters while maintaining performance.
    """
    
    def __init__(self, channels, use_dropout=False):
        super().__init__()
        
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        
        layers.extend([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.block(x)  # Residual connection


class EnhancementConv(nn.Module):
    """
    Double convolution block with LeakyReLU activation.
    (Conv -> InstanceNorm -> LeakyReLU) * 2
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block: MaxPool -> EnhancementConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            EnhancementConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block: Upsample -> Concatenate -> EnhancementConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = EnhancementConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate skip connection from encoder
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EnhancementUNet(nn.Module):
    """
    Hybrid U-Net model for task-aware image enhancement.
    
    Architecture:
    - U-Net backbone for spatial information preservation
    - Residual blocks in the bottleneck for efficiency
    - LeakyReLU activations throughout to preserve low-contrast details
    - Instance normalization for better generalization
    
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_output_channels: Number of output channels (3 for enhanced RGB)
        base_channels: Number of channels in first layer (default: 64)
        n_residual_blocks: Number of residual blocks in bottleneck (default: 6)
    """
    
    def __init__(
        self, 
        n_channels=3, 
        n_output_channels=3, 
        base_channels=64,
        n_residual_blocks=6
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_output_channels = n_output_channels
        
        # Initial convolution
        self.inc = EnhancementConv(n_channels, base_channels)
        
        # Encoder (downsampling path)
        self.enc1 = EncoderBlock(base_channels, base_channels * 2)
        self.enc2 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc3 = EncoderBlock(base_channels * 4, base_channels * 8)
        self.enc4 = EncoderBlock(base_channels * 8, base_channels * 16)
        
        # Bottleneck with residual blocks
        residual_blocks = []
        for _ in range(n_residual_blocks):
            residual_blocks.append(ResidualBlock(base_channels * 16, use_dropout=False))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Decoder (upsampling path)
        self.dec1 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.dec2 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.dec3 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.dec4 = DecoderBlock(base_channels * 2, base_channels)
        
        # Output layer: maps to RGB image
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, n_output_channels, kernel_size=1),
            nn.Tanh()  # Output in range [-1, 1], can be scaled to [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass through the enhancement network.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Enhanced image tensor [B, C, H, W]
        """
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        # Bottleneck
        x5 = self.residual_blocks(x5)
        
        # Decoder with skip connections
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        
        # Output
        enhanced = self.outc(x)
        
        # Scale from [-1, 1] to [0, 1]
        enhanced = (enhanced + 1) / 2
        
        return enhanced


class EnhancementSegmentationPipeline(nn.Module):
    """
    End-to-end pipeline combining Enhancement Model (M_E) and Segmentation Model (M_S).
    
    This allows for task-aware training where the enhancement model is optimized
    to maximize the segmentation performance of the downstream model.
    
    Args:
        enhancement_model: EnhancementUNet model (M_E)
        segmentation_model: UNet segmentation model (M_S)
        freeze_segmentation: If True, only train enhancement model. If False, train both.
    """
    
    def __init__(
        self, 
        enhancement_model,
        segmentation_model,
        freeze_segmentation=True
    ):
        super().__init__()
        self.enhancement_model = enhancement_model
        self.segmentation_model = segmentation_model
        
        # Optionally freeze segmentation model weights
        if freeze_segmentation:
            for param in self.segmentation_model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through both models.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            enhanced: Enhanced image [B, C, H, W]
            segmentation: Segmentation logits [B, n_classes, H, W]
        """
        # Step 1: Enhance the image
        enhanced = self.enhancement_model(x)
        
        # Step 2: Segment the enhanced image
        segmentation = self.segmentation_model(enhanced)
        
        return enhanced, segmentation
    
    def enhance_only(self, x):
        """Only apply enhancement without segmentation."""
        return self.enhancement_model(x)
    
    def segment_only(self, x):
        """Only apply segmentation without enhancement."""
        return self.segmentation_model(x)


if __name__ == '__main__':
    """Test the enhancement model"""
    print("=" * 80)
    print("Testing Enhancement U-Net Model")
    print("=" * 80)
    
    # Test enhancement model alone
    model = EnhancementUNet(
        n_channels=3,
        n_output_channels=3,
        base_channels=64,
        n_residual_blocks=6
    )
    
    # Test with random input
    x = torch.randn(2, 3, 512, 512)
    enhanced = model(x)
    
    print(f"\nEnhancement Model:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {enhanced.shape}")
    print(f"  Output range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test combined pipeline
    print("\n" + "=" * 80)
    print("Testing Enhancement-Segmentation Pipeline")
    print("=" * 80)
    
    from unet import UNet
    
    enhancement_model = EnhancementUNet(n_channels=3, n_output_channels=3, base_channels=32)
    segmentation_model = UNet(n_channels=3, n_classes=3, base_channels=32)
    
    pipeline = EnhancementSegmentationPipeline(
        enhancement_model=enhancement_model,
        segmentation_model=segmentation_model,
        freeze_segmentation=True
    )
    
    enhanced, segmentation = pipeline(x)
    
    print(f"\nPipeline:")
    print(f"  Input shape: {x.shape}")
    print(f"  Enhanced shape: {enhanced.shape}")
    print(f"  Segmentation shape: {segmentation.shape}")
    print(f"  Enhancement params: {sum(p.numel() for p in enhancement_model.parameters()):,}")
    print(f"  Segmentation params: {sum(p.numel() for p in segmentation_model.parameters()):,}")
    print(f"  Total trainable params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,}")
    print(f"  Segmentation frozen: {not any(p.requires_grad for p in segmentation_model.parameters())}")
    
    print("\n" + "=" * 80)
