"""
Joint pipeline combining Enhancer and frozen Segmentation network.

The enhancer is trained to improve images specifically for the downstream
segmentation task, while the segmentation network remains frozen.
"""

import torch
import torch.nn as nn
from models.enhancer import EnhancerUNet
from models.unet import UNet


class EnhancerSegmentationPipeline(nn.Module):
    """
    Two-stage pipeline: Enhancer → Frozen Segmentation Network

    The enhancer learns to transform input images to maximize segmentation
    performance on the frozen segmentation network.

    Args:
        enhancer: Enhancement network
        segmentation_model: Pre-trained segmentation network (will be frozen)
        freeze_segmentation: If True, freeze segmentation model weights
    """

    def __init__(self, enhancer, segmentation_model, freeze_segmentation=True):
        super().__init__()

        self.enhancer = enhancer
        self.segmentation = segmentation_model

        # Freeze segmentation network
        if freeze_segmentation:
            for param in self.segmentation.parameters():
                param.requires_grad = False
            self.segmentation.eval()

    def forward(self, x, return_enhanced=False):
        """
        Forward pass through enhancer and segmentation.

        Args:
            x: Input image [B, 3, H, W]
            return_enhanced: If True, also return enhanced image

        Returns:
            segmentation_output: Segmentation logits [B, n_classes, H, W]
            enhanced_image: Enhanced image (if return_enhanced=True)
        """
        # Enhancement
        enhanced = self.enhancer(x)

        # Segmentation (frozen)
        with torch.set_grad_enabled(self.training and return_enhanced):
            seg_output = self.segmentation(enhanced)

        if return_enhanced:
            return seg_output, enhanced
        else:
            return seg_output

    def freeze_segmentation(self):
        """Freeze the segmentation network."""
        for param in self.segmentation.parameters():
            param.requires_grad = False
        self.segmentation.eval()

    def unfreeze_segmentation(self):
        """Unfreeze the segmentation network (for optional fine-tuning)."""
        for param in self.segmentation.parameters():
            param.requires_grad = True
        self.segmentation.train()


def load_pretrained_segmentation(checkpoint_path, device='cuda'):
    """
    Load pre-trained segmentation model from checkpoint.

    Args:
        checkpoint_path: Path to segmentation model checkpoint
        device: Device to load model on

    Returns:
        Loaded UNet model
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    # Create model
    model = UNet(
        n_channels=3,
        n_classes=3,
        base_channels=checkpoint.get('base_channels', 64)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded segmentation model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")

    return model


def create_enhancer_pipeline(
    segmentation_checkpoint_path,
    enhancer_type='unet',
    enhancer_base_channels=16,
    device='cuda',
    freeze_segmentation=True
):
    """
    Create enhancer-segmentation pipeline with pre-trained segmentation model.

    Args:
        segmentation_checkpoint_path: Path to pre-trained segmentation checkpoint
        enhancer_type: Type of enhancer ('unet' or 'simple')
        enhancer_base_channels: Base channels for enhancer network
        device: Device to create model on
        freeze_segmentation: Whether to freeze segmentation network

    Returns:
        EnhancerSegmentationPipeline
    """
    # Load pre-trained segmentation model
    segmentation_model = load_pretrained_segmentation(
        segmentation_checkpoint_path, device)

    # Create enhancer
    if enhancer_type == 'unet':
        from models.enhancer import EnhancerUNet
        enhancer = EnhancerUNet(
            in_channels=3,
            base_channels=enhancer_base_channels,
            num_residual_blocks=3,
            residual_learning=True
        )
    elif enhancer_type == 'simple':
        from models.enhancer import SimpleCNNEnhancer
        enhancer = SimpleCNNEnhancer(
            in_channels=3,
            hidden_channels=32,
            num_layers=5
        )
    else:
        raise ValueError(f"Unknown enhancer type: {enhancer_type}")

    enhancer = enhancer.to(device)

    # Create pipeline
    pipeline = EnhancerSegmentationPipeline(
        enhancer=enhancer,
        segmentation_model=segmentation_model,
        freeze_segmentation=freeze_segmentation
    )

    # Count parameters
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel()
                           for p in pipeline.parameters() if p.requires_grad)

    print(f"\nPipeline created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters (enhancer): {trainable_params:,}")
    print(
        f"  Frozen parameters (segmentation): {total_params - trainable_params:,}")

    return pipeline


if __name__ == '__main__':
    """Test the pipeline"""
    import sys
    from pathlib import Path

    # Example usage
    print("Testing EnhancerSegmentationPipeline...")

    # Create dummy models for testing
    from models.enhancer import EnhancerUNet
    from models.unet import UNet

    enhancer = EnhancerUNet(in_channels=3, base_channels=16)
    segmentation = UNet(n_channels=3, n_classes=3, base_channels=64)

    pipeline = EnhancerSegmentationPipeline(
        enhancer=enhancer,
        segmentation_model=segmentation,
        freeze_segmentation=True
    )

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)

    # Without enhanced image
    output = pipeline(x, return_enhanced=False)
    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {output.shape}")

    # With enhanced image
    output, enhanced = pipeline(x, return_enhanced=True)
    print(f"Enhanced image shape: {enhanced.shape}")
    print(
        f"Enhanced image range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")

    # Check parameter counts
    total = sum(p.numel() for p in pipeline.parameters())
    trainable = sum(p.numel()
                    for p in pipeline.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\nParameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable (enhancer): {trainable:,}")
    print(f"  Frozen (segmentation): {frozen:,}")
