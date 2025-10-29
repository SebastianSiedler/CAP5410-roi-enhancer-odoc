"""
Test Enhancement Model - Quick verification script
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.enhancement_unet import EnhancementUNet, EnhancementSegmentationPipeline
from src.models.unet import UNet
from src.training.task_aware_train import TaskAwareLoss, compute_dice_score


def test_enhancement_model():
    """Test enhancement model architecture."""
    print("=" * 80)
    print("Testing Enhancement U-Net Model")
    print("=" * 80)
    
    model = EnhancementUNet(
        n_channels=3,
        n_output_channels=3,
        base_channels=64,
        n_residual_blocks=6
    )
    
    # Test input
    x = torch.randn(2, 3, 512, 512)
    enhanced = model(x)
    
    print(f"\n✓ Enhancement Model Test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {enhanced.shape}")
    print(f"  Output range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check output range
    assert enhanced.min() >= 0 and enhanced.max() <= 1, "Output should be in [0, 1]"
    print("  ✓ Output range correct")
    
    return model


def test_pipeline():
    """Test end-to-end pipeline."""
    print("\n" + "=" * 80)
    print("Testing Enhancement-Segmentation Pipeline")
    print("=" * 80)
    
    # Create models
    enhancement_model = EnhancementUNet(
        n_channels=3,
        n_output_channels=3,
        base_channels=32,
        n_residual_blocks=4
    )
    
    segmentation_model = UNet(
        n_channels=3,
        n_classes=3,
        base_channels=32
    )
    
    # Create pipeline
    pipeline = EnhancementSegmentationPipeline(
        enhancement_model=enhancement_model,
        segmentation_model=segmentation_model,
        freeze_segmentation=True
    )
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    enhanced, segmentation = pipeline(x)
    
    print(f"\n✓ Pipeline Test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Enhanced shape: {enhanced.shape}")
    print(f"  Segmentation shape: {segmentation.shape}")
    print(f"  Enhancement params: {sum(p.numel() for p in enhancement_model.parameters()):,}")
    print(f"  Segmentation params: {sum(p.numel() for p in segmentation_model.parameters()):,}")
    print(f"  Total trainable params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,}")
    
    # Check frozen state
    seg_frozen = not any(p.requires_grad for p in segmentation_model.parameters())
    enh_trainable = all(p.requires_grad for p in enhancement_model.parameters())
    
    print(f"  Segmentation frozen: {seg_frozen}")
    print(f"  Enhancement trainable: {enh_trainable}")
    
    assert seg_frozen, "Segmentation should be frozen"
    assert enh_trainable, "Enhancement should be trainable"
    print("  ✓ Training mode correct")
    
    return pipeline


def test_loss_functions():
    """Test loss functions."""
    print("\n" + "=" * 80)
    print("Testing Loss Functions")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    n_classes = 3
    height, width = 256, 256
    
    logits = torch.randn(batch_size, n_classes, height, width)
    targets = torch.randint(0, n_classes, (batch_size, height, width))
    enhanced = torch.randn(batch_size, 3, height, width)
    original = torch.randn(batch_size, 3, height, width)
    
    # Test combined loss
    criterion = TaskAwareLoss(lambda_dice=1.0, lambda_perceptual=0.1, n_classes=n_classes)
    loss, loss_dict = criterion(logits, targets, enhanced, original)
    
    print(f"\n✓ Loss Function Test:")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Dice loss: {loss_dict['dice']:.4f}")
    print(f"  Perceptual loss: {loss_dict['perceptual']:.4f}")
    
    # Test metrics
    metrics = compute_dice_score(logits, targets, n_classes=n_classes)
    print(f"\n✓ Metrics Test:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test backward pass
    loss.backward()
    print("\n  ✓ Backward pass successful")


def test_training_step():
    """Test a single training step."""
    print("\n" + "=" * 80)
    print("Testing Training Step")
    print("=" * 80)
    
    # Create models
    enhancement_model = EnhancementUNet(n_channels=3, n_output_channels=3, base_channels=16)
    segmentation_model = UNet(n_channels=3, n_classes=3, base_channels=16)
    
    pipeline = EnhancementSegmentationPipeline(
        enhancement_model=enhancement_model,
        segmentation_model=segmentation_model,
        freeze_segmentation=True
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(enhancement_model.parameters(), lr=0.0002)
    criterion = TaskAwareLoss(lambda_dice=1.0, lambda_perceptual=0.1)
    
    # Dummy data
    images = torch.randn(2, 3, 256, 256)
    masks = torch.randint(0, 3, (2, 256, 256))
    
    # Training step
    pipeline.train()
    optimizer.zero_grad()
    
    enhanced, segmentation = pipeline(images)
    loss, loss_dict = criterion(segmentation, masks, enhanced, images)
    
    loss.backward()
    optimizer.step()
    
    print(f"\n✓ Training Step Test:")
    print(f"  Loss: {loss_dict['total']:.4f}")
    print(f"  Gradients computed: {enhancement_model.inc.conv[0].weight.grad is not None}")
    print(f"  Segmentation frozen: {segmentation_model.inc.double_conv[0].weight.grad is None}")
    print("  ✓ Training step successful")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ENHANCEMENT MODEL TEST SUITE")
    print("=" * 80 + "\n")
    
    try:
        # Test individual components
        test_enhancement_model()
        test_pipeline()
        test_loss_functions()
        test_training_step()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ TEST FAILED: {str(e)}")
        print("=" * 80 + "\n")
        raise


if __name__ == '__main__':
    main()
