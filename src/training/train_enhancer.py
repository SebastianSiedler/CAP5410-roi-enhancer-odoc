"""
Training script for the task-aware ROI enhancer network.

Trains the enhancer with a frozen segmentation network using a combined loss:
- Segmentation loss (primary objective)
- Perceptual similarity loss (regularization)
- Total variation loss (smoothness)
"""

from training.train import DiceLoss, CombinedLoss, calculate_iou
from models.joint_pipeline import create_enhancer_pipeline
from data_loader.dataset import get_dataloaders
from data_loader.transforms import get_training_transforms, get_validation_transforms
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for smoothness regularization.
    Penalizes rapid changes in pixel values to prevent artifacts.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: Image tensor [B, C, H, W]
        """
        # Calculate differences
        diff_i = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_j = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        # Sum over all dimensions
        tv_loss = diff_i.sum() + diff_j.sum()

        # Normalize by batch size and image dimensions
        tv_loss = tv_loss / (x.shape[0] * x.shape[2] * x.shape[3])

        return tv_loss


class EnhancerLoss(nn.Module):
    """
    Combined loss for training the enhancer network.

    Components:
    1. Segmentation loss (Dice + CE) - Primary objective
    2. Perceptual/L1 loss - Keep enhanced image similar to input
    3. Total Variation loss - Prevent artifacts

    Args:
        seg_weight: Weight for segmentation loss (default: 1.0)
        perceptual_weight: Weight for L1/perceptual loss (default: 0.1)
        tv_weight: Weight for total variation loss (default: 0.01)
        class_weights: Weights for each class in segmentation loss
        device: Device for class weights
    """

    def __init__(self, seg_weight=1.0, perceptual_weight=0.1, tv_weight=0.01,
                 class_weights=None, device=None):
        super().__init__()

        self.seg_weight = seg_weight
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight

        # Segmentation loss (same as in UNet training)
        self.seg_loss = CombinedLoss(
            ce_weight=0.5,
            dice_weight=0.5,
            class_weights=class_weights,
            device=device
        )

        # Perceptual similarity (L1 loss between input and enhanced)
        self.l1_loss = nn.L1Loss()

        # Total variation for smoothness
        self.tv_loss = TotalVariationLoss()

    def forward(self, seg_output, target_mask, enhanced_image, original_image):
        """
        Calculate combined loss.

        Args:
            seg_output: Segmentation network output [B, C, H, W]
            target_mask: Ground truth mask [B, H, W]
            enhanced_image: Enhanced image from enhancer [B, 3, H, W]
            original_image: Original input image [B, 3, H, W]

        Returns:
            total_loss, loss_dict
        """
        # Segmentation loss (primary objective)
        seg_loss = self.seg_loss(seg_output, target_mask)

        # Perceptual similarity loss (prevent drastic changes)
        perceptual_loss = self.l1_loss(enhanced_image, original_image)

        # Total variation loss (smoothness)
        tv_loss = self.tv_loss(enhanced_image)

        # Combined loss
        total_loss = (
            self.seg_weight * seg_loss +
            self.perceptual_weight * perceptual_loss +
            self.tv_weight * tv_loss
        )

        # Return loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'segmentation': seg_loss.item(),
            'perceptual': perceptual_loss.item(),
            'tv': tv_loss.item()
        }

        return total_loss, loss_dict


def train_epoch(pipeline, train_loader, criterion, optimizer, device):
    """Train enhancer for one epoch (segmentation network is frozen)"""
    pipeline.train()
    pipeline.freeze_segmentation()  # Ensure segmentation stays frozen

    total_loss = 0
    total_iou = np.zeros(3)
    loss_components = {'segmentation': 0, 'perceptual': 0, 'tv': 0}

    pbar = tqdm(train_loader, desc='Training Enhancer')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass through enhancer + frozen segmentation
        optimizer.zero_grad()
        seg_output, enhanced_images = pipeline(images, return_enhanced=True)

        # Calculate loss
        loss, loss_dict = criterion(seg_output, masks, enhanced_images, images)

        # Backward pass (only enhancer gradients)
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]

        iou = calculate_iou(seg_output, masks)
        total_iou += np.array(iou)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'seg_loss': f'{loss_dict["segmentation"]:.4f}',
            'iou_disc': f'{iou[1]:.4f}',
            'iou_cup': f'{iou[2]:.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_loss_components = {k: v / len(train_loader)
                           for k, v in loss_components.items()}

    return avg_loss, avg_iou, avg_loss_components


@torch.no_grad()
def validate_epoch(pipeline, val_loader, criterion, device):
    """Validate enhancer for one epoch"""
    pipeline.eval()

    total_loss = 0
    total_iou = np.zeros(3)
    loss_components = {'segmentation': 0, 'perceptual': 0, 'tv': 0}

    pbar = tqdm(val_loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        seg_output, enhanced_images = pipeline(images, return_enhanced=True)

        # Calculate loss
        loss, loss_dict = criterion(seg_output, masks, enhanced_images, images)

        # Metrics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]

        iou = calculate_iou(seg_output, masks)
        total_iou += np.array(iou)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou_disc': f'{iou[1]:.4f}',
            'iou_cup': f'{iou[2]:.4f}'
        })

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_loss_components = {k: v / len(val_loader)
                           for k, v in loss_components.items()}

    return avg_loss, avg_iou, avg_loss_components


def train_enhancer(
    root_dir,
    segmentation_checkpoint,
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    image_size=256,
    enhancer_type='unet',
    enhancer_base_channels=16,
    num_workers=4,
    device=None,
    save_dir='checkpoints_enhancer',
    filter_incomplete=True,
    use_clahe=False,
    seg_weight=1.0,
    perceptual_weight=0.1,
    tv_weight=0.01
):
    """
    Main training function for the enhancer network.

    Args:
        root_dir: Project root directory
        segmentation_checkpoint: Path to pre-trained segmentation model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        image_size: Input image size
        enhancer_type: 'unet' or 'simple'
        enhancer_base_channels: Base channels for enhancer
        num_workers: Number of data loading workers
        device: Device to train on
        save_dir: Directory to save checkpoints
        filter_incomplete: Filter images without all 3 classes
        use_clahe: Apply CLAHE (not recommended with enhancer)
        seg_weight: Weight for segmentation loss
        perceptual_weight: Weight for perceptual loss
        tv_weight: Weight for TV loss

    Returns:
        pipeline: Trained pipeline
        history: Training history
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_train=get_training_transforms(
            image_size=image_size, use_clahe=use_clahe),
        transform_val=get_validation_transforms(
            image_size=image_size, use_clahe=use_clahe),
        filter_incomplete=filter_incomplete
    )

    # Create pipeline with frozen segmentation
    print("\nCreating enhancer-segmentation pipeline...")
    pipeline = create_enhancer_pipeline(
        segmentation_checkpoint_path=segmentation_checkpoint,
        enhancer_type=enhancer_type,
        enhancer_base_channels=enhancer_base_channels,
        device=device,
        freeze_segmentation=True
    )

    # Loss and optimizer
    class_weights = torch.tensor(
        [1.0, 1.0, 2.0], device=device)  # Same as UNet training

    criterion = EnhancerLoss(
        seg_weight=seg_weight,
        perceptual_weight=perceptual_weight,
        tv_weight=tv_weight,
        class_weights=class_weights,
        device=device
    )

    # Optimizer only for enhancer parameters
    optimizer = optim.AdamW(
        pipeline.enhancer.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_seg_loss': [],
        'val_seg_loss': [],
        'train_perceptual_loss': [],
        'val_perceptual_loss': [],
        'train_tv_loss': [],
        'val_tv_loss': [],
        'train_iou_bg': [],
        'train_iou_disc': [],
        'train_iou_cup': [],
        'val_iou_bg': [],
        'val_iou_disc': [],
        'val_iou_cup': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_val_iou_disc = 0.0

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_iou, train_loss_comp = train_epoch(
            pipeline, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_iou, val_loss_comp = validate_epoch(
            pipeline, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_seg_loss'].append(train_loss_comp['segmentation'])
        history['val_seg_loss'].append(val_loss_comp['segmentation'])
        history['train_perceptual_loss'].append(train_loss_comp['perceptual'])
        history['val_perceptual_loss'].append(val_loss_comp['perceptual'])
        history['train_tv_loss'].append(train_loss_comp['tv'])
        history['val_tv_loss'].append(val_loss_comp['tv'])
        history['train_iou_bg'].append(train_iou[0])
        history['train_iou_disc'].append(train_iou[1])
        history['train_iou_cup'].append(train_iou[2])
        history['val_iou_bg'].append(val_iou[0])
        history['val_iou_disc'].append(val_iou[1])
        history['val_iou_cup'].append(val_iou[2])
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(
            f"  Seg Loss:   Train {train_loss_comp['segmentation']:.4f} | Val {val_loss_comp['segmentation']:.4f}")
        print(f"  IoU (Disc): Train {train_iou[1]:.4f} | Val {val_iou[1]:.4f}")
        print(f"  IoU (Cup):  Train {train_iou[2]:.4f} | Val {val_iou[2]:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'enhancer_state_dict': pipeline.enhancer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'enhancer_type': enhancer_type,
                'enhancer_base_channels': enhancer_base_channels,
                'segmentation_checkpoint': str(segmentation_checkpoint),
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

        # Save best IoU model
        if val_iou[1] > best_val_iou_disc:
            best_val_iou_disc = val_iou[1]
            checkpoint_path = save_dir / 'best_model_iou.pth'
            torch.save({
                'epoch': epoch,
                'enhancer_state_dict': pipeline.enhancer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'enhancer_type': enhancer_type,
                'enhancer_base_channels': enhancer_base_channels,
                'segmentation_checkpoint': str(segmentation_checkpoint),
            }, checkpoint_path)
            print(f"  ✓ Saved best IoU model (disc IoU: {val_iou[1]:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'enhancer_state_dict': pipeline.enhancer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'enhancer_type': enhancer_type,
                'enhancer_base_channels': enhancer_base_channels,
                'segmentation_checkpoint': str(segmentation_checkpoint),
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best disc IoU: {best_val_iou_disc:.4f}")

    # Save training history
    history_path = save_dir / 'training_history_enhancer.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    return pipeline, history


if __name__ == '__main__':
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train task-aware enhancer network')
    parser.add_argument('--root_dir', type=str,
                        default='/home/robolab/dev/CAP5410-roi-enhancer-odoc',
                        help='Project root directory')
    parser.add_argument('--seg_checkpoint', type=str,
                        default='checkpoints_clahe/best_model.pth',
                        help='Path to pre-trained segmentation checkpoint')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--enhancer_type', type=str, default='unet',
                        choices=['unet', 'simple'],
                        help='Enhancer architecture type')
    parser.add_argument('--save_dir', type=str, default='checkpoints_enhancer',
                        help='Directory to save checkpoints')
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight for segmentation loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Weight for perceptual/L1 loss')
    parser.add_argument('--tv_weight', type=float, default=0.01,
                        help='Weight for total variation loss')

    args = parser.parse_args()

    # Train enhancer
    pipeline, history = train_enhancer(
        root_dir=args.root_dir,
        segmentation_checkpoint=args.seg_checkpoint,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        enhancer_type=args.enhancer_type,
        save_dir=args.save_dir,
        seg_weight=args.seg_weight,
        perceptual_weight=args.perceptual_weight,
        tv_weight=args.tv_weight
    )
