"""
Main Training Script for Task-Aware Enhancement Model.

This script trains the enhancement model (M_E) end-to-end with the segmentation model (M_S)
using a task-aware approach where M_E is optimized to maximize the segmentation performance.

Training Strategy:
1. Load pre-trained segmentation model (M_S) - frozen during enhancement training
2. Initialize enhancement model (M_E)
3. Train M_E to minimize segmentation Dice loss on enhanced images
4. Compare against baseline (CLAHE preprocessing)

Usage:
    python train_enhancement.py --config config.yaml
    
Or with direct arguments:
    python train_enhancement.py --epochs 100 --batch_size 8 --lr 0.0002
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.enhancement_unet import EnhancementUNet, EnhancementSegmentationPipeline
from src.models.unet import UNet
from src.data_loader.dataset import GlaucomaDataset
from src.data_loader.transforms import get_training_transforms, get_validation_transforms
from src.training.task_aware_train import TaskAwareLoss, EnhancementTrainer, compute_dice_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Enhancement Model with Task-Aware Approach')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default=str(project_root),
                        help='Root directory of the project')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--enhancement_channels', type=int, default=64,
                        help='Base channels for enhancement model')
    parser.add_argument('--n_residual_blocks', type=int, default=6,
                        help='Number of residual blocks in bottleneck')
    parser.add_argument('--segmentation_checkpoint', type=str, default=None,
                        help='Path to pre-trained segmentation model checkpoint')
    parser.add_argument('--freeze_segmentation', action='store_true', default=True,
                        help='Freeze segmentation model during training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--lambda_dice', type=float, default=1.0,
                        help='Weight for Dice loss (task-aware component)')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1,
                        help='Weight for perceptual loss (quality component)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='experiments/enhancement',
                        help='Directory to save models and logs')
    parser.add_argument('--exp_name', type=str, default='task_aware_enhancement',
                        help='Experiment name')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log training metrics every N batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides CLI args)')
    
    return parser.parse_args()


def load_config(args):
    """Load configuration from YAML file if provided."""
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
    return args


def setup_experiment_dir(args):
    """Create experiment directory structure."""
    exp_dir = Path(args.save_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'visualizations').mkdir(exist_ok=True)
    
    # Save configuration
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    print(f"Experiment directory: {exp_dir}")
    return exp_dir


def load_segmentation_model(checkpoint_path, device):
    """Load pre-trained segmentation model."""
    print(f"\nLoading segmentation model from: {checkpoint_path}")
    
    # Initialize model
    model = UNet(n_channels=3, n_classes=3, base_channels=64)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"  ✓ Loaded checkpoint")
    else:
        print(f"  ⚠ No checkpoint found, using randomly initialized model")
    
    return model


def create_data_loaders(args):
    """Create training and validation data loaders."""
    print("\nCreating data loaders...")
    
    # Get transforms
    train_transform = get_training_transforms(image_size=512)
    val_transform = get_validation_transforms(image_size=512)
    
    # Create datasets
    train_dataset = GlaucomaDataset(
        root_dir=args.data_root,
        split='train',
        transform=train_transform
    )
    
    val_dataset = GlaucomaDataset(
        root_dir=args.data_root,
        split='val',
        transform=val_transform
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def visualize_enhancement(original, enhanced, mask, prediction, epoch, save_path):
    """Visualize enhancement results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(original.cpu().permute(1, 2, 0).numpy())
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Enhanced image
    axes[1].imshow(enhanced.cpu().permute(1, 2, 0).numpy())
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')
    
    # Ground truth mask
    axes[2].imshow(mask.cpu().numpy(), cmap='jet', vmin=0, vmax=2)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Prediction
    pred_mask = torch.argmax(prediction, dim=0).cpu().numpy()
    axes[3].imshow(pred_mask, cmap='jet', vmin=0, vmax=2)
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / f'epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(trainer, train_loader, epoch, writer, log_interval):
    """Train for one epoch."""
    epoch_losses = []
    epoch_metrics = {
        'dice_od': [],
        'dice_oc': [],
        'dice_mean': []
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, masks) in enumerate(pbar):
        # Training step
        loss_dict, metrics_dict = trainer.train_step(images, masks)
        
        # Accumulate metrics
        epoch_losses.append(loss_dict['total'])
        epoch_metrics['dice_od'].append(metrics_dict['dice_od'])
        epoch_metrics['dice_oc'].append(metrics_dict['dice_oc'])
        epoch_metrics['dice_mean'].append(metrics_dict['dice_mean'])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss_dict['total'],
            'dice_od': metrics_dict['dice_od'],
            'dice_oc': metrics_dict['dice_oc']
        })
        
        # Log to tensorboard
        if batch_idx % log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss_total', loss_dict['total'], step)
            writer.add_scalar('train/loss_dice', loss_dict['dice'], step)
            writer.add_scalar('train/loss_perceptual', loss_dict['perceptual'], step)
            writer.add_scalar('train/dice_od', metrics_dict['dice_od'], step)
            writer.add_scalar('train/dice_oc', metrics_dict['dice_oc'], step)
            writer.add_scalar('train/dice_mean', metrics_dict['dice_mean'], step)
    
    # Return epoch averages
    return {
        'loss': np.mean(epoch_losses),
        'dice_od': np.mean(epoch_metrics['dice_od']),
        'dice_oc': np.mean(epoch_metrics['dice_oc']),
        'dice_mean': np.mean(epoch_metrics['dice_mean'])
    }


def validate_epoch(trainer, val_loader, epoch, writer, vis_path):
    """Validate for one epoch."""
    epoch_losses = []
    epoch_metrics = {
        'dice_od': [],
        'dice_oc': [],
        'dice_mean': []
    }
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    for batch_idx, (images, masks) in enumerate(pbar):
        # Validation step
        loss_dict, metrics_dict = trainer.validate_step(images, masks)
        
        # Accumulate metrics
        epoch_losses.append(loss_dict['total'])
        epoch_metrics['dice_od'].append(metrics_dict['dice_od'])
        epoch_metrics['dice_oc'].append(metrics_dict['dice_oc'])
        epoch_metrics['dice_mean'].append(metrics_dict['dice_mean'])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss_dict['total'],
            'dice_od': metrics_dict['dice_od'],
            'dice_oc': metrics_dict['dice_oc']
        })
        
        # Visualize first batch
        if batch_idx == 0:
            with torch.no_grad():
                images_vis = images[:1].to(trainer.device)
                enhanced, logits = trainer.pipeline(images_vis)
                visualize_enhancement(
                    images[0], enhanced[0], masks[0], logits[0],
                    epoch, vis_path
                )
    
    # Log to tensorboard
    avg_metrics = {
        'loss': np.mean(epoch_losses),
        'dice_od': np.mean(epoch_metrics['dice_od']),
        'dice_oc': np.mean(epoch_metrics['dice_oc']),
        'dice_mean': np.mean(epoch_metrics['dice_mean'])
    }
    
    writer.add_scalar('val/loss', avg_metrics['loss'], epoch)
    writer.add_scalar('val/dice_od', avg_metrics['dice_od'], epoch)
    writer.add_scalar('val/dice_oc', avg_metrics['dice_oc'], epoch)
    writer.add_scalar('val/dice_mean', avg_metrics['dice_mean'], epoch)
    
    return avg_metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    args = load_config(args)
    
    print("=" * 80)
    print("Task-Aware Enhancement Model Training")
    print("=" * 80)
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(args)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    
    # Initialize models
    print("\nInitializing models...")
    
    # Enhancement model (M_E)
    enhancement_model = EnhancementUNet(
        n_channels=3,
        n_output_channels=3,
        base_channels=args.enhancement_channels,
        n_residual_blocks=args.n_residual_blocks
    )
    print(f"  Enhancement model parameters: {sum(p.numel() for p in enhancement_model.parameters()):,}")
    
    # Segmentation model (M_S)
    segmentation_model = load_segmentation_model(args.segmentation_checkpoint, args.device)
    print(f"  Segmentation model parameters: {sum(p.numel() for p in segmentation_model.parameters()):,}")
    
    # Create pipeline
    pipeline = EnhancementSegmentationPipeline(
        enhancement_model=enhancement_model,
        segmentation_model=segmentation_model,
        freeze_segmentation=args.freeze_segmentation
    )
    
    # Loss function
    criterion = TaskAwareLoss(
        lambda_dice=args.lambda_dice,
        lambda_perceptual=args.lambda_perceptual,
        n_classes=3
    )
    
    # Optimizer (only for enhancement model if segmentation is frozen)
    if args.freeze_segmentation:
        optimizer = optim.Adam(enhancement_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    else:
        optimizer = optim.Adam(pipeline.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Trainer
    trainer = EnhancementTrainer(
        pipeline=pipeline,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        scheduler=scheduler
    )
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=exp_dir / 'logs')
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    best_dice = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            trainer, train_loader, epoch, writer, args.log_interval
        )
        
        # Validate
        val_metrics = validate_epoch(
            trainer, val_loader, epoch, writer, exp_dir / 'visualizations'
        )
        
        # Print epoch summary
        print(f"\n  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dice OD: {train_metrics['dice_od']:.4f}, "
              f"Dice OC: {train_metrics['dice_oc']:.4f}, "
              f"Dice Mean: {train_metrics['dice_mean']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice OD: {val_metrics['dice_od']:.4f}, "
              f"Dice OC: {val_metrics['dice_oc']:.4f}, "
              f"Dice Mean: {val_metrics['dice_mean']:.4f}")
        
        # Update learning rate scheduler
        trainer.update_scheduler(val_metrics['dice_mean'])
        
        # Save checkpoint
        is_best = val_metrics['dice_mean'] > best_dice
        if is_best:
            best_dice = val_metrics['dice_mean']
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'enhancement_state_dict': enhancement_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }
            torch.save(checkpoint, exp_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
        
        # Regular checkpoint
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'enhancement_state_dict': enhancement_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }
            torch.save(checkpoint, exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch:03d}.pth')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered after {epoch} epochs")
            break
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Models saved in: {exp_dir / 'checkpoints'}")
    print("=" * 80)
    
    writer.close()


if __name__ == '__main__':
    main()
