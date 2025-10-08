"""
Main training script for EE-TransUNet segmentation
Trains EE-TransUNet on REFUGE dataset for optic disc/cup segmentation
"""
from utils.metrics import batch_metrics, MetricsTracker
from utils.loss_functions import CombinedSegmentationLoss
from models.ee_transunet import VisionTransformer, CONFIGS
from data_loader.dataset import RetinaDataset, RetinaDatasetValidation
import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train EE-TransUNet for OD/OC segmentation')

    # Data parameters
    parser.add_argument('--data_dir', type=str,
                        default='datasets/REFUGE',
                        help='Path to REFUGE dataset directory')
    parser.add_argument('--train_csv', type=str,
                        default='datasets/REFUGE/REFUGETrain.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str,
                        default='datasets/REFUGE/REFUGE1Val.csv',
                        help='Path to validation CSV file')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='ViT-Tiny',
                        choices=['ViT-Tiny', 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 
                                'R50-ViT-B_16', 'R50-ViT-L_16', 'testing'],
                        help='EE-TransUNet model variant')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                        help='Vision Transformer patch size')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (EE-TransUNet default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size (square)')

    # Loss parameters
    parser.add_argument('--lambda_dice', type=float, default=0.5,
                        help='Weight for Dice loss')
    parser.add_argument('--lambda_bce', type=float, default=0.5,
                        help='Weight for BCE loss')

    # Checkpoint parameters
    parser.add_argument('--save_dir', type=str,
                        default='experiments/ee_transunet',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained weights (ImageNet21k)')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu/mps)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    return parser.parse_args()


def setup_training(args):
    """Setup directories, device, and save configuration"""
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    return device


def create_dataloaders(args):
    """Create training and validation dataloaders"""
    print("Creating dataloaders...")

    # Training dataset (no CLAHE - EE-TransUNet doesn't use it in original repo)
    train_dataset = RetinaDataset(
        root_dir=args.data_dir,
        csv_file=args.train_csv,
        target_size=(args.img_size, args.img_size),
        use_cropped=True,  # Use cropped datasets as in original code
        use_clahe=False,  # EE-TransUNet doesn't use CLAHE preprocessing
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Validation dataset
    val_dataset = RetinaDatasetValidation(
        root_dir=args.data_dir,
        csv_file=args.val_csv,
        target_size=(args.img_size, args.img_size),
        use_cropped=True,  # Use cropped datasets as in original code
        use_clahe=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    metrics_tracker = MetricsTracker()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            metrics = batch_metrics(outputs, masks)
            metrics['loss'] = loss.item()
            metrics_tracker.update(metrics)

        # Update progress bar
        current_metrics = metrics_tracker.get_average()
        pbar.set_postfix({
            'loss': f"{current_metrics['loss']:.4f}",
            'dice': f"{current_metrics['dice_mean']:.4f}"
        })

    return metrics_tracker.get_average()


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    metrics_tracker = MetricsTracker()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Calculate metrics
            metrics = batch_metrics(outputs, masks)
            metrics['loss'] = loss.item()
            metrics_tracker.update(metrics)

            # Update progress bar
            current_metrics = metrics_tracker.get_average()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'dice': f"{current_metrics['dice_mean']:.4f}"
            })

    return metrics_tracker.get_average()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'args': vars(args)
    }

    filepath = os.path.join(args.save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")


def plot_training_curves(history, save_dir):
    """
    Generate and save training curves showing loss and metrics over epochs
    
    Args:
        history: Dictionary containing training history with keys:
                'train_loss', 'val_loss', 'train_dice', 'val_dice', 'val_cdr_mae'
        save_dir: Directory to save the plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dice coefficient
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    ax2.plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Coefficient', fontsize=12)
    ax2.set_title('Dice Coefficient (Mean)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: CDR MAE
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_cdr_mae'], 'g-', label='Val CDR MAE', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('CDR MAE', fontsize=12)
    ax3.set_title('Cup-to-Disc Ratio Mean Absolute Error', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    best_val_dice = max(history['val_dice'])
    best_val_dice_epoch = history['val_dice'].index(best_val_dice) + 1
    final_val_dice = history['val_dice'][-1]
    final_val_loss = history['val_loss'][-1]
    final_cdr_mae = history['val_cdr_mae'][-1]
    
    summary_text = f"""
    Training Summary
    {'=' * 40}
    
    Total Epochs: {len(epochs)}
    
    Best Validation Dice: {best_val_dice:.4f}
    Best Dice at Epoch: {best_val_dice_epoch}
    
    Final Validation Metrics:
      - Dice: {final_val_dice:.4f}
      - Loss: {final_val_loss:.4f}
      - CDR MAE: {final_cdr_mae:.4f}
    
    Improvement:
      - Dice: {history['val_dice'][0]:.4f} → {final_val_dice:.4f}
      - Loss: {history['val_loss'][0]:.4f} → {final_val_loss:.4f}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {plot_path}")
    
    plt.close()
    
    # Also save history as JSON for later analysis
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['metrics']


def main():
    """Main training function"""
    args = get_args()

    # Setup
    device = setup_training(args)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    print(f"Creating {args.model_name} model...")
    config = CONFIGS[args.model_name]
    config.n_classes = 2  # Optic disc and cup
    config.n_skip = args.n_skip
    
    # Update config with image size
    if args.pretrained_path:
        config.pretrained_path = args.pretrained_path
    
    model = VisionTransformer(config, img_size=args.img_size, num_classes=2).to(device)
    
    # Load pretrained weights if available
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        model.load_from(np.load(args.pretrained_path))
        print("Pretrained weights loaded successfully!")

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    if num_trainable != num_params:
        print(f"Trainable parameters: {num_trainable:,}")


    # Create loss function
    criterion = CombinedSegmentationLoss(
        lambda_dice=args.lambda_dice,
        lambda_bce=args.lambda_bce
    )

    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(
            model, optimizer, scheduler, args.resume)
        start_epoch += 1

    # Training log
    log_file = os.path.join(args.save_dir, 'training_log.txt')
    
    # Initialize history tracking for plotting
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_cdr_mae': []
    }

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        scheduler.step(val_metrics['dice_mean'])
        
        # Track metrics for plotting
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice_mean'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice_mean'])
        history['val_cdr_mae'].append(val_metrics['cdr_mae'])

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice_mean']:.4f}, "
              f"CDR MAE: {train_metrics['cdr_mae']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice_mean']:.4f}, "
              f"CDR MAE: {val_metrics['cdr_mae']:.4f}")

        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_metrics['loss']:.4f},{train_metrics['dice_mean']:.4f},"
                    f"{val_metrics['loss']:.4f},{val_metrics['dice_mean']:.4f},"
                    f"{val_metrics['cdr_mae']:.4f}\n")

        # Save best model
        if val_metrics['dice_mean'] > best_dice:
            best_dice = val_metrics['dice_mean']
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_metrics, args, 'best_model.pth')
            print(f"  New best model! Dice: {best_dice:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args,
                            f'checkpoint_epoch_{epoch}.pth')

    print("\nTraining completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    
    # Generate and save training plots
    print("\nGenerating training curves...")
    plot_training_curves(history, args.save_dir)


if __name__ == '__main__':
    main()
