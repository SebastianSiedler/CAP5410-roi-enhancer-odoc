"""
Main training script for baseline U-Net segmentation
Trains U-Net on REFUGE dataset for optic disc/cup segmentation
"""
from utils.metrics import batch_metrics, MetricsTracker
from utils.loss_functions import CombinedSegmentationLoss
from models.unet import UNet
from data_loader.dataset import RetinaDataset, RetinaDatasetValidation
import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train baseline U-Net for OD/OC segmentation')

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
    parser.add_argument('--base_features', type=int, default=64,
                        help='Base number of features in U-Net')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transposed conv')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--target_size', type=int, default=512,
                        help='Target image size (square)')

    # Loss parameters
    parser.add_argument('--lambda_dice', type=float, default=0.5,
                        help='Weight for Dice loss')
    parser.add_argument('--lambda_bce', type=float, default=0.5,
                        help='Weight for BCE loss')

    # Checkpoint parameters
    parser.add_argument('--save_dir', type=str,
                        default='experiments/baseline_unet',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')

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

    # Training dataset
    train_dataset = RetinaDataset(
        root_dir=args.data_dir,
        csv_file=args.train_csv,
        target_size=(args.target_size, args.target_size),
        use_cropped=True
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
        target_size=(args.target_size, args.target_size),
        use_cropped=True
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
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice_mean']:.4f}"
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
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice_mean']:.4f}"
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
    print("Creating model...")
    model = UNet(
        n_channels=3,
        n_classes=2,
        bilinear=args.bilinear,
        base_features=args.base_features
    ).to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

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


if __name__ == '__main__':
    main()
