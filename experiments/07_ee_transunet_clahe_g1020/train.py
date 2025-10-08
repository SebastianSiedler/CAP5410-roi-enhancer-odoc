"""
Training script for Experiment 07: EE-TransUNet + CLAHE on G1020 Dataset
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.models.ee_transunet import VisionTransformer
from src.models.configs import get_tiny_config
from src.data_loader.dataset import RetinaDataset, RetinaDatasetTest
from src.utils.metrics import batch_metrics


def get_args():
    parser = argparse.ArgumentParser(description='Train EE-TransUNet with CLAHE on G1020')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='ViT-Tiny',
                       help='Model variant')
    parser.add_argument('--img_size', type=int, default=430,
                       help='Image size (G1020 native size)')
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of output classes')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../../datasets/G1020',
                       help='Path to G1020 dataset')
    parser.add_argument('--train_csv', type=str, default='../../datasets/G1020/train.csv',
                       help='Path to train CSV')
    parser.add_argument('--val_csv', type=str, default='../../datasets/G1020/val.csv',
                       help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, default='../../datasets/G1020/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--use_clahe', action='store_true', default=True,
                       help='Enable CLAHE preprocessing')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (reduced for 430x430 images)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            metrics = batch_metrics(outputs, masks)
            epoch_loss += loss.item()
            epoch_dice += metrics['dice_mean']
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{metrics['dice_mean']:.4f}"})
    
    return epoch_loss / len(loader), epoch_dice / len(loader)


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0
    epoch_dice = 0
    epoch_dice_disc = 0
    epoch_dice_cup = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            metrics = batch_metrics(outputs, masks)
            epoch_loss += loss.item()
            epoch_dice += metrics['dice_mean']
            epoch_dice_disc += metrics['dice_disc']
            epoch_dice_cup += metrics['dice_cup']
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{metrics['dice_mean']:.4f}"})
    
    return {
        'loss': epoch_loss / len(loader),
        'dice_mean': epoch_dice / len(loader),
        'dice_disc': epoch_dice_disc / len(loader),
        'dice_cup': epoch_dice_cup / len(loader),
    }


def main():
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/visualizations", exist_ok=True)
    
    print("="*60)
    print("EE-TransUNet + CLAHE Training on G1020 Dataset")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Image size: {args.img_size}")
    print(f"CLAHE: {args.use_clahe}")
    print(f"Device: {args.device}")
    print(f"Dataset: G1020")
    print("="*60)
    
    # Save configuration
    config_dict = vars(args)
    with open(f"{args.output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)
    print("✅ Configuration saved")
    
    # Load datasets
    print("\nLoading G1020 datasets...")
    train_dataset = RetinaDataset(
        csv_file=args.train_csv,
        root_dir=args.data_dir,
        use_cropped=True,
        use_clahe=args.use_clahe,
    )
    
    val_dataset = RetinaDataset(
        csv_file=args.val_csv,
        root_dir=args.data_dir,
        use_cropped=True,
        use_clahe=args.use_clahe,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    config_vit = get_tiny_config()
    config_vit.n_classes = args.n_classes
    config_vit.n_skip = 0
    
    model = VisionTransformer(config_vit, img_size=args.img_size, num_classes=args.n_classes)
    model = model.to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created ({total_params:,} parameters)")
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    
    print(f"✅ Optimizer: Adam (LR={args.learning_rate})")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [],
               'val_dice_disc': [], 'val_dice_cup': []}
    best_dice = 0.0
    epochs_without_improvement = 0
    start_time = datetime.now()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_metrics = validate_epoch(model, val_loader, criterion, args.device)
        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice_mean'])
        history['val_dice_disc'].append(val_metrics['dice_disc'])
        history['val_dice_cup'].append(val_metrics['dice_cup'])
        
        overfitting_gap = train_dice - val_metrics['dice_mean']
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Dice:   {val_metrics['dice_mean']:.4f}")
        print(f"  Val Disc:   {val_metrics['dice_disc']:.4f} | Val Cup:    {val_metrics['dice_cup']:.4f}")
        print(f"  Overfitting Gap: {overfitting_gap:.4f}")
        
        if val_metrics['dice_mean'] > best_dice:
            best_dice = val_metrics['dice_mean']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': config_dict,
            }, f"{args.output_dir}/best_model.pth")
            print(f"  ✅ Best model saved (Dice: {best_dice:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  📊 No improvement ({epochs_without_improvement}/{args.early_stop_patience})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': config_dict,
            }, f"{args.output_dir}/checkpoint_epoch_{epoch+1}.pth")
            print(f"  💾 Checkpoint saved (epoch {epoch+1})")
        
        # Early stopping
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n⏹️  Early stopping triggered")
            break
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 3600
    
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time:.2f} hours")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"{'='*60}")
    
    # Save training history
    with open(f"{args.output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
    axes[1].plot(history['val_dice_disc'], label='Val Disc', linewidth=2, linestyle='--')
    axes[1].plot(history['val_dice_cup'], label='Val Cup', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Training curves saved to {args.output_dir}/training_curves.png")
    print(f"✅ Training history saved to {args.output_dir}/training_history.json")
    print(f"\n🎯 Best model available at: {args.output_dir}/best_model.pth")


if __name__ == '__main__':
    main()
