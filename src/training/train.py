"""
Training and evaluation functions for UNet segmentation.
"""

from data_loader.transforms import get_training_transforms, get_validation_transforms
from data_loader.dataset import get_dataloaders
from models.unet import UNet
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] - predicted logits
            target: [B, H, W] - target labels (0, 1, 2)
        """
        # Convert logits to probabilities
        pred = torch.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = torch.nn.functional.one_hot(
            target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Flatten
        pred = pred.contiguous().view(-1)
        target_one_hot = target_one_hot.contiguous().view(-1)

        # Dice coefficient
        intersection = (pred * target_one_hot).sum()
        dice = (2. * intersection + self.smooth) / \
            (pred.sum() + target_one_hot.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def calculate_iou(pred, target, num_classes=3):
    """
    Calculate IoU for each class.

    Args:
        pred: [B, C, H, W] - predicted logits
        target: [B, H, W] - target labels

    Returns:
        iou_per_class: [C] - IoU for each class
    """
    pred = torch.argmax(pred, dim=1)  # [B, H, W]

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = 0.0  # No ground truth or prediction for this class
        else:
            iou = (intersection / union).item()

        ious.append(iou)

    return ious


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = np.zeros(3)

    pbar = tqdm(train_loader, desc='Training')
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

        # Metrics
        total_loss += loss.item()
        iou = calculate_iou(outputs, masks)
        total_iou += np.array(iou)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou_bg': f'{iou[0]:.4f}',
            'iou_disc': f'{iou[1]:.4f}',
            'iou_cup': f'{iou[2]:.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)

    return avg_loss, avg_iou


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_iou = np.zeros(3)

    pbar = tqdm(val_loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Metrics
        total_loss += loss.item()
        iou = calculate_iou(outputs, masks)
        total_iou += np.array(iou)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou_bg': f'{iou[0]:.4f}',
            'iou_disc': f'{iou[1]:.4f}',
            'iou_cup': f'{iou[2]:.4f}'
        })

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    return avg_loss, avg_iou


def train_model(
    root_dir,
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    image_size=512,
    base_channels=64,
    num_workers=4,
    device=None,
    save_dir='checkpoints',
    filter_incomplete=True
):
    """
    Main training function.

    Args:
        root_dir: Project root directory
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        image_size: Input image size
        base_channels: Base channels for UNet
        num_workers: Number of data loading workers
        device: Device to train on (None = auto-detect)
        save_dir: Directory to save checkpoints
        filter_incomplete: Filter images without all 3 classes

    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_train=get_training_transforms(image_size=image_size),
        transform_val=get_validation_transforms(image_size=image_size),
        filter_incomplete=filter_incomplete
    )

    # Create model
    print("\nInitializing model...")
    model = UNet(n_channels=3, n_classes=3, base_channels=base_channels)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou_bg': [],
        'train_iou_disc': [],
        'train_iou_cup': [],
        'val_iou_bg': [],
        'val_iou_disc': [],
        'val_iou_cup': [],
        'lr': []
    }

    best_val_loss = float('inf')

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_iou = validate_epoch(
            model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou_bg'].append(train_iou[0])
        history['train_iou_disc'].append(train_iou[1])
        history['train_iou_cup'].append(train_iou[2])
        history['val_iou_bg'].append(val_iou[0])
        history['val_iou_disc'].append(val_iou[1])
        history['val_iou_cup'].append(val_iou[2])
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Train IoU - BG: {train_iou[0]:.4f}, Disc: {train_iou[1]:.4f}, Cup: {train_iou[2]:.4f}")
        print(
            f"  Val IoU   - BG: {val_iou[0]:.4f}, Disc: {val_iou[1]:.4f}, Cup: {val_iou[2]:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, history


def plot_training_history(history, save_path=None):
    """
    Plot training history.

    Args:
        history: Dictionary with training history
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-',
                    label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-',
                    label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # IoU - Background
    axes[0, 1].plot(epochs, history['train_iou_bg'],
                    'b-', label='Train IoU', linewidth=2)
    axes[0, 1].plot(epochs, history['val_iou_bg'],
                    'r-', label='Val IoU', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Background IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IoU - Optic Disc
    axes[1, 0].plot(epochs, history['train_iou_disc'],
                    'b-', label='Train IoU', linewidth=2)
    axes[1, 0].plot(epochs, history['val_iou_disc'],
                    'r-', label='Val IoU', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Optic Disc IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # IoU - Optic Cup
    axes[1, 1].plot(epochs, history['train_iou_cup'],
                    'b-', label='Train IoU', linewidth=2)
    axes[1, 1].plot(epochs, history['val_iou_cup'],
                    'r-', label='Val IoU', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('Optic Cup IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


if __name__ == '__main__':
    """Example usage"""
    root_dir = '/path/to/your/project'
    model, history = train_model(
        root_dir=root_dir,
        num_epochs=50,
        batch_size=8,
        learning_rate=1e-4
    )

    plot_training_history(history, save_path='training_history.png')
    plt.show()
