"""
Training functions for ASPP-UNet segmentation.
"""

from training.train import train_epoch, validate_epoch, CombinedLoss, plot_training_history
from data_loader.transforms import get_training_transforms, get_validation_transforms
from data_loader.dataset import get_dataloaders
from models.aspp_unet import ASPPUNet, LightweightASPPUNet
import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def train_aspp_unet(
    root_dir,
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    image_size=256,
    base_channels=64,
    num_workers=4,
    device=None,
    save_dir='checkpoints_aspp_unet',
    filter_incomplete=True,
    use_clahe=False,
    model_type='full',
    dilation_rates=[1, 6, 12, 18],
    patience=15
):
    """
    Train ASPP-UNet model for optic disc/cup segmentation.

    Args:
        root_dir: Project root directory
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        image_size: Input image size
        base_channels: Base channels for ASPP-UNet
        num_workers: Number of data loading workers
        device: Device to train on (None = auto-detect)
        save_dir: Directory to save checkpoints
        filter_incomplete: Filter images without all 3 classes
        use_clahe: Apply CLAHE for contrast enhancement
        model_type: 'full' or 'lightweight'
        dilation_rates: Dilation rates for ASPP module
        patience: Early stopping patience

    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seeds for reproducible results
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    print("Random seeds set for reproducible training")

    if use_clahe:
        print("Using CLAHE for contrast enhancement")

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

    # Create model
    print(f"\nInitializing {model_type} ASPP-UNet model...")
    if model_type == 'lightweight':
        model = LightweightASPPUNet(
            n_channels=3,
            n_classes=3,
            base_channels=base_channels,
            # Use fewer rates for lightweight
            dilation_rates=dilation_rates[:3]
        )
    else:
        model = ASPPUNet(
            n_channels=3,
            n_classes=3,
            base_channels=base_channels,
            dilation_rates=dilation_rates
        )

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dilation rates: {dilation_rates}")

    # Loss and optimizer
    class_weights = torch.tensor([
        1.0,  # Background
        1.0,  # Disc
        2.0   # Cup (Cup is harder to segment -> higher weight)
    ], device=device)
    criterion = CombinedLoss(
        ce_weight=0.5, dice_weight=0.5, class_weights=class_weights, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    }

    best_val_loss = float('inf')
    patience_counter = 0

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

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou_bg'].append(train_iou[0])
        history['train_iou_disc'].append(train_iou[1])
        history['train_iou_cup'].append(train_iou[2])
        history['val_iou_bg'].append(val_iou[0])
        history['val_iou_disc'].append(val_iou[1])
        history['val_iou_cup'].append(val_iou[2])

        # Print epoch summary
        mean_train_iou = np.mean(train_iou)
        mean_val_iou = np.mean(val_iou)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}")
        print(
            f"  Train - IoU: BG={train_iou[0]:.4f}, Disc={train_iou[1]:.4f}, Cup={train_iou[2]:.4f}, Mean={mean_train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}")
        print(
            f"  Val   - IoU: BG={val_iou[0]:.4f}, Disc={val_iou[1]:.4f}, Cup={val_iou[2]:.4f}, Mean={mean_val_iou:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{patience}")

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

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            print(f"  No improvement for {patience} epochs")
            break

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, history


if __name__ == '__main__':
    """Example usage"""
    root_dir = '/path/to/your/project'
    model, history = train_aspp_unet(
        root_dir=root_dir,
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        model_type='full',
        dilation_rates=[1, 6, 12, 18]
    )

    from training.train import plot_training_history
    plot_training_history(history, save_path='aspp_unet_training_history.png')
    plt.show()
