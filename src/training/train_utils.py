"""
Common training utilities and helper functions.
"""

import torch
import numpy as np
from pathlib import Path
from data_loader.dataset import get_dataloaders, GlaucomaDataset
from data_loader.transforms import get_training_transforms, get_validation_transforms


def setup_training_environment(device=None, seed=42):
    """
    Setup device and random seeds for reproducible training.

    Args:
        device: Device to use (None = auto-detect)
        seed: Random seed for reproducibility

    Returns:
        device: Configured device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Set seeds for reproducible results
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    print(f"Random seeds set to {seed} for reproducible training")

    return device


def create_dataloaders(
    root_dir,
    batch_size,
    image_size,
    num_workers=0,
    filter_incomplete=True,
    use_clahe=False
):
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Project root directory
        batch_size: Batch size for training
        image_size: Input image size
        num_workers: Number of data loading workers
        filter_incomplete: Filter images without all 3 classes
        use_clahe: Apply CLAHE for contrast enhancement

    Returns:
        train_loader, val_loader, test_loader
    """
    # Clear dataset cache
    GlaucomaDataset._split_cache.clear()

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

    if use_clahe:
        print("Using CLAHE for contrast enhancement")

    return train_loader, val_loader, test_loader


def save_checkpoint(
    save_dir,
    filename,
    epoch,
    model,
    optimizer,
    val_loss,
    val_iou,
    best_val_loss=None,
    **extra_state
):
    """
    Save model checkpoint.

    Args:
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer to save
        val_loss: Validation loss
        val_iou: Validation IoU
        best_val_loss: Best validation loss (optional)
        **extra_state: Additional state to save
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_path = save_dir / filename

    state_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_iou': val_iou,
    }

    if best_val_loss is not None:
        state_dict['best_val_loss'] = best_val_loss

    # Add any extra state
    state_dict.update(extra_state)

    torch.save(state_dict, checkpoint_path)


def initialize_history(phase_based=False):
    """
    Initialize training history dictionary.

    Args:
        phase_based: Whether to include phase-specific metrics (seg_loss, l1_loss)

    Returns:
        history: Dictionary for tracking training metrics
    """
    history = {
        'train_loss': [],
        'train_iou_bg': [],
        'train_iou_disc': [],
        'train_iou_cup': [],
        'val_loss': [],
        'val_iou_bg': [],
        'val_iou_disc': [],
        'val_iou_cup': [],
    }

    if phase_based:
        history.update({
            'train_seg_loss': [],
            'train_l1_loss': [],
            'val_seg_loss': [],
            'val_l1_loss': [],
        })

    return history


def update_history(history, train_metrics, val_metrics, phase_based=False):
    """
    Update training history with new metrics.

    Args:
        history: History dictionary to update
        train_metrics: Training metrics (loss, iou) or (loss, seg_loss, l1_loss, iou)
        val_metrics: Validation metrics (loss, iou) or (loss, seg_loss, l1_loss, iou)
        phase_based: Whether metrics include phase-specific losses
    """
    if phase_based:
        train_loss, train_seg, train_l1, train_iou = train_metrics
        val_loss, val_seg, val_l1, val_iou = val_metrics

        history['train_loss'].append(train_loss)
        history['train_seg_loss'].append(train_seg)
        history['train_l1_loss'].append(train_l1)
        history['val_loss'].append(val_loss)
        history['val_seg_loss'].append(val_seg)
        history['val_l1_loss'].append(val_l1)
    else:
        train_loss, train_iou = train_metrics
        val_loss, val_iou = val_metrics

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    # IoU metrics are the same for both
    history['train_iou_bg'].append(train_iou[0])
    history['train_iou_disc'].append(train_iou[1])
    history['train_iou_cup'].append(train_iou[2])
    history['val_iou_bg'].append(val_iou[0])
    history['val_iou_disc'].append(val_iou[1])
    history['val_iou_cup'].append(val_iou[2])


def print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, phase_based=False):
    """
    Print epoch summary.

    Args:
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        train_metrics: Training metrics
        val_metrics: Validation metrics
        phase_based: Whether to print phase-specific metrics
    """
    print(f"\nEpoch {epoch + 1} Summary:")

    if phase_based:
        train_loss, train_seg, train_l1, train_iou = train_metrics
        val_loss, val_seg, val_l1, val_iou = val_metrics

        print(
            f"  Train - Loss: {train_loss:.4f} (Seg: {train_seg:.4f}, L1: {train_l1:.4f})")
        print(
            f"  Train - IoU: BG={train_iou[0]:.4f}, Disc={train_iou[1]:.4f}, Cup={train_iou[2]:.4f}")
        print(
            f"  Val   - Loss: {val_loss:.4f} (Seg: {val_seg:.4f}, L1: {val_l1:.4f})")
        print(
            f"  Val   - IoU: BG={val_iou[0]:.4f}, Disc={val_iou[1]:.4f}, Cup={val_iou[2]:.4f}")
    else:
        train_loss, train_iou = train_metrics
        val_loss, val_iou = val_metrics

        print(f"  Train - Loss: {train_loss:.4f}")
        print(
            f"  Train - IoU: BG={train_iou[0]:.4f}, Disc={train_iou[1]:.4f}, Cup={train_iou[2]:.4f}, Mean={np.mean(train_iou):.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}")
        print(
            f"  Val   - IoU: BG={val_iou[0]:.4f}, Disc={val_iou[1]:.4f}, Cup={val_iou[2]:.4f}, Mean={np.mean(val_iou):.4f}")


def run_training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_dir,
    train_fn,
    validate_fn,
    patience=None,
    phase_based=False,
    phase_name="",
    **train_kwargs
):
    """
    Generic training loop that can be reused across different models.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs
        save_dir: Directory to save checkpoints
        train_fn: Training function (train_epoch or train_epoch_phase1/2)
        validate_fn: Validation function
        patience: Early stopping patience (None = no early stopping)
        phase_based: Whether training uses phase-specific metrics
        phase_name: Name prefix for saved checkpoints (e.g., "phase1_")
        **train_kwargs: Additional kwargs for train_fn (e.g., l1_weight)

    Returns:
        history: Training history dictionary
        best_val_loss: Best validation loss achieved
    """
    history = initialize_history(phase_based=phase_based)
    best_val_loss = float('inf')
    patience_counter = 0
    save_dir = Path(save_dir)

    for epoch in range(num_epochs):
        print(f"\n{phase_name}Epoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_metrics = train_fn(
            model, train_loader, criterion, optimizer, device, **train_kwargs
        )

        # Validate
        val_metrics = validate_fn(
            model, val_loader, criterion, device, **train_kwargs
        )

        # Extract validation loss
        val_loss = val_metrics[0]
        val_iou = val_metrics[-1]

        # Update history
        update_history(history, train_metrics, val_metrics,
                       phase_based=phase_based)

        # Print summary
        print_epoch_summary(epoch, num_epochs, train_metrics,
                            val_metrics, phase_based=phase_based)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_checkpoint(
                save_dir=save_dir,
                filename=f'{phase_name}best_model.pth',
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                val_iou=val_iou,
                best_val_loss=best_val_loss
            )
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience is not None:
                print(f"  Patience: {patience_counter}/{patience}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                save_dir=save_dir,
                filename=f'{phase_name}checkpoint_epoch_{epoch + 1}.pth',
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                val_iou=val_iou
            )
            print(f"  ✓ Saved checkpoint")

        # Early stopping
        if patience is not None and patience_counter >= patience:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            print(f"  No improvement for {patience} epochs")
            break

    return history, best_val_loss
