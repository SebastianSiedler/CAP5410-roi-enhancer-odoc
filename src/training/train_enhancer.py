"""
Training functions for the two-phase enhancer + UNet training.

Phase 1: Train enhancer only with frozen UNet
Phase 2: Fine-tune both enhancer and UNet jointly
"""

from models.enhancer import ImageEnhancer
from models.unet import UNet
from training.train_utils import (
    setup_training_environment,
    create_dataloaders,
    run_training_loop
)
from training.train import calculate_iou, CombinedLoss
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


class EnhancerUNetModel(nn.Module):
    """Combined Enhancer + UNet model"""

    def __init__(self, enhancer, unet):
        super().__init__()
        self.enhancer = enhancer
        self.unet = unet

    def forward(self, x):
        """
        Forward pass: image -> enhancer -> unet -> segmentation

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            enhanced: Enhanced image [B, 3, H, W]
            segmentation: Segmentation logits [B, n_classes, H, W]
        """
        enhanced = self.enhancer(x)
        segmentation = self.unet(enhanced)
        return enhanced, segmentation


def train_epoch_phase1(model, train_loader, criterion, optimizer, device, l1_weight=0.01):
    """
    Train for one epoch - Phase 1 (Enhancer only, UNet frozen)

    Args:
        model: EnhancerUNetModel with frozen UNet
        train_loader: Training data loader
        criterion: Segmentation loss (Combined CE + Dice)
        optimizer: Optimizer for enhancer only
        device: torch device
        l1_weight: Weight for L1 regularization on enhancement

    Returns:
        avg_loss: Average total loss
        avg_seg_loss: Average segmentation loss
        avg_l1_loss: Average L1 regularization loss
        avg_iou: Average IoU per class
    """
    model.train()
    model.unet.eval()  # Keep UNet in eval mode

    total_loss = 0
    total_seg_loss = 0
    total_l1_loss = 0
    total_iou = np.zeros(3)

    pbar = tqdm(train_loader, desc='Phase 1 Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        enhanced, outputs = model(images)

        # Segmentation loss
        seg_loss = criterion(outputs, masks)

        # L1 regularization: penalize large changes
        l1_loss = torch.mean(torch.abs(enhanced - images))

        # Combined loss
        loss = seg_loss + l1_weight * l1_loss

        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_l1_loss += l1_loss.item()
        iou = calculate_iou(outputs, masks)
        total_iou += np.array(iou)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'seg': f'{seg_loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'iou_cup': f'{iou[2]:.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_l1_loss = total_l1_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)

    return avg_loss, avg_seg_loss, avg_l1_loss, avg_iou


def train_epoch_phase2(model, train_loader, criterion, optimizer, device, l1_weight=0.01):
    """
    Train for one epoch - Phase 2 (Both enhancer and UNet, joint fine-tuning)
    """
    model.train()  # Both enhancer and UNet in training mode

    total_loss = 0
    total_seg_loss = 0
    total_l1_loss = 0
    total_iou = np.zeros(3)

    pbar = tqdm(train_loader, desc='Phase 2 Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        enhanced, outputs = model(images)

        seg_loss = criterion(outputs, masks)
        l1_loss = torch.mean(torch.abs(enhanced - images))
        loss = seg_loss + l1_weight * l1_loss

        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_l1_loss += l1_loss.item()
        iou = calculate_iou(outputs, masks)
        total_iou += np.array(iou)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'seg': f'{seg_loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'iou_cup': f'{iou[2]:.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_l1_loss = total_l1_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)

    return avg_loss, avg_seg_loss, avg_l1_loss, avg_iou


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device, l1_weight=0.01):
    """Validate for one epoch"""
    model.eval()

    total_loss = 0
    total_seg_loss = 0
    total_l1_loss = 0
    total_iou = np.zeros(3)

    pbar = tqdm(val_loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        enhanced, outputs = model(images)

        seg_loss = criterion(outputs, masks)
        l1_loss = torch.mean(torch.abs(enhanced - images))
        loss = seg_loss + l1_weight * l1_loss

        # Metrics
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_l1_loss += l1_loss.item()
        iou = calculate_iou(outputs, masks)
        total_iou += np.array(iou)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'seg': f'{seg_loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'iou_cup': f'{iou[2]:.4f}'
        })

    avg_loss = total_loss / len(val_loader)
    avg_seg_loss = total_seg_loss / len(val_loader)
    avg_l1_loss = total_l1_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    return avg_loss, avg_seg_loss, avg_l1_loss, avg_iou


def plot_training_history(history, save_path=None):
    """Plot training history with multiple subplots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (Seg + L1)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Segmentation loss
    axes[0, 1].plot(history['train_seg_loss'], label='Train')
    axes[0, 1].plot(history['val_seg_loss'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Segmentation Loss')
    axes[0, 1].set_title('Segmentation Loss (CE + Dice)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # L1 loss
    axes[0, 2].plot(history['train_l1_loss'], label='Train')
    axes[0, 2].plot(history['val_l1_loss'], label='Val')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('L1 Loss')
    axes[0, 2].set_title('L1 Regularization Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # IoU - Background
    axes[1, 0].plot(history['train_iou_bg'], label='Train')
    axes[1, 0].plot(history['val_iou_bg'], label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Background IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # IoU - Disc
    axes[1, 1].plot(history['train_iou_disc'], label='Train')
    axes[1, 1].plot(history['val_iou_disc'], label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].set_title('Disc IoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # IoU - Cup
    axes[1, 2].plot(history['train_iou_cup'], label='Train')
    axes[1, 2].plot(history['val_iou_cup'], label='Val')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('IoU')
    axes[1, 2].set_title('Cup IoU')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    return fig


def _train_enhancer_two_phase(
    enhancer,
    unet,
    train_loader,
    val_loader,
    device,
    num_epochs_phase1,
    num_epochs_phase2,
    learning_rate_phase1,
    learning_rate_phase2,
    l1_weight,
    save_dir,
    patience
):
    """
    Internal function to run two-phase training for any enhancer type.

    Returns:
        model: Trained combined model
        history_phase1: Phase 1 training history
        history_phase2: Phase 2 training history
    """
    # Create combined model
    model = EnhancerUNetModel(enhancer, unet)
    model = model.to(device)

    # Loss function
    class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)
    criterion = CombinedLoss(
        ce_weight=0.5, dice_weight=0.5, class_weights=class_weights, device=device)

    # ========== PHASE 1: Train Enhancer Only ==========
    print("\n" + "=" * 80)
    print("PHASE 1: Training Enhancer (UNet Frozen)")
    print("=" * 80)

    # Freeze UNet
    for param in model.unet.parameters():
        param.requires_grad = False

    optimizer_phase1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate_phase1
    )

    history_phase1, best_val_loss_phase1 = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_phase1,
        device=device,
        num_epochs=num_epochs_phase1,
        save_dir=save_dir,
        train_fn=train_epoch_phase1,
        validate_fn=validate_epoch,
        patience=patience,
        phase_based=True,
        phase_name="phase1_",
        l1_weight=l1_weight
    )

    print(f"\nPhase 1 completed! Best val loss: {best_val_loss_phase1:.4f}")

    # ========== PHASE 2: Fine-tune Both ==========
    print("\n" + "=" * 80)
    print("PHASE 2: Fine-tuning Enhancer + UNet")
    print("=" * 80)

    # Unfreeze UNet
    for param in model.unet.parameters():
        param.requires_grad = True

    optimizer_phase2 = optim.Adam(
        model.parameters(),
        lr=learning_rate_phase2
    )

    history_phase2, best_val_loss_phase2 = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_phase2,
        device=device,
        num_epochs=num_epochs_phase2,
        save_dir=save_dir,
        train_fn=train_epoch_phase2,
        validate_fn=validate_epoch,
        patience=patience,
        phase_based=True,
        phase_name="phase2_",
        l1_weight=l1_weight
    )

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Phase 2 best validation loss: {best_val_loss_phase2:.4f}")

    return model, history_phase1, history_phase2


def train_enhancer_model(
    root_dir,
    unet_checkpoint,
    num_epochs_phase1=30,
    num_epochs_phase2=20,
    batch_size=16,
    learning_rate_phase1=1e-4,
    learning_rate_phase2=1e-5,
    image_size=256,
    enhancer_base_channels=32,
    enhancer_levels=3,
    unet_base_channels=64,
    num_workers=0,
    device=None,
    save_dir='checkpoints_enhancer',
    filter_incomplete=True,
    use_clahe=False,
    l1_weight=0.001,
    patience=10
):
    """
    Train Image Enhancer + UNet model with two-phase training.

    Phase 1: Train enhancer only (UNet frozen)
    Phase 2: Fine-tune both enhancer and UNet jointly

    Args:
        root_dir: Project root directory
        unet_checkpoint: Path to pretrained UNet checkpoint
        num_epochs_phase1: Number of epochs for phase 1
        num_epochs_phase2: Number of epochs for phase 2
        batch_size: Batch size for training
        learning_rate_phase1: Learning rate for phase 1
        learning_rate_phase2: Learning rate for phase 2
        image_size: Input image size
        enhancer_base_channels: Base channels for enhancer
        enhancer_levels: Number of levels in enhancer (3 or 4)
        unet_base_channels: Base channels for UNet
        num_workers: Number of data loading workers
        device: Device to train on (None = auto-detect)
        save_dir: Directory to save checkpoints
        filter_incomplete: Filter images without all 3 classes
        use_clahe: Apply CLAHE for contrast enhancement
        l1_weight: Weight for L1 regularization
        patience: Early stopping patience

    Returns:
        model: Trained combined model
        history_phase1: Phase 1 training history
        history_phase2: Phase 2 training history
    """
    # Setup environment
    device = setup_training_environment(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        filter_incomplete=filter_incomplete,
        use_clahe=use_clahe
    )

    # Load pretrained UNet
    print(f"\nLoading pretrained UNet from: {unet_checkpoint}")
    unet = UNet(n_channels=3, n_classes=3, base_channels=unet_base_channels)
    checkpoint = torch.load(
        unet_checkpoint, map_location='cpu', weights_only=False)
    unet.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded UNet (epoch {checkpoint['epoch']})")

    # Create enhancer
    print(f"\nCreating Image Enhancer...")
    enhancer = ImageEnhancer(
        n_channels=3,
        base_channels=enhancer_base_channels,
        num_levels=enhancer_levels
    )
    print(
        f"✓ Enhancer parameters: {sum(p.numel() for p in enhancer.parameters()):,}")
    print(f"✓ UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # Run two-phase training
    return _train_enhancer_two_phase(
        enhancer=enhancer,
        unet=unet,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs_phase1=num_epochs_phase1,
        num_epochs_phase2=num_epochs_phase2,
        learning_rate_phase1=learning_rate_phase1,
        learning_rate_phase2=learning_rate_phase2,
        l1_weight=l1_weight,
        save_dir=save_dir,
        patience=patience
    )


def train_atrous_enhancer_model(
    root_dir,
    unet_checkpoint,
    num_epochs_phase1=50,
    num_epochs_phase2=30,
    batch_size=16,
    learning_rate_phase1=1e-4,
    learning_rate_phase2=1e-5,
    image_size=256,
    enhancer_type='lightweight',
    enhancer_base_channels=24,
    dilation_rates=[1, 3, 6],
    residual_weight=0.3,
    unet_base_channels=64,
    num_workers=0,
    device=None,
    save_dir='checkpoints_atrous_enhancer',
    filter_incomplete=True,
    use_clahe=False,
    l1_weight=0.001,
    patience=15
):
    """
    Train Atrous Enhancer + UNet model with two-phase training.

    Phase 1: Train atrous enhancer only (UNet frozen)
    Phase 2: Fine-tune both enhancer and UNet jointly

    Args:
        root_dir: Project root directory
        unet_checkpoint: Path to pretrained UNet checkpoint
        num_epochs_phase1: Number of epochs for phase 1
        num_epochs_phase2: Number of epochs for phase 2
        batch_size: Batch size for training
        learning_rate_phase1: Learning rate for phase 1
        learning_rate_phase2: Learning rate for phase 2
        image_size: Input image size
        enhancer_type: 'full' or 'lightweight'
        enhancer_base_channels: Base channels for enhancer
        dilation_rates: Dilation rates for ASPP
        residual_weight: Weight for residual connection
        unet_base_channels: Base channels for UNet
        num_workers: Number of data loading workers
        device: Device to train on (None = auto-detect)
        save_dir: Directory to save checkpoints
        filter_incomplete: Filter images without all 3 classes
        use_clahe: Apply CLAHE for contrast enhancement
        l1_weight: Weight for L1 regularization
        patience: Early stopping patience

    Returns:
        model: Trained combined model
        history_phase1: Phase 1 training history
        history_phase2: Phase 2 training history
    """
    from models.atrous_enhancer import AtrousImageEnhancer, LightweightAtrousEnhancer

    # Setup environment
    device = setup_training_environment(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        filter_incomplete=filter_incomplete,
        use_clahe=use_clahe
    )

    # Load pretrained UNet
    print(f"\nLoading pretrained UNet from: {unet_checkpoint}")
    unet = UNet(n_channels=3, n_classes=3, base_channels=unet_base_channels)
    checkpoint = torch.load(
        unet_checkpoint, map_location='cpu', weights_only=False)
    unet.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded UNet (epoch {checkpoint['epoch']})")

    # Create atrous enhancer
    print(f"\nCreating {enhancer_type} Atrous Enhancer...")
    if enhancer_type == 'lightweight':
        enhancer = LightweightAtrousEnhancer(
            n_channels=3,
            base_channels=enhancer_base_channels,
            residual_weight=residual_weight
        )
    else:
        enhancer = AtrousImageEnhancer(
            n_channels=3,
            base_channels=enhancer_base_channels,
            dilation_rates=dilation_rates,
            residual_weight=residual_weight
        )

    print(
        f"✓ Enhancer parameters: {sum(p.numel() for p in enhancer.parameters()):,}")
    print(f"✓ UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"✓ Dilation rates: {dilation_rates}")

    # Run two-phase training
    return _train_enhancer_two_phase(
        enhancer=enhancer,
        unet=unet,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs_phase1=num_epochs_phase1,
        num_epochs_phase2=num_epochs_phase2,
        learning_rate_phase1=learning_rate_phase1,
        learning_rate_phase2=learning_rate_phase2,
        l1_weight=l1_weight,
        save_dir=save_dir,
        patience=patience
    )
