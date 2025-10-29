"""
Task-Aware Training Module for Enhancement-Segmentation Pipeline.

This module implements the end-to-end training approach where the enhancement model (M_E)
is optimized to minimize the segmentation loss (Dice Loss) of the downstream segmentation 
model (M_S).

Key Components:
- Combined Dice Loss for segmentation quality
- Perceptual losses for image quality preservation
- Task-aware optimization strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    
    The Dice coefficient measures the overlap between predicted and ground truth masks.
    Dice Loss = 1 - Dice Coefficient
    
    Args:
        n_classes: Number of segmentation classes
        smooth: Smoothing factor to avoid division by zero
        weight: Optional class weights
    """
    
    def __init__(self, n_classes=3, smooth=1.0, weight=None):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.weight = weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions [B, n_classes, H, W]
            targets: Ground truth masks [B, H, W] with class indices
            
        Returns:
            Dice loss (scalar)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.n_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient for each class
        dice_scores = []
        for c in range(self.n_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)  # [B, n_classes]
        
        # Apply class weights if provided
        if self.weight is not None:
            dice_scores = dice_scores * self.weight.to(dice_scores.device)
        
        # Return 1 - mean Dice (loss to minimize)
        return 1.0 - dice_scores.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss to ensure enhanced images maintain visual quality.
    Uses L1 distance between enhanced and original images.
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, enhanced, original):
        """
        Args:
            enhanced: Enhanced images [B, C, H, W]
            original: Original images [B, C, H, W]
            
        Returns:
            L1 distance (scalar)
        """
        return self.criterion(enhanced, original)


class TaskAwareLoss(nn.Module):
    """
    Combined loss function for task-aware enhancement training.
    
    L_total = λ_dice * L_dice + λ_perceptual * L_perceptual
    
    where:
    - L_dice: Dice loss on segmentation predictions (task-aware component)
    - L_perceptual: Perceptual loss to maintain image quality
    
    Args:
        lambda_dice: Weight for segmentation (task-aware) loss
        lambda_perceptual: Weight for perceptual (quality) loss
        n_classes: Number of segmentation classes
        class_weights: Optional weights for each class in Dice loss
    """
    
    def __init__(
        self,
        lambda_dice=1.0,
        lambda_perceptual=0.1,
        n_classes=3,
        class_weights=None
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_perceptual = lambda_perceptual
        
        self.dice_loss = DiceLoss(n_classes=n_classes, weight=class_weights)
        self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self, 
        segmentation_logits,
        targets,
        enhanced_images,
        original_images
    ):
        """
        Compute combined task-aware loss.
        
        Args:
            segmentation_logits: Predicted segmentation [B, n_classes, H, W]
            targets: Ground truth masks [B, H, W]
            enhanced_images: Enhanced images [B, C, H, W]
            original_images: Original images [B, C, H, W]
            
        Returns:
            total_loss: Combined loss (scalar)
            loss_dict: Dictionary with individual loss components
        """
        # Task-aware loss: optimize for segmentation performance
        dice_loss = self.dice_loss(segmentation_logits, targets)
        
        # Perceptual loss: maintain image quality
        perceptual_loss = self.perceptual_loss(enhanced_images, original_images)
        
        # Combine losses
        total_loss = (
            self.lambda_dice * dice_loss +
            self.lambda_perceptual * perceptual_loss
        )
        
        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'dice': dice_loss.item(),
            'perceptual': perceptual_loss.item(),
        }
        
        return total_loss, loss_dict


def compute_dice_score(logits, targets, n_classes=3):
    """
    Compute Dice score (metric) for evaluation.
    
    Args:
        logits: Model predictions [B, n_classes, H, W]
        targets: Ground truth masks [B, H, W]
        n_classes: Number of classes
        
    Returns:
        dice_scores: Dictionary with per-class and mean Dice scores
    """
    with torch.no_grad():
        # Convert logits to class predictions
        preds = torch.argmax(logits, dim=1)
        
        dice_scores = {}
        valid_classes = []
        
        for c in range(n_classes):
            pred_c = (preds == c).float()
            target_c = (targets == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union > 0:
                dice = (2. * intersection) / (union + 1e-8)
                dice_scores[f'dice_class_{c}'] = dice.item()
                valid_classes.append(dice.item())
            else:
                dice_scores[f'dice_class_{c}'] = 0.0
        
        # Mean Dice across all classes (including background)
        dice_scores['dice_mean'] = sum(dice_scores.values()) / n_classes
        
        # Mean Dice excluding background (OD and OC only) - more clinically relevant
        if n_classes >= 3:
            dice_scores['dice_od'] = dice_scores['dice_class_1']  # Optic Disc
            dice_scores['dice_oc'] = dice_scores['dice_class_2']  # Optic Cup
            dice_scores['dice_od_oc_mean'] = (dice_scores['dice_od'] + dice_scores['dice_oc']) / 2
    
    return dice_scores


class EnhancementTrainer:
    """
    Trainer class for task-aware enhancement model training.
    
    This handles the training loop for the enhancement-segmentation pipeline,
    implementing the task-aware optimization strategy.
    
    Args:
        pipeline: EnhancementSegmentationPipeline model
        optimizer: PyTorch optimizer
        criterion: TaskAwareLoss instance
        device: Device to train on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
    """
    
    def __init__(
        self,
        pipeline,
        optimizer,
        criterion,
        device='cuda',
        scheduler=None
    ):
        self.pipeline = pipeline.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
    
    def train_step(self, images, masks):
        """
        Single training step.
        
        Args:
            images: Input images [B, C, H, W]
            masks: Ground truth segmentation masks [B, H, W]
            
        Returns:
            loss_dict: Dictionary with loss values
            metrics_dict: Dictionary with metrics (Dice scores)
        """
        self.pipeline.train()
        
        # Move data to device
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        # Forward pass through enhancement + segmentation
        enhanced_images, segmentation_logits = self.pipeline(images)
        
        # Compute task-aware loss
        loss, loss_dict = self.criterion(
            segmentation_logits=segmentation_logits,
            targets=masks,
            enhanced_images=enhanced_images,
            original_images=images
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        metrics_dict = compute_dice_score(segmentation_logits, masks)
        
        return loss_dict, metrics_dict
    
    def validate_step(self, images, masks):
        """
        Single validation step.
        
        Args:
            images: Input images [B, C, H, W]
            masks: Ground truth segmentation masks [B, H, W]
            
        Returns:
            loss_dict: Dictionary with loss values
            metrics_dict: Dictionary with metrics (Dice scores)
        """
        self.pipeline.eval()
        
        with torch.no_grad():
            # Move data to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            enhanced_images, segmentation_logits = self.pipeline(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                segmentation_logits=segmentation_logits,
                targets=masks,
                enhanced_images=enhanced_images,
                original_images=images
            )
            
            # Compute metrics
            metrics_dict = compute_dice_score(segmentation_logits, masks)
        
        return loss_dict, metrics_dict
    
    def update_scheduler(self, metric=None):
        """Update learning rate scheduler."""
        if self.scheduler is not None:
            if metric is not None and hasattr(self.scheduler, 'step'):
                # For ReduceLROnPlateau
                self.scheduler.step(metric)
            else:
                self.scheduler.step()


def train_enhancement_model(
    root_dir,
    segmentation_model_path,
    num_epochs=50,
    batch_size=1,
    learning_rate=1e-4,
    image_size=256,
    enhancement_channels=24,
    n_residual_blocks=3,
    num_workers=0,
    device=None,
    save_dir='enhancement_checkpoints',
    lambda_dice=1.0,
    lambda_perceptual=0.1,
    patience=10,
    save_interval=5,
    use_tensorboard=True,
    filter_incomplete=False
):
    """
    Main training function for enhancement model with frozen segmentation model.
    
    Args:
        root_dir: Project root directory
        segmentation_model_path: Path to pre-trained segmentation model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        image_size: Input image size
        enhancement_channels: Base channels for enhancement model
        n_residual_blocks: Number of residual blocks in enhancement model
        num_workers: Number of data loading workers
        device: Device to train on (None = auto-detect)
        save_dir: Directory to save checkpoints
        lambda_dice: Weight for dice loss
        lambda_perceptual: Weight for perceptual loss
        patience: Early stopping patience
        save_interval: Save checkpoint every N epochs
        use_tensorboard: Whether to use TensorBoard logging
        filter_incomplete: If True, filter out images without all 3 classes
        
    Returns:
        pipeline: Trained enhancement-segmentation pipeline
        history: Dictionary with training history
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path
    import numpy as np
    from tqdm.auto import tqdm
    from datetime import datetime
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import matplotlib.pyplot as plt
    
    from models.unet import UNet
    from models.enhancement_unet import EnhancementUNet, EnhancementSegmentationPipeline
    from data_loader.dataset import GlaucomaDataset
odel    
    # Clear dataset cache to ensure filter_incomplete works correctly
    GlaucomaDataset._split_cache.clear()
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True, parents=True)
    (save_dir / 'visualizations').mkdir(exist_ok=True, parents=True)
    
    # Setup TensorBoard
    writer = None
    if use_tensorboard:
        log_dir = save_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(str(log_dir))
        print(f"TensorBoard logging to: {log_dir}")
    
    # Define transforms
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Create datasets
    print("\nLoading datasets...")
    # GlaucomaDataset expects project root and constructs the path internally
    train_dataset = GlaucomaDataset(root_dir, split='train', transform=train_transform, filter_incomplete=filter_incomplete)
    val_dataset = GlaucomaDataset(root_dir, split='val', transform=val_transform, filter_incomplete=filter_incomplete)
    
    print(f"Train samples: {len(train_dataset)} (filter_incomplete={filter_incomplete})")
    print(f"Val samples: {len(val_dataset)} (filter_incomplete={filter_incomplete})")
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty after filtering! Try setting filter_incomplete=False")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty after filtering! Try setting filter_incomplete=False")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Load pre-trained segmentation model
    print(f"\nLoading pre-trained segmentation model from: {segmentation_model_path}")
    segmentation_model = UNet(n_channels=3, n_classes=3, base_channels=64)
    checkpoint = torch.load(segmentation_model_path, map_location=device, weights_only=False)
    segmentation_model.load_state_dict(checkpoint['model_state_dict'])
    segmentation_model = segmentation_model.to(device)
    
    # Freeze segmentation model
    for param in segmentation_model.parameters():
        param.requires_grad = False
    segmentation_model.eval()
    print("✓ Segmentation model frozen")
    
    # Create enhancement model
    print("\nInitializing enhancement model...")
    enhancement_model = EnhancementUNet(
        n_channels=3,
        n_output_channels=3,
        base_channels=enhancement_channels,
        n_residual_blocks=n_residual_blocks
    )
    enhancement_model = enhancement_model.to(device)
    print(f"Enhancement model parameters: {sum(p.numel() for p in enhancement_model.parameters()):,}")
    
    # Create pipeline
    pipeline = EnhancementSegmentationPipeline(enhancement_model, segmentation_model)
    
    # Loss and optimizer
    criterion = TaskAwareLoss(lambda_dice=lambda_dice, lambda_perceptual=lambda_perceptual)
    optimizer = optim.Adam(enhancement_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Create trainer with correct parameter order
    trainer = EnhancementTrainer(pipeline, optimizer, criterion, device, scheduler)
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }
    
    best_dice = 0.0
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)
        
        # Train
        pipeline.train()
        epoch_losses = []
        epoch_dice = []
        
        pbar = tqdm(train_loader, desc=f'[Train]')
        for batch_idx, (images, masks) in enumerate(pbar):
            loss_dict, metrics_dict = trainer.train_step(images, masks)
            
            epoch_losses.append(loss_dict['total'])
            epoch_dice.append(metrics_dict['dice_mean'])
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'dice': f"{metrics_dict['dice_mean']:.4f}"
            })
            
            # Log to tensorboard
            if writer is not None:
                step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('train/loss_total', loss_dict['total'], step)
                writer.add_scalar('train/loss_dice', loss_dict['dice'], step)
                writer.add_scalar('train/loss_perceptual', loss_dict['perceptual'], step)
                writer.add_scalar('train/dice_mean', metrics_dict['dice_mean'], step)
            
            # Clear GPU cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        train_loss = np.mean(epoch_losses)
        train_dice = np.mean(epoch_dice)
        
        # Validate
        pipeline.eval()
        epoch_losses = []
        epoch_dice = []
        
        pbar = tqdm(val_loader, desc=f'[Val]  ')
        for batch_idx, (images, masks) in enumerate(pbar):
            loss_dict, metrics_dict = trainer.validate_step(images, masks)
            
            epoch_losses.append(loss_dict['total'])
            epoch_dice.append(metrics_dict['dice_mean'])
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'dice': f"{metrics_dict['dice_mean']:.4f}"
            })
            
            # Visualize first batch
            if batch_idx == 0:
                with torch.no_grad():
                    images_vis = images[:1].to(device)
                    masks_vis = masks[:1].to(device)
                    enhanced, logits = pipeline(images_vis)
                    pred = torch.argmax(logits, dim=1)
                    
                    # Save visualization
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    
                    # Denormalize for visualization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    
                    # Original
                    img = images_vis[0].cpu().permute(1, 2, 0).numpy()
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                    axes[0].imshow(img)
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    
                    # Enhanced
                    enh = enhanced[0].cpu().permute(1, 2, 0).numpy()
                    enh = enh * std + mean
                    enh = np.clip(enh, 0, 1)
                    axes[1].imshow(enh)
                    axes[1].set_title('Enhanced')
                    axes[1].axis('off')
                    
                    # Ground Truth
                    axes[2].imshow(masks_vis[0].cpu().numpy(), cmap='jet', vmin=0, vmax=2)
                    axes[2].set_title('Ground Truth')
                    axes[2].axis('off')
                    
                    # Prediction
                    axes[3].imshow(pred[0].cpu().numpy(), cmap='jet', vmin=0, vmax=2)
                    axes[3].set_title('Prediction')
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(save_dir / 'visualizations' / f'epoch_{epoch:03d}.png', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    del images_vis, masks_vis, enhanced, logits, pred
                    torch.cuda.empty_cache()
        
        val_loss = np.mean(epoch_losses)
        val_dice = np.mean(epoch_dice)
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/dice_mean', val_dice, epoch)
        
        # Update learning rate
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'enhancement_state_dict': enhancement_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'history': history
            }
            torch.save(checkpoint, save_dir / 'checkpoints' / 'best_model.pth')
            print(f"  ✓ Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'enhancement_state_dict': enhancement_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'history': history
            }
            torch.save(checkpoint, save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch:03d}.pth')
            print(f"  ✓ Saved checkpoint")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"{'='*80}")
            break
        
        # Log learning rate to tensorboard
        if writer is not None:
            writer.add_scalar('learning_rate', current_lr, epoch)
    
    if writer is not None:
        writer.close()
    
    print(f"\n{'='*80}")
    print("Training Completed!")
    print(f"{'='*80}")
    print(f"Best Validation Dice Score: {best_dice:.4f}")
    print(f"Models saved in: {save_dir / 'checkpoints'}")
    print(f"Visualizations saved in: {save_dir / 'visualizations'}")
    if use_tensorboard:
        print(f"TensorBoard logs: {log_dir}")
        print(f"\nTo view TensorBoard: tensorboard --logdir {log_dir}")
    print(f"{'='*80}")
    
    return pipeline, history


if __name__ == '__main__':
    """Test the loss functions and trainer"""
    print("=" * 80)
    print("Testing Task-Aware Loss Functions")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    n_classes = 3
    height, width = 256, 256
    
    # Dummy predictions and targets
    logits = torch.randn(batch_size, n_classes, height, width)
    targets = torch.randint(0, n_classes, (batch_size, height, width))
    enhanced = torch.randn(batch_size, 3, height, width)
    original = torch.randn(batch_size, 3, height, width)
    
    # Test Dice Loss
    print("\nDice Loss:")
    dice_criterion = DiceLoss(n_classes=n_classes)
    dice_loss = dice_criterion(logits, targets)
    print(f"  Loss value: {dice_loss.item():.4f}")
    
    # Test Perceptual Loss
    print("\nPerceptual Loss:")
    perceptual_criterion = PerceptualLoss()
    perceptual_loss = perceptual_criterion(enhanced, original)
    print(f"  Loss value: {perceptual_loss.item():.4f}")
    
    # Test Combined Loss
    print("\nTask-Aware Combined Loss:")
    combined_criterion = TaskAwareLoss(lambda_dice=1.0, lambda_perceptual=0.1)
    total_loss, loss_dict = combined_criterion(logits, targets, enhanced, original)
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Dice loss: {loss_dict['dice']:.4f}")
    print(f"  Perceptual loss: {loss_dict['perceptual']:.4f}")
    
    # Test Dice Score Computation
    print("\nDice Score Metrics:")
    dice_scores = compute_dice_score(logits, targets, n_classes=n_classes)
    for key, value in dice_scores.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 80)
