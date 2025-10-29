"""
Visualization utilities for enhancement model training and evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional


def denormalize_image(img: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Denormalize an image using ImageNet statistics.
    
    Args:
        img: Image array in [H, W, C] format
        mean: Mean values for denormalization (default: ImageNet mean)
        std: Std values for denormalization (default: ImageNet std)
    
    Returns:
        Denormalized image clipped to [0, 1]
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    
    # Check if image needs denormalization
    if img.min() < 0:  # Likely normalized with ImageNet stats
        img = std * img + mean
        img = np.clip(img, 0, 1)
    else:
        # Just normalize to [0, 1] range
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    return img


def visualize_sample(image: torch.Tensor, mask: torch.Tensor, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Visualize a single image-mask pair.
    
    Args:
        image: Image tensor [C, H, W]
        mask: Mask tensor [H, W]
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Image - denormalize for display
    img = image.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
    img = denormalize_image(img)
    
    axes[0].imshow(img)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Mask (0: background, 1: OD, 2: OC)
    mask_np = mask.numpy()
    im = axes[1].imshow(mask_np, cmap='jet', vmin=0, vmax=2)
    bg_count = (mask_np == 0).sum()
    od_count = (mask_np == 1).sum()
    oc_count = (mask_np == 2).sum()
    axes[1].set_title(f'Mask (0:BG, 1:OD, 2:OC)\nBG:{bg_count}, OD:{od_count}, OC:{oc_count} px')
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_segmentation_results(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize segmentation results: original image, ground truth, and prediction.
    
    Args:
        image: Image tensor [C, H, W]
        ground_truth: Ground truth mask [H, W]
        prediction: Predicted mask [H, W]
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image - denormalize for display
    img = image.cpu().permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
    img = denormalize_image(img)
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground Truth
    gt = ground_truth.cpu().numpy()
    im1 = axes[1].imshow(gt, cmap='jet', vmin=0, vmax=2)
    bg_gt = (gt == 0).sum()
    od_gt = (gt == 1).sum()
    oc_gt = (gt == 2).sum()
    axes[1].set_title(f'Ground Truth\nBG:{bg_gt}, OD:{od_gt}, OC:{oc_gt} px')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Prediction
    pred = prediction.cpu().numpy()
    im2 = axes[2].imshow(pred, cmap='jet', vmin=0, vmax=2)
    bg_pred = (pred == 0).sum()
    od_pred = (pred == 1).sum()
    oc_pred = (pred == 2).sum()
    axes[2].set_title(f'Prediction\nBG:{bg_pred}, OD:{od_pred}, OC:{oc_pred} px')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_enhancement_pipeline(
    original: torch.Tensor,
    enhanced: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    save_path: Optional[Path] = None,
    num_samples: int = 1
) -> plt.Figure:
    """
    Visualize the full enhancement-segmentation pipeline results.
    
    Args:
        original: Original images [B, C, H, W]
        enhanced: Enhanced images [B, C, H, W]
        ground_truth: Ground truth masks [B, H, W]
        prediction: Predicted masks [B, H, W]
        save_path: Optional path to save the figure
        num_samples: Number of samples to visualize
    
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, len(original))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = original[i].cpu().permute(1, 2, 0).numpy()
        img = denormalize_image(img)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Enhanced image
        enh = enhanced[i].cpu().permute(1, 2, 0).numpy()
        enh = denormalize_image(enh)
        axes[i, 1].imshow(enh)
        axes[i, 1].set_title('Enhanced')
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(ground_truth[i].cpu().numpy(), cmap='jet', vmin=0, vmax=2)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Prediction
        axes[i, 3].imshow(prediction[i].cpu().numpy(), cmap='jet', vmin=0, vmax=2)
        axes[i, 3].set_title('Prediction')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(history: dict, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary containing training history with keys:
                 'train_loss', 'val_loss', 'train_dice', 'val_dice', 'lr'
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice Score
    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_mask_analysis(mask: torch.Tensor, prefix: str = "Mask") -> None:
    """
    Print detailed analysis of mask values.
    
    Args:
        mask: Mask tensor [B, H, W] or [H, W]
        prefix: Prefix for print statements
    """
    print(f"{prefix} Analysis:")
    print(f"  Shape: {mask.shape}")
    print(f"  Unique values: {torch.unique(mask).tolist()}")
    print(f"  Min/Max: {mask.min().item()}, {mask.max().item()}")
    print(f"  Class distribution:")
    for c in range(3):
        count = (mask == c).sum().item()
        percentage = 100.0 * count / mask.numel()
        print(f"    Class {c}: {count} pixels ({percentage:.2f}%)")
