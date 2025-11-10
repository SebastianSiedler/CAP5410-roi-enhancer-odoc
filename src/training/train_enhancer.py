"""
Training functions for the two-phase enhancer + UNet training.

Phase 1: Train enhancer only with frozen UNet
Phase 2: Fine-tune both enhancer and UNet jointly
"""

from training.train import calculate_iou, CombinedLoss
from models.unet import UNet
from models.enhancer import ImageEnhancer
import os
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
    # Keep UNet in eval mode even though model.train() was called
    model.unet.eval()

    total_loss = 0
    total_seg_loss = 0
    total_l1_loss = 0
    total_iou = np.zeros(3)

    pbar = tqdm(train_loader, desc='Phase 1 Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        enhanced, outputs = model(images)

        # Segmentation loss
        seg_loss = criterion(outputs, masks)

        # L1 regularization: penalize large changes
        # Compare enhanced vs original in normalized space
        l1_loss = torch.mean(torch.abs(enhanced - images))

        # Combined loss
        loss = seg_loss + l1_weight * l1_loss

        # Backward pass
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

    Similar to phase1 but both models are trainable.
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

        # Forward pass
        optimizer.zero_grad()
        enhanced, outputs = model(images)

        # Segmentation loss
        seg_loss = criterion(outputs, masks)

        # L1 regularization
        l1_loss = torch.mean(torch.abs(enhanced - images))

        # Combined loss
        loss = seg_loss + l1_weight * l1_loss

        # Backward pass
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

        # Forward pass
        enhanced, outputs = model(images)

        # Segmentation loss
        seg_loss = criterion(outputs, masks)

        # L1 regularization
        l1_loss = torch.mean(torch.abs(enhanced - images))

        # Combined loss
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
