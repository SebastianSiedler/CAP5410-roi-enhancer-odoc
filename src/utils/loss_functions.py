"""
Loss functions for optic disc/cup segmentation
Includes Dice Loss, BCE Loss, and combined losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation

    Args:
        smooth: Smoothing factor to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted segmentation (logits or probabilities) of shape (B, C, H, W)
            target: Ground truth segmentation of shape (B, C, H, W)

        Returns:
            Dice loss (scalar)
        """
        # Apply sigmoid if input is logits
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()

        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # Return Dice loss
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross Entropy Loss

    Args:
        dice_weight: Weight for Dice loss component
        bce_weight: Weight for BCE loss component
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted segmentation logits of shape (B, C, H, W)
            target: Ground truth segmentation of shape (B, C, H, W)

        Returns:
            Combined loss (scalar)
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)

        return self.dice_weight * dice + self.bce_weight * bce


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Dice Loss (computes Dice per class and averages)
    Useful for disc and cup segmentation

    Args:
        smooth: Smoothing factor
        class_weights: Optional weights for each class
    """

    def __init__(self, smooth: float = 1.0, class_weights: list = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted segmentation logits of shape (B, C, H, W)
            target: Ground truth segmentation of shape (B, C, H, W)

        Returns:
            Mean Dice loss across classes
        """
        pred = torch.sigmoid(pred)

        dice_per_class = []
        num_classes = pred.shape[1]

        for c in range(num_classes):
            pred_c = pred[:, c, :, :].contiguous().view(-1)
            target_c = target[:, c, :, :].contiguous().view(-1)

            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (
                pred_c.sum() + target_c.sum() + self.smooth
            )

            # Weight if provided
            weight = self.class_weights[c] if self.class_weights else 1.0
            dice_per_class.append(weight * (1 - dice))

        return torch.mean(torch.stack(dice_per_class))


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation with Dice and BCE
    Specifically designed for disc/cup segmentation

    Args:
        lambda_dice: Weight for Dice loss
        lambda_bce: Weight for BCE loss
        class_weights: Weights for disc and cup classes [w_disc, w_cup]
    """

    def __init__(
        self,
        lambda_dice: float = 0.5,
        lambda_bce: float = 0.5,
        class_weights: list = None
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.dice_loss = MultiClassDiceLoss(class_weights=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted segmentation logits of shape (B, 2, H, W)
            target: Ground truth segmentation of shape (B, 2, H, W)

        Returns:
            Combined loss
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)

        total_loss = self.lambda_dice * dice + self.lambda_bce * bce

        return total_loss


class TaskAwareLoss(nn.Module):
    """
    Task-Aware Loss combining enhancement and segmentation losses
    Used for joint training of enhancer and segmenter

    L_total = λ * L_enhancement + (1-λ) * L_segmentation

    Args:
        lambda_enhancement: Weight for enhancement loss (λ)
        enhancement_loss_fn: Loss function for enhancement (e.g., MSE, L1)
        segmentation_loss_fn: Loss function for segmentation
    """

    def __init__(
        self,
        lambda_enhancement: float = 0.3,
        enhancement_loss_fn: nn.Module = None,
        segmentation_loss_fn: nn.Module = None
    ):
        super().__init__()
        self.lambda_enhancement = lambda_enhancement

        # Default losses if not provided
        self.enhancement_loss = enhancement_loss_fn or nn.L1Loss()
        self.segmentation_loss = segmentation_loss_fn or CombinedSegmentationLoss()

    def forward(
        self,
        enhanced_img: torch.Tensor,
        target_img: torch.Tensor,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor
    ) -> tuple:
        """
        Args:
            enhanced_img: Enhanced image from enhancer network
            target_img: Target clean image
            seg_pred: Predicted segmentation from segmenter
            seg_target: Ground truth segmentation

        Returns:
            tuple: (total_loss, enhancement_loss, segmentation_loss)
        """
        loss_enh = self.enhancement_loss(enhanced_img, target_img)
        loss_seg = self.segmentation_loss(seg_pred, seg_target)

        total_loss = (
            self.lambda_enhancement * loss_enh +
            (1 - self.lambda_enhancement) * loss_seg
        )

        return total_loss, loss_enh, loss_seg
