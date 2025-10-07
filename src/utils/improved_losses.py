"""
Improved Loss Functions for Optic Disc/Cup Segmentation
Implements Focal Loss and weighted combinations to handle class imbalance
and improve cup segmentation performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples by down-weighting easy examples.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter >= 0 (gamma=0 is equivalent to BCE)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (before sigmoid) of shape (B, C, H, W)
            target: Ground truth masks of shape (B, C, H, W)
        
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate focal term: (1 - pt)^gamma
        # pt is the probability of the true class
        pt = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Combine
        focal_loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Particularly effective for imbalanced datasets.
    """
    
    def __init__(self, smooth=1.0, reduction='mean'):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (after sigmoid) of shape (B, C, H, W)
            target: Ground truth masks of shape (B, C, H, W)
        
        Returns:
            Dice loss value (1 - Dice coefficient)
        """
        # Flatten spatial dimensions
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss.
    Allows controlling the trade-off between false positives and false negatives.
    Good for imbalanced segmentation tasks like cup detection.
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives (alpha + beta should = 1)
            smooth: Smoothing constant
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (after sigmoid) of shape (B, C, H, W)
            target: Ground truth masks of shape (B, C, H, W)
        """
        # Flatten
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        
        # True positives, false positives, false negatives
        tp = (pred * target).sum(dim=2)
        fp = (pred * (1 - target)).sum(dim=2)
        fn = ((1 - pred) * target).sum(dim=2)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return (1 - tversky).mean()


class ImprovedCombinedLoss(nn.Module):
    """
    Improved combined loss with emphasis on cup segmentation.
    Uses Focal Loss for classification and Dice Loss for overlap.
    Applies higher weights to cup predictions.
    """
    
    def __init__(
        self,
        cup_weight=2.0,
        use_focal=True,
        focal_alpha_disc=0.25,
        focal_alpha_cup=0.5,
        focal_gamma=2.0,
        dice_smooth=1.0
    ):
        """
        Args:
            cup_weight: Multiplicative weight for cup losses (>1 emphasizes cup)
            use_focal: Whether to use Focal Loss (True) or BCE (False)
            focal_alpha_disc: Focal loss alpha for disc
            focal_alpha_cup: Focal loss alpha for cup (higher = more weight)
            focal_gamma: Focal loss gamma (higher = focus on hard examples)
            dice_smooth: Smoothing for Dice loss
        """
        super().__init__()
        self.cup_weight = cup_weight
        self.use_focal = use_focal
        
        if use_focal:
            self.disc_focal = FocalLoss(alpha=focal_alpha_disc, gamma=focal_gamma)
            self.cup_focal = FocalLoss(alpha=focal_alpha_cup, gamma=focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits of shape (B, 2, H, W)
            target: Ground truth masks of shape (B, 2, H, W)
        
        Returns:
            Combined weighted loss
        """
        # Split disc and cup predictions/targets
        pred_disc = pred[:, 0:1]
        pred_cup = pred[:, 1:2]
        target_disc = target[:, 0:1]
        target_cup = target[:, 1:2]
        
        # Calculate disc losses
        if self.use_focal:
            disc_focal = self.disc_focal(pred_disc, target_disc)
        else:
            disc_focal = self.bce_loss(pred_disc, target_disc)
        
        disc_dice = self.dice_loss(torch.sigmoid(pred_disc), target_disc)
        disc_loss = disc_focal + disc_dice
        
        # Calculate cup losses (with higher weight)
        if self.use_focal:
            cup_focal = self.cup_focal(pred_cup, target_cup)
        else:
            cup_focal = self.bce_loss(pred_cup, target_cup)
        
        cup_dice = self.dice_loss(torch.sigmoid(pred_cup), target_cup)
        cup_loss = cup_focal + cup_dice
        
        # Combine with emphasis on cup
        total_loss = disc_loss + self.cup_weight * cup_loss
        
        # Normalize by total weight
        total_loss = total_loss / (1 + self.cup_weight)
        
        return total_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss to emphasize accurate boundary prediction.
    Particularly useful for small structures like the cup.
    """
    
    def __init__(self, theta=5):
        """
        Args:
            theta: Width of boundary region to emphasize
        """
        super().__init__()
        self.theta = theta
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (after sigmoid) of shape (B, C, H, W)
            target: Ground truth masks of shape (B, C, H, W)
        """
        # Calculate boundary using gradient
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        
        # Pad target
        target_pad = F.pad(target, (1, 1, 1, 1), mode='replicate')
        
        # Calculate gradients
        grad_x = F.conv2d(target_pad, sobel_x)
        grad_y = F.conv2d(target_pad, sobel_y)
        
        # Boundary mask (where gradient is non-zero)
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary = (boundary > 0).float()
        
        # Calculate loss only on boundary regions
        boundary_loss = F.binary_cross_entropy(pred, target, reduction='none')
        boundary_loss = boundary_loss * boundary
        
        return boundary_loss.sum() / (boundary.sum() + 1e-6)


def test_losses():
    """Test loss functions"""
    # Create dummy data
    batch_size = 2
    pred = torch.randn(batch_size, 2, 512, 512)  # Logits
    target = torch.randint(0, 2, (batch_size, 2, 512, 512)).float()
    
    # Test Focal Loss
    focal = FocalLoss()
    focal_loss = focal(pred, target)
    print(f"✅ Focal Loss: {focal_loss.item():.4f}")
    
    # Test Dice Loss
    dice = DiceLoss()
    dice_loss = dice(torch.sigmoid(pred), target)
    print(f"✅ Dice Loss: {dice_loss.item():.4f}")
    
    # Test Improved Combined Loss
    combined = ImprovedCombinedLoss(cup_weight=2.0)
    combined_loss = combined(pred, target)
    print(f"✅ Improved Combined Loss: {combined_loss.item():.4f}")
    
    # Test Tversky Loss
    tversky = TverskyLoss()
    tversky_loss = tversky(torch.sigmoid(pred), target)
    print(f"✅ Tversky Loss: {tversky_loss.item():.4f}")
    
    print("\n✅ All loss function tests passed!")


if __name__ == '__main__':
    test_losses()
