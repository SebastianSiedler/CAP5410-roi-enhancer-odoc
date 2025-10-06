"""
Evaluation metrics for optic disc/cup segmentation
Includes Dice score, IoU, and CDR (Cup-to-Disc Ratio) calculation
"""
import torch
import numpy as np


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Calculate Dice coefficient for binary segmentation

    Args:
        pred: Predicted segmentation (probabilities) of shape (B, C, H, W) or (H, W)
        target: Ground truth segmentation of shape (B, C, H, W) or (H, W)
        smooth: Smoothing factor

    Returns:
        Dice coefficient (0-1, higher is better)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()

    dice = (2. * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )

    return dice.item()


def multi_class_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """
    Calculate Dice coefficient for each class

    Args:
        pred: Predicted segmentation logits of shape (B, C, H, W)
        target: Ground truth segmentation of shape (B, C, H, W)
        num_classes: Number of classes

    Returns:
        Dictionary with Dice scores for each class
    """
    pred = torch.sigmoid(pred)
    dice_scores = {}

    class_names = ['disc', 'cup'] if num_classes == 2 else [
        f'class_{i}' for i in range(num_classes)]

    for i in range(num_classes):
        pred_i = pred[:, i, :, :]
        target_i = target[:, i, :, :]
        dice_i = dice_coefficient(pred_i, target_i)
        dice_scores[class_names[i]] = dice_i

    # Mean Dice
    dice_scores['mean'] = np.mean(list(dice_scores.values()))

    return dice_scores


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Calculate Intersection over Union (IoU) score

    Args:
        pred: Predicted segmentation (probabilities) of shape (B, C, H, W) or (H, W)
        target: Ground truth segmentation of shape (B, C, H, W) or (H, W)
        smooth: Smoothing factor

    Returns:
        IoU score (0-1, higher is better)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def calculate_cdr(
    disc_mask: torch.Tensor,
    cup_mask: torch.Tensor,
    method: str = 'area'
) -> float:
    """
    Calculate Cup-to-Disc Ratio (CDR)

    Args:
        disc_mask: Binary mask for optic disc of shape (H, W) or (B, H, W)
        cup_mask: Binary mask for optic cup of shape (H, W) or (B, H, W)
        method: Method to calculate CDR ('area' or 'vertical')
            - 'area': ratio of cup area to disc area
            - 'vertical': ratio of vertical cup diameter to vertical disc diameter

    Returns:
        CDR value (0-1)
    """
    # Ensure binary masks
    disc_mask = (disc_mask > 0.5).float()
    cup_mask = (cup_mask > 0.5).float()

    if method == 'area':
        # Area-based CDR
        disc_area = disc_mask.sum().item()
        cup_area = cup_mask.sum().item()

        if disc_area == 0:
            return 0.0

        cdr = cup_area / disc_area

    elif method == 'vertical':
        # Vertical diameter-based CDR
        cdr = calculate_vertical_cdr(disc_mask, cup_mask)

    else:
        raise ValueError(f"Unknown CDR method: {method}")

    return min(cdr, 1.0)  # CDR should not exceed 1.0


def calculate_vertical_cdr(disc_mask: torch.Tensor, cup_mask: torch.Tensor) -> float:
    """
    Calculate CDR based on vertical diameter ratio

    Args:
        disc_mask: Binary disc mask
        cup_mask: Binary cup mask

    Returns:
        Vertical CDR
    """
    # Convert to numpy for easier processing
    if isinstance(disc_mask, torch.Tensor):
        disc_mask = disc_mask.cpu().numpy()
    if isinstance(cup_mask, torch.Tensor):
        cup_mask = cup_mask.cpu().numpy()

    # Remove batch dimension if present
    if len(disc_mask.shape) == 3:
        disc_mask = disc_mask[0]
    if len(cup_mask.shape) == 3:
        cup_mask = cup_mask[0]

    # Find vertical extent of disc and cup
    disc_vert = np.where(disc_mask.sum(axis=1) > 0)[0]
    cup_vert = np.where(cup_mask.sum(axis=1) > 0)[0]

    if len(disc_vert) == 0 or len(cup_vert) == 0:
        return 0.0

    disc_diameter = disc_vert[-1] - disc_vert[0]
    cup_diameter = cup_vert[-1] - cup_vert[0]

    if disc_diameter == 0:
        return 0.0

    return cup_diameter / disc_diameter


def batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Calculate comprehensive metrics for a batch

    Args:
        pred: Predicted segmentation logits of shape (B, 2, H, W)
        target: Ground truth segmentation of shape (B, 2, H, W)
        threshold: Threshold for binary prediction

    Returns:
        Dictionary with all metrics
    """
    # Apply sigmoid and threshold
    pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > threshold).float()

    # Calculate Dice scores
    dice_scores = multi_class_dice(pred, target, num_classes=2)

    # Calculate IoU for each class
    iou_disc = iou_score(pred_binary[:, 0, :, :], target[:, 0, :, :])
    iou_cup = iou_score(pred_binary[:, 1, :, :], target[:, 1, :, :])

    # Calculate CDR for each sample in batch
    batch_size = pred.shape[0]
    cdr_values = []
    cdr_errors = []

    for i in range(batch_size):
        pred_cdr = calculate_cdr(
            pred_binary[i, 0, :, :],
            pred_binary[i, 1, :, :],
            method='area'
        )
        target_cdr = calculate_cdr(
            target[i, 0, :, :],
            target[i, 1, :, :],
            method='area'
        )

        cdr_values.append(pred_cdr)
        cdr_errors.append(abs(pred_cdr - target_cdr))

    metrics = {
        'dice_disc': dice_scores['disc'],
        'dice_cup': dice_scores['cup'],
        'dice_mean': dice_scores['mean'],
        'iou_disc': iou_disc,
        'iou_cup': iou_cup,
        'cdr_mean': np.mean(cdr_values),
        'cdr_mae': np.mean(cdr_errors),  # Mean Absolute Error for CDR
        'cdr_std': np.std(cdr_values)
    }

    return metrics


class MetricsTracker:
    """Helper class to track metrics during training/validation"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics"""
        self.metrics = {
            'dice_disc': [],
            'dice_cup': [],
            'dice_mean': [],
            'iou_disc': [],
            'iou_cup': [],
            'cdr_mean': [],
            'cdr_mae': [],
            'loss': []
        }

    def update(self, metrics_dict: dict):
        """Update metrics with new values"""
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_average(self) -> dict:
        """Get average of all tracked metrics"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        return avg_metrics

    def get_std(self) -> dict:
        """Get standard deviation of all tracked metrics"""
        std_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                std_metrics[key] = np.std(values)
            else:
                std_metrics[key] = 0.0
        return std_metrics
