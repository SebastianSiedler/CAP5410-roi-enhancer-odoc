"""
Evaluation utilities for segmentation and enhancement models.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from src.training.task_aware_train import compute_dice_score


def test_segmentation_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_classes: int = 3
) -> Dict[str, float]:
    """
    Test a segmentation model on a data loader.
    
    Args:
        model: Segmentation model
        test_loader: DataLoader for test data
        device: Device to run on
        n_classes: Number of segmentation classes
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        logits = model(images)
        metrics = compute_dice_score(logits, masks, n_classes=n_classes)
    
    return metrics, logits, masks


def test_pipeline(
    pipeline: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test enhancement-segmentation pipeline.
    
    Args:
        pipeline: Enhancement-segmentation pipeline
        test_loader: DataLoader for test data
        device: Device to run on
    
    Returns:
        Tuple of (original_images, enhanced_images, ground_truth, predictions)
    """
    pipeline.eval()
    
    images, masks = next(iter(test_loader))
    images_gpu = images.to(device)
    
    with torch.no_grad():
        enhanced, logits = pipeline(images_gpu)
        predictions = torch.argmax(logits, dim=1)
    
    return images, enhanced, masks, predictions
