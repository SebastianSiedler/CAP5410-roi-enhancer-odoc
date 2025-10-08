"""
Strong Data Augmentation for Medical Image Segmentation

Implements aggressive augmentation strategies to combat overfitting
when training with limited data (e.g., 400 samples).

Key techniques:
- Geometric: Rotation, flipping, affine transforms
- Color: Brightness, contrast, saturation adjustments
- Noise: Gaussian blur, sharpness
- Occlusion: Random erasing
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import numpy as np
from typing import Tuple


class StrongAugmentation:
    """
    Strong augmentation pipeline for fundus images and segmentation masks.
    
    Applies synchronized geometric and photometric transformations to combat overfitting.
    
    Args:
        p: Probability of applying augmentation (default: 0.8)
        geometric_p: Probability of geometric transforms (default: 0.8)
        color_p: Probability of color transforms (default: 0.5)
        blur_p: Probability of blur/sharpness (default: 0.3)
    """
    
    def __init__(
        self, 
        p: float = 0.8,
        geometric_p: float = 0.8,
        color_p: float = 0.5,
        blur_p: float = 0.3
    ):
        self.p = p
        self.geometric_p = geometric_p
        self.color_p = color_p
        self.blur_p = blur_p
        
        # Geometric transforms (applied to both image and mask)
        self.geometric_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(
                degrees=30,
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10,
                interpolation=T.InterpolationMode.BILINEAR
            ),
        ])
        
        # Color transforms (applied only to image)
        self.color_transform = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )
        
        # Blur/Sharpness (applied only to image)
        self.blur_transform = T.GaussianBlur(
            kernel_size=5, 
            sigma=(0.1, 2.0)
        )
        
        self.sharpen_transform = T.RandomAdjustSharpness(
            sharpness_factor=2, 
            p=0.5
        )
    
    def __call__(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to image and mask.
        
        Args:
            image: RGB image tensor [3, H, W]
            mask: Segmentation mask tensor [2, H, W] (disc, cup)
            
        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        # Apply augmentation with probability p
        if torch.rand(1) < self.p:
            
            # 1. Geometric transforms (synchronized for image and mask)
            if torch.rand(1) < self.geometric_p:
                # Stack image and mask for synchronized transform
                combined = torch.cat([image, mask], dim=0)  # [5, H, W]
                combined = self.geometric_transform(combined)
                
                # Split back
                image = combined[:3]
                mask = combined[3:]
            
            # 2. Color transforms (only on image)
            if torch.rand(1) < self.color_p:
                image = self.color_transform(image)
            
            # 3. Blur/Sharpness (only on image)
            if torch.rand(1) < self.blur_p:
                if torch.rand(1) < 0.5:
                    image = self.blur_transform(image)
                else:
                    image = self.sharpen_transform(image)
        
        return image, mask


class MediumAugmentation:
    """
    Medium augmentation pipeline (less aggressive than StrongAugmentation).
    
    Use this if strong augmentation is too aggressive for your data.
    
    Args:
        p: Probability of applying augmentation (default: 0.6)
    """
    
    def __init__(self, p: float = 0.6):
        self.p = p
        
        self.geometric_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            ),
        ])
        
        self.color_transform = T.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02
        )
    
    def __call__(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply medium augmentation."""
        if torch.rand(1) < self.p:
            # Geometric
            combined = torch.cat([image, mask], dim=0)
            combined = self.geometric_transform(combined)
            image = combined[:3]
            mask = combined[3:]
            
            # Color
            image = self.color_transform(image)
        
        return image, mask


def get_augmentation(augmentation_type: str = 'strong'):
    """
    Factory function to get augmentation pipeline.
    
    Args:
        augmentation_type: 'none', 'medium', 'strong'
        
    Returns:
        Augmentation callable
    """
    if augmentation_type == 'none':
        return lambda x, y: (x, y)
    elif augmentation_type == 'medium':
        return MediumAugmentation(p=0.6)
    elif augmentation_type == 'strong':
        return StrongAugmentation(p=0.8)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
