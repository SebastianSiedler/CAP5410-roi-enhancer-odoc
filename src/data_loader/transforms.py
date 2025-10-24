"""
Data augmentation transforms for optic disc/cup segmentation.

Provides standard transforms for training and validation/testing.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transforms(image_size: int = 512):
    """
    Get training transforms with data augmentation.

    Args:
        image_size: Target image size (will resize to image_size x image_size)

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        # Resize all images to same size
        A.Resize(image_size, image_size),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5
        ),

        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.5),

        # Blur/Noise (subtle, for robustness)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(p=0.2),
        ], p=0.3),

        # Normalize using ImageNet stats (standard practice)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def get_validation_transforms(image_size: int = 512):
    """
    Get validation/test transforms without data augmentation.

    Args:
        image_size: Target image size (will resize to image_size x image_size)

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        # Only resize, no augmentation
        A.Resize(image_size, image_size),

        # Same normalization as training
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def get_minimal_transforms(image_size: int = 512):
    """
    Get minimal transforms (resize + normalize only).
    Useful for quick testing or inference.

    Args:
        image_size: Target image size (will resize to image_size x image_size)

    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
