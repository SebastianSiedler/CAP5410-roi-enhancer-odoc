"""
Training Augmentation for Optic Disc/Cup Segmentation
Implements geometric and photometric augmentations to improve generalization
"""
import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image


class TrainingAugmentation:
    """
    Advanced augmentation for fundus image segmentation.
    Applies synchronized transforms to image and masks.
    """
    
    def __init__(self, training=True, p=0.5):
        """
        Args:
            training: Whether to apply augmentations (True for train, False for val/test)
            p: Probability of applying each augmentation
        """
        self.training = training
        self.p = p
    
    def __call__(self, image, mask_disc, mask_cup):
        """
        Apply augmentations to image and masks synchronously
        
        Args:
            image: PIL Image (RGB)
            mask_disc: PIL Image (grayscale)
            mask_cup: PIL Image (grayscale)
            
        Returns:
            Augmented image and masks
        """
        if not self.training:
            return image, mask_disc, mask_cup
        
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask_disc = TF.hflip(mask_disc)
            mask_cup = TF.hflip(mask_cup)
        
        # 2. Random Vertical Flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask_disc = TF.vflip(mask_disc)
            mask_cup = TF.vflip(mask_cup)
        
        # 3. Random Rotation
        if random.random() > self.p:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_disc = TF.rotate(mask_disc, angle, interpolation=TF.InterpolationMode.NEAREST)
            mask_cup = TF.rotate(mask_cup, angle, interpolation=TF.InterpolationMode.NEAREST)
        
        # 4. Random Affine (translation, scale, shear)
        if random.random() > self.p:
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)
            
            # Convert translate from ratio to pixels
            w, h = image.size
            translate_pixels = (int(translate[0] * w), int(translate[1] * h))
            
            image = TF.affine(
                image, angle=0, translate=translate_pixels, scale=scale, shear=shear,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            mask_disc = TF.affine(
                mask_disc, angle=0, translate=translate_pixels, scale=scale, shear=shear,
                interpolation=TF.InterpolationMode.NEAREST
            )
            mask_cup = TF.affine(
                mask_cup, angle=0, translate=translate_pixels, scale=scale, shear=shear,
                interpolation=TF.InterpolationMode.NEAREST
            )
        
        # 5. Color Jitter (only on image, not masks)
        if random.random() > self.p:
            # Brightness
            brightness_factor = random.uniform(0.7, 1.3)
            image = TF.adjust_brightness(image, brightness_factor)
            
            # Contrast
            contrast_factor = random.uniform(0.7, 1.3)
            image = TF.adjust_contrast(image, contrast_factor)
            
            # Saturation
            saturation_factor = random.uniform(0.7, 1.3)
            image = TF.adjust_saturation(image, saturation_factor)
            
            # Hue (small adjustment for fundus images)
            hue_factor = random.uniform(-0.1, 0.1)
            image = TF.adjust_hue(image, hue_factor)
        
        # 6. Random Gamma Correction (simulates lighting variations)
        if random.random() > self.p:
            gamma = random.uniform(0.8, 1.2)
            image = TF.adjust_gamma(image, gamma)
        
        # 7. Gaussian Blur (simulate slight defocus)
        if random.random() > 0.7:  # Less frequent
            kernel_size = random.choice([3, 5])
            image = TF.gaussian_blur(image, kernel_size)
        
        # 8. Random Crop and Resize (zoom in/out effect)
        if random.random() > 0.6:  # Less frequent to preserve anatomy
            w, h = image.size
            
            # Random crop size (85-100% of original)
            crop_scale = random.uniform(0.85, 1.0)
            new_w = int(w * crop_scale)
            new_h = int(h * crop_scale)
            
            # Random crop position
            i = random.randint(0, h - new_h) if h > new_h else 0
            j = random.randint(0, w - new_w) if w > new_w else 0
            
            # Crop
            image = TF.crop(image, i, j, new_h, new_w)
            mask_disc = TF.crop(mask_disc, i, j, new_h, new_w)
            mask_cup = TF.crop(mask_cup, i, j, new_h, new_w)
            
            # Resize back to original size
            image = TF.resize(image, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
            mask_disc = TF.resize(mask_disc, (h, w), interpolation=TF.InterpolationMode.NEAREST)
            mask_cup = TF.resize(mask_cup, (h, w), interpolation=TF.InterpolationMode.NEAREST)
        
        return image, mask_disc, mask_cup


def test_augmentation():
    """Test augmentation pipeline"""
    # Create dummy data
    image = Image.new('RGB', (512, 512), color='red')
    mask_disc = Image.new('L', (512, 512), color=128)
    mask_cup = Image.new('L', (512, 512), color=64)
    
    # Test augmentation
    aug = TrainingAugmentation(training=True, p=0.5)
    image_aug, mask_disc_aug, mask_cup_aug = aug(image, mask_disc, mask_cup)
    
    print("✅ Augmentation test passed!")
    print(f"   Image size: {image_aug.size}")
    print(f"   Disc mask size: {mask_disc_aug.size}")
    print(f"   Cup mask size: {mask_cup_aug.size}")


if __name__ == '__main__':
    test_augmentation()
