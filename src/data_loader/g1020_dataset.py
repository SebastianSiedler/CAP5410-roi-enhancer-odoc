"""
G1020 Dataset loader for optic disc and cup segmentation.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional, Callable
import torchvision.transforms as transforms


class G1020Dataset(Dataset):
    """
    G1020 Dataset for optic disc and cup segmentation.
    
    The G1020 dataset contains fundus images with masks where:
    - 0: Background
    - 1: Optic disc
    - 2: Optic cup
    
    Args:
        csv_file: Path to CSV file (G1020_train.csv, G1020_val.csv, or G1020_test.csv)
        root_dir: Root directory of G1020 dataset
        images_dir: Subdirectory containing images (default: 'Images_Cropped/img')
        masks_dir: Subdirectory containing masks (default: 'Masks_Cropped/img')
        transform: Optional transform for images
        target_size: Target size for resizing (height, width)
        use_clahe: Whether to apply CLAHE preprocessing
        clahe_clip_limit: CLAHE clip limit
        clahe_mode: CLAHE color space ('LAB' or 'RGB')
        augmentation: Optional augmentation function (for training only)
    """
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        images_dir: str = 'Images_Cropped/img',
        masks_dir: str = 'Masks_Cropped/img',
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_mode: str = 'LAB',
        augmentation: Optional[Callable] = None
    ):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, images_dir)
        self.masks_dir = os.path.join(root_dir, masks_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.augmentation = augmentation
        
        # Initialize CLAHE preprocessor if enabled
        if self.use_clahe:
            from data_loader.clahe_preprocessing import CLAHEPreprocessor
            self.clahe_preprocessor = CLAHEPreprocessor(
                clip_limit=clahe_clip_limit,
                tile_grid_size=(8, 8),
                apply_to=clahe_mode
            )
        else:
            self.clahe_preprocessor = None
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        # Transform for masks (no normalization, nearest neighbor interpolation)
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (3, H, W) - RGB fundus image
            mask: Tensor of shape (2, H, W) - Binary masks for disc (ch 0) and cup (ch 1)
        """
        # Get image name and label
        row = self.df.iloc[idx]
        image_name = row['imageID']
        label = row['binaryLabels']
        
        # Construct paths
        image_path = os.path.join(self.images_dir, image_name)
        # Mask has .png extension (not .jpg)
        mask_name = image_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply CLAHE preprocessing if enabled
        if self.clahe_preprocessor is not None:
            image = self.clahe_preprocessor(image)
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        
        # Split mask into disc and cup channels
        # Mask values: 0=background, 1=disc, 2=cup
        # Disc mask: everything that is disc or cup (values 1 or 2)
        disc_mask = ((mask_array == 1) | (mask_array == 2)).astype(np.uint8) * 255
        # Cup mask: only cup (value 2)
        cup_mask = (mask_array == 2).astype(np.uint8) * 255
        
        # Convert to PIL Images
        disc_mask = Image.fromarray(disc_mask)
        cup_mask = Image.fromarray(cup_mask)
        
        # Apply transforms
        image = self.transform(image)
        disc_mask = self.mask_transform(disc_mask)
        cup_mask = self.mask_transform(cup_mask)
        
        # Stack masks: [disc, cup]
        mask = torch.cat([disc_mask, cup_mask], dim=0)
        
        # Apply augmentation if provided (typically only for training)
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        
        return image, mask
    
    def get_label(self, idx: int) -> int:
        """Get the glaucoma label for an image."""
        return self.df.iloc[idx]['binaryLabels']
    
    def get_image_name(self, idx: int) -> str:
        """Get the image name for display purposes."""
        return self.df.iloc[idx]['imageID']
