"""
REFUGE Dataset Loader for Optic Disc and Cup Segmentation
"""
import os
from typing import Tuple, Optional, Callable
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RetinaDataset(Dataset):
    """
    PyTorch Dataset for REFUGE retina images with optic disc/cup segmentation masks.

    Loads cropped fundus images and their corresponding disc/cup segmentation masks.

    Args:
        root_dir: Root directory containing the REFUGE dataset
        csv_file: Path to CSV file containing image/mask metadata
        transform: Optional transform to apply to images
        target_size: Tuple of (height, width) to resize images and masks
        use_cropped: Whether to use cropped images (default True for ROI-based approach)
        cropped_masks_dir: Directory containing generated cropped masks (if use_cropped=True)
    """

    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_cropped: bool = True,
        cropped_masks_dir: str = 'datasets/REFUGE_cropped_masks',
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_mode: str = 'LAB',
        augmentation: Optional[Callable] = None
    ):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)

        # Filter out rows with NaN in multimaskName (validation data mixed in training CSV)
        self.df = self.df[self.df['multimaskName'].notna()
                          ].reset_index(drop=True)

        self.transform = transform
        self.target_size = target_size
        self.use_cropped = use_cropped
        self.cropped_masks_dir = cropped_masks_dir
        self.use_clahe = use_clahe
        self.augmentation = augmentation  # NEW: Store augmentation callable
        
        # Initialize CLAHE preprocessor if enabled
        if self.use_clahe:
            from .clahe_preprocessing import CLAHEPreprocessor
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

        # Transform for masks (no normalization)
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
        # Get image and mask paths from dataframe
        row = self.df.iloc[idx]

        # Extract folder ID from the training folder structure
        folder_name = self._extract_folder_name(row)

        # Construct paths based on whether we're using cropped versions
        if self.use_cropped:
            # Use the generated cropped masks from REFUGE_cropped_masks
            img_path = os.path.join(self.cropped_masks_dir, folder_name, 
                                    f'{folder_name}_cropped.jpg')
            disc_mask_path = os.path.join(self.cropped_masks_dir, folder_name,
                                          f'{folder_name}_disc_cropped.bmp')
            cup_mask_path = os.path.join(self.cropped_masks_dir, folder_name,
                                         f'{folder_name}_cup_cropped.bmp')
        else:
            # Use full resolution images and masks
            img_path = os.path.join(self.root_dir, 'Training-400',
                                    folder_name, f'{folder_name}.jpg')
            disc_mask_path = os.path.join(self.root_dir, 'Training-400',
                                          folder_name, f'{folder_name}_disc.bmp')
            cup_mask_path = os.path.join(self.root_dir, 'Training-400',
                                         folder_name, f'{folder_name}_cup.bmp')

        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply CLAHE preprocessing if enabled
        if self.clahe_preprocessor is not None:
            image = self.clahe_preprocessor(image)

        # Load masks
        disc_mask = Image.open(disc_mask_path).convert('L')  # Grayscale
        cup_mask = Image.open(cup_mask_path).convert('L')

        # Apply transforms
        image = self.transform(image)
        disc_mask = self.mask_transform(disc_mask)
        cup_mask = self.mask_transform(cup_mask)

        # Binarize masks (threshold at 0.5 after normalization)
        disc_mask = (disc_mask < 0.75).float()  # BMP: 128=disc foreground
        cup_mask = (cup_mask < 0.5).float()  # BMP: 0=cup foreground

        # Combine disc and cup masks into a single tensor (2 channels)
        mask = torch.cat([disc_mask, cup_mask], dim=0)

        # Apply augmentation if provided (after transforms, synchronized for image and mask)
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        return image, mask

    def _extract_folder_name(self, row: pd.Series) -> str:
        """
        Extract folder name from the CSV row.
        The multimaskName column contains paths like 'Dataset-new-2/Training-400/0853/0853_cup.bmp'
        """
        multimask_path = row['multimaskName']

        # Handle NaN or invalid values
        if pd.isna(multimask_path) or not isinstance(multimask_path, str):
            raise ValueError(f"Invalid multimaskName value: {multimask_path}")

        # Extract the folder number (e.g., '0853' from the path)
        folder_name = multimask_path.split('/')[-2]
        return folder_name

    def get_label(self, idx: int) -> int:
        """
        Get the glaucoma label for the image (1=glaucoma, 0=normal)
        """
        return self.df.iloc[idx]['label']


class RetinaDatasetValidation(Dataset):
    """
    Validation dataset for REFUGE (uses Validation-400 folder)
    """

    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_cropped: bool = True,
        cropped_masks_dir: str = 'datasets/REFUGE_cropped_masks_val',
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_mode: str = 'LAB'
    ):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = target_size
        self.use_cropped = use_cropped
        self.cropped_masks_dir = cropped_masks_dir
        self.use_clahe = use_clahe
        
        # Initialize CLAHE preprocessor if enabled
        if self.use_clahe:
            from .clahe_preprocessing import CLAHEPreprocessor
            self.clahe_preprocessor = CLAHEPreprocessor(
                clip_limit=clahe_clip_limit,
                tile_grid_size=(8, 8),
                apply_to=clahe_mode
            )
        else:
            self.clahe_preprocessor = None

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        folder_name = self._extract_folder_name(row)

        # Use cropped masks if available, otherwise fall back to Validation-400
        if self.use_cropped:
            img_path = os.path.join(self.cropped_masks_dir, folder_name,
                                    f'{folder_name}_cropped.jpg')
            disc_mask_path = os.path.join(self.cropped_masks_dir, folder_name,
                                          f'{folder_name}_disc_cropped.bmp')
            cup_mask_path = os.path.join(self.cropped_masks_dir, folder_name,
                                         f'{folder_name}_cup_cropped.bmp')
        else:
            img_path = os.path.join(self.root_dir, 'Validation-400',
                                    folder_name, f'{folder_name}.jpg')
            disc_mask_path = os.path.join(self.root_dir, 'Validation-400',
                                          folder_name, f'{folder_name}_disc.bmp')
            cup_mask_path = os.path.join(self.root_dir, 'Validation-400',
                                         folder_name, f'{folder_name}_cup.bmp')

        image = Image.open(img_path).convert('RGB')
        
        # Apply CLAHE preprocessing if enabled
        if self.clahe_preprocessor is not None:
            image = self.clahe_preprocessor(image)
        
        disc_mask = Image.open(disc_mask_path).convert('L')
        cup_mask = Image.open(cup_mask_path).convert('L')

        image = self.transform(image)
        disc_mask = self.mask_transform(disc_mask)
        cup_mask = self.mask_transform(cup_mask)

        disc_mask = (disc_mask < 0.75).float()  # BMP: 128=disc foreground
        cup_mask = (cup_mask < 0.5).float()  # BMP: 0=cup foreground

        mask = torch.cat([disc_mask, cup_mask], dim=0)

        return image, mask

    def _extract_folder_name(self, row: pd.Series) -> str:
        multimask_path = row['multimaskName']
        # Extract folder name from path
        parts = multimask_path.split('/')
        # Find the numeric folder name
        for part in parts:
            if part.isdigit():
                return part
        # Fallback: use the second-to-last part
        return parts[-2]

    def get_label(self, idx: int) -> int:
        return self.df.iloc[idx]['label']


class RetinaDatasetTest(Dataset):
    """
    Test dataset for REFUGE with support for cropped masks.
    Similar to RetinaDatasetValidation but for Test-400 folder.
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_cropped: bool = True,
        cropped_masks_dir: str = 'datasets/REFUGE_cropped_masks_test',
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_mode: str = 'LAB',
        augmentation: Optional[Callable] = None
    ):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = target_size
        self.use_cropped = use_cropped
        self.cropped_masks_dir = cropped_masks_dir
        self.use_clahe = use_clahe
        self.augmentation = augmentation  # NEW: Store augmentation (typically None for test)
        
        # Initialize CLAHE preprocessor if enabled
        if self.use_clahe:
            from .clahe_preprocessing import CLAHEPreprocessor
            self.clahe_preprocessor = CLAHEPreprocessor(
                clip_limit=clahe_clip_limit,
                tile_grid_size=(8, 8),
                apply_to=clahe_mode
            )
        else:
            self.clahe_preprocessor = None

        # Filter out rows with NaN in multimaskName
        self.df = self.df[self.df['multimaskName'].notna()
                          ].reset_index(drop=True)

        # Image transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        folder_name = self._extract_folder_name(row)

        # Use cropped masks if available, otherwise fall back to Test-400
        if self.use_cropped:
            img_path = os.path.join(self.cropped_masks_dir, folder_name,
                                    f'{folder_name}_cropped.jpg')
            disc_mask_path = os.path.join(self.cropped_masks_dir, folder_name,
                                          f'{folder_name}_disc_cropped.bmp')
            cup_mask_path = os.path.join(self.cropped_masks_dir, folder_name,
                                         f'{folder_name}_cup_cropped.bmp')
        else:
            img_path = os.path.join(self.root_dir, 'Test-400',
                                    folder_name, f'{folder_name}.jpg')
            disc_mask_path = os.path.join(self.root_dir, 'Test-400',
                                          folder_name, f'{folder_name}_disc.bmp')
            cup_mask_path = os.path.join(self.root_dir, 'Test-400',
                                         folder_name, f'{folder_name}_cup.bmp')

        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply CLAHE preprocessing if enabled
        if self.clahe_preprocessor is not None:
            image = self.clahe_preprocessor(image)

        # Load masks
        disc_mask = cv2.imread(disc_mask_path, cv2.IMREAD_GRAYSCALE)
        cup_mask = cv2.imread(cup_mask_path, cv2.IMREAD_GRAYSCALE)

        # Threshold masks: disc (128), cup (0), background (255)
        disc_mask = (disc_mask < 200).astype(np.uint8) * 255
        cup_mask = (cup_mask < 128).astype(np.uint8) * 255

        # Convert to PIL
        disc_mask = Image.fromarray(disc_mask)
        cup_mask = Image.fromarray(cup_mask)

        # Apply transforms
        image = self.transform(image)
        disc_mask = self.mask_transform(disc_mask)
        cup_mask = self.mask_transform(cup_mask)

        # Stack masks: [disc, cup]
        mask = torch.cat([disc_mask, cup_mask], dim=0)

        return image, mask

    def _extract_folder_name(self, row: pd.Series) -> str:
        """Extract folder name from multimaskName path."""
        multimask_path = row['multimaskName']
        parts = multimask_path.split('/')
        for part in parts:
            if part.isdigit():
                return part
        return parts[-2]

    def get_label(self, idx: int) -> int:
        return self.df.iloc[idx]['label']


class EnhancedRetinaDataset(Dataset):
    """
    Dataset that applies defects to images for training the enhancement pipeline
    Returns: (degraded_image, clean_image, mask)

    Args:
        root_dir: Root directory containing the REFUGE dataset
        csv_file: Path to CSV file containing image/mask metadata
        defect_simulator: Defect simulation function/class
        transform: Optional transform to apply to images
        target_size: Tuple of (height, width) to resize images and masks
        use_cropped: Whether to use cropped images
    """

    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        defect_simulator: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_cropped: bool = True
    ):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)

        # Filter out rows with NaN in multimaskName
        self.df = self.df[self.df['multimaskName'].notna()
                          ].reset_index(drop=True)

        self.defect_simulator = defect_simulator
        self.transform = transform
        self.target_size = target_size
        self.use_cropped = use_cropped

        # Image transforms (without normalization for defect simulation)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])

        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        # Normalization (applied after defect simulation)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            degraded_image: Defected image of shape (3, H, W) - normalized
            clean_image: Original clean image of shape (3, H, W) - normalized
            mask: Segmentation mask of shape (2, H, W)
        """
        # Get image and mask paths
        row = self.df.iloc[idx]
        folder_name = self._extract_folder_name(row)

        # Construct paths
        if self.use_cropped:
            img_path = os.path.join(self.root_dir, 'Training-400',
                                    folder_name, f'{folder_name}_cropped.jpg')
        else:
            img_path = os.path.join(self.root_dir, 'Training-400',
                                    folder_name, f'{folder_name}.jpg')

        disc_mask_path = os.path.join(self.root_dir, 'Training-400',
                                      folder_name, f'{folder_name}_disc.bmp')
        cup_mask_path = os.path.join(self.root_dir, 'Training-400',
                                     folder_name, f'{folder_name}_cup.bmp')

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Load masks
        disc_mask = Image.open(disc_mask_path).convert('L')
        cup_mask = Image.open(cup_mask_path).convert('L')

        # Apply transforms to get clean image (no normalization yet)
        clean_image = self.transform(image)

        # Apply defect simulation
        if self.defect_simulator is not None:
            degraded_image = self.defect_simulator(clean_image)
        else:
            degraded_image = clean_image.clone()

        # Normalize both images
        clean_image_norm = self.normalize(clean_image)
        degraded_image_norm = self.normalize(degraded_image)

        # Process masks
        disc_mask = self.mask_transform(disc_mask)
        cup_mask = self.mask_transform(cup_mask)

        # Binarize masks
        disc_mask = (disc_mask < 0.75).float()  # BMP: 128=disc foreground
        cup_mask = (cup_mask < 0.5).float()  # BMP: 0=cup foreground

        # Combine masks
        mask = torch.cat([disc_mask, cup_mask], dim=0)

        return degraded_image_norm, clean_image_norm, mask

    def _extract_folder_name(self, row: pd.Series) -> str:
        """Extract folder name from CSV row"""
        multimask_path = row['multimaskName']
        parts = multimask_path.split('/')
        for part in parts:
            if part.isdigit():
                return part
        return parts[-2]

    def get_label(self, idx: int) -> int:
        return self.df.iloc[idx]['label']


class EnhancedRetinaDatasetValidation(Dataset):
    """
    Validation dataset with defects for testing enhancement pipeline
    Returns: (degraded_image, clean_image, mask)
    """

    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        defect_simulator: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_cropped: bool = True
    ):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.defect_simulator = defect_simulator
        self.transform = transform
        self.target_size = target_size
        self.use_cropped = use_cropped

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        folder_name = self._extract_folder_name(row)

        # Use Validation-400 folder
        if self.use_cropped:
            img_path = os.path.join(self.root_dir, 'Validation-400',
                                    folder_name, f'{folder_name}_cropped.jpg')
        else:
            img_path = os.path.join(self.root_dir, 'Validation-400',
                                    folder_name, f'{folder_name}.jpg')

        disc_mask_path = os.path.join(self.root_dir, 'Validation-400',
                                      folder_name, f'{folder_name}_disc.bmp')
        cup_mask_path = os.path.join(self.root_dir, 'Validation-400',
                                     folder_name, f'{folder_name}_cup.bmp')

        image = Image.open(img_path).convert('RGB')
        disc_mask = Image.open(disc_mask_path).convert('L')
        cup_mask = Image.open(cup_mask_path).convert('L')

        clean_image = self.transform(image)

        if self.defect_simulator is not None:
            degraded_image = self.defect_simulator(clean_image)
        else:
            degraded_image = clean_image.clone()

        clean_image_norm = self.normalize(clean_image)
        degraded_image_norm = self.normalize(degraded_image)

        disc_mask = self.mask_transform(disc_mask)
        cup_mask = self.mask_transform(cup_mask)

        disc_mask = (disc_mask < 0.75).float()  # BMP: 128=disc foreground
        cup_mask = (cup_mask < 0.5).float()  # BMP: 0=cup foreground

        mask = torch.cat([disc_mask, cup_mask], dim=0)

        return degraded_image_norm, clean_image_norm, mask

    def _extract_folder_name(self, row: pd.Series) -> str:
        multimask_path = row['multimaskName']
        parts = multimask_path.split('/')
        for part in parts:
            if part.isdigit():
                return part
        return parts[-2]

    def get_label(self, idx: int) -> int:
        return self.df.iloc[idx]['label']
