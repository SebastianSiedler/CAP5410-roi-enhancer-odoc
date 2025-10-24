import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable, List
from PIL import Image
import torch
from torch.utils.data import Dataset


class GlaucomaDataset(Dataset):
    """
    PyTorch Dataset for optic disc/cup segmentation from G1020, ORIGA, and REFUGE datasets.

    Uses only cropped images and masks with 70/15/15 train/val/test split.

    Mask values:
        0: background
        1: optic disc
        2: optic cup
    """

    # Class variable to store the split indices (shared across all instances)
    _split_cache = {}

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        seed: int = 42,
        filter_incomplete: bool = False
    ):
        """
        Args:
            root_dir: Project root directory (datasets will be loaded from <root>/datasets/glaucoma-datasets)
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on images and masks
            seed: Random seed for reproducible splits
            filter_incomplete: If True, remove images that don't have all 3 classes (0, 1, 2)
        """
        assert split in ['train', 'val',
                         'test'], "Split must be 'train', 'val', or 'test'"

        self.root_dir = Path(root_dir) / 'datasets' / 'glaucoma-datasets'
        self.split = split
        self.transform = transform
        self.seed = seed
        self.filter_incomplete = filter_incomplete

        # Create cache key based on root_dir, seed, and filter setting
        cache_key = (str(self.root_dir), seed, filter_incomplete)

        # Load all samples if not cached
        if cache_key not in self._split_cache:
            all_samples = self._load_all_samples()
            self._create_split_cache(cache_key, all_samples)

        # Get the appropriate split from cache
        self.samples = self._split_cache[cache_key][split]

        # Print split statistics
        print(f"{self.split.upper()} split: {len(self.samples)} samples")

    def _load_all_samples(self) -> List[dict]:
        """Load all image-mask-label triplets from all three datasets."""
        all_samples = []

        # Load G1020
        all_samples.extend(self._load_g1020())

        # Load ORIGA
        all_samples.extend(self._load_origa())

        # Load REFUGE
        all_samples.extend(self._load_refuge())

        return all_samples

    def _load_g1020(self) -> List[dict]:
        """Load G1020 dataset samples."""
        samples = []
        g1020_dir = self.root_dir / 'G1020'
        images_dir = g1020_dir / 'Images_Cropped' / 'img'
        masks_dir = g1020_dir / 'Masks_Cropped' / 'img'

        if not images_dir.exists():
            print(
                f"Warning: G1020 Images_Cropped/img not found at {images_dir}")
            return samples

        # Scan images in the img subdirectory only
        for image_path in images_dir.glob('*.jpg'):
            image_name = image_path.name
            # Convert .jpg to .png for mask
            mask_name = image_name.replace('.jpg', '.png')
            mask_path = masks_dir / mask_name

            if mask_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),
                    'dataset': 'G1020'
                })

        return samples

    def _load_origa(self) -> List[dict]:
        """Load ORIGA dataset samples."""
        samples = []
        origa_dir = self.root_dir / 'ORIGA'

        # ORIGA images are in Images_Cropped/img subdirectory (same as G1020)
        images_dir = origa_dir / 'Images_Cropped' / 'img'
        masks_dir = origa_dir / 'Masks_Cropped' / 'img'

        if not images_dir.exists():
            print(
                f"Warning: ORIGA Images_Cropped/img not found at {images_dir}")
            return samples

        # Scan images in the img subdirectory
        for image_path in images_dir.glob('*.jpg'):
            image_name = image_path.name
            # Convert to .png for mask
            mask_name = image_name.replace('.jpg', '.png')
            mask_path = masks_dir / mask_name

            if mask_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),
                    'dataset': 'ORIGA'
                })

        return samples

    def _load_refuge(self) -> List[dict]:
        """
        Load REFUGE dataset samples.
        Ignores train/val/test folder structure and treats all as one pool.
        """
        samples = []
        refuge_dir = self.root_dir / 'REFUGE'

        # Combine all splits from REFUGE
        for subset in ['train', 'val', 'test']:
            images_dir = refuge_dir / subset / 'Images_Cropped' / 'img'
            masks_dir = refuge_dir / subset / 'Masks_Cropped' / 'img'

            if not images_dir.exists():
                continue

            for image_path in images_dir.glob('*.jpg'):
                image_name = image_path.name
                mask_name = image_name.replace('.jpg', '.png')
                mask_path = masks_dir / mask_name

                if mask_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'mask_path': str(mask_path),
                        'dataset': 'REFUGE'
                    })

        return samples

    def _filter_incomplete_masks(self, samples: List[dict]) -> List[dict]:
        """Filter out samples that don't have all 3 classes (0, 1, 2) in their masks."""
        from PIL import Image

        filtered_samples = []
        removed_count = 0

        for sample in samples:
            mask_path = sample['mask_path']
            mask = np.array(Image.open(mask_path))
            unique_values = np.unique(mask)

            # Keep only if it has all 3 classes
            if len(unique_values) == 3 and 0 in unique_values and 1 in unique_values and 2 in unique_values:
                filtered_samples.append(sample)
            else:
                removed_count += 1

        if removed_count > 0:
            print(
                f"  Filtered out {removed_count} images with incomplete masks")

        return filtered_samples

    def _create_split_cache(self, cache_key: tuple, all_samples: List[dict]):
        """Create and cache the train/val/test splits."""
        # Filter incomplete masks if requested
        if self.filter_incomplete:
            print("Filtering incomplete masks...")
            all_samples = self._filter_incomplete_masks(all_samples)

        indices = np.arange(len(all_samples))

        # Set random seed for reproducibility
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        # Calculate split sizes
        n_samples = len(all_samples)
        n_train = int(0.70 * n_samples)
        n_val = int(0.15 * n_samples)

        # Split indices
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        # Cache the splits
        self._split_cache[cache_key] = {
            'train': [all_samples[i] for i in train_idx],
            'val': [all_samples[i] for i in val_idx],
            'test': [all_samples[i] for i in test_idx]
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Index

        Returns:
            image: RGB image tensor
            mask: Segmentation mask tensor (values: 0, 1, 2)
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Load mask
        mask = Image.open(sample['mask_path'])
        # Convert to numpy to ensure proper value handling
        mask = np.array(mask)

        # Apply transforms if provided
        if self.transform:
            # Transform expects dict with 'image' and 'mask'
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensors if no transform provided
            image = torch.from_numpy(np.array(image)).permute(
                2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    def get_dataset_stats(self) -> dict:
        """Get statistics about the dataset split."""
        datasets = [s['dataset'] for s in self.samples]

        stats = {
            'total_samples': len(self.samples),
            'g1020_samples': datasets.count('G1020'),
            'origa_samples': datasets.count('ORIGA'),
            'refuge_samples': datasets.count('REFUGE')
        }

        return stats


def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    seed: int = 42,
    filter_incomplete: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Project root directory (datasets will be loaded from <root>/datasets/glaucoma-datasets)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        transform_train: Transforms for training data (augmentations)
        transform_val: Transforms for validation/test data (no augmentation)
        seed: Random seed for reproducible splits
        filter_incomplete: If True, remove images that don't have all 3 classes (0, 1, 2)

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = GlaucomaDataset(
        root_dir=root_dir,
        split='train',
        transform=transform_train,
        seed=seed,
        filter_incomplete=filter_incomplete
    )

    val_dataset = GlaucomaDataset(
        root_dir=root_dir,
        split='val',
        transform=transform_val,
        seed=seed,
        filter_incomplete=filter_incomplete
    )

    test_dataset = GlaucomaDataset(
        root_dir=root_dir,
        split='test',
        transform=transform_val,
        seed=seed,
        filter_incomplete=filter_incomplete
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Example usage and testing."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <root_dir>")
        print("Example: python dataset.py datasets")
        sys.exit(1)

    root_dir = sys.argv[1]

    print("Creating datasets...")
    train_dataset = GlaucomaDataset(root_dir, split='train')
    val_dataset = GlaucomaDataset(root_dir, split='val')
    test_dataset = GlaucomaDataset(root_dir, split='test')

    print("\nDataset Statistics:")
    print("="*50)
    for split_name, dataset in [('Train', train_dataset),
                                ('Val', val_dataset),
                                ('Test', test_dataset)]:
        stats = dataset.get_dataset_stats()
        print(f"\n{split_name}:")
        print(f"  Total: {stats['total_samples']}")
        print(f"  G1020: {stats['g1020_samples']}")
        print(f"  ORIGA: {stats['origa_samples']}")
        print(f"  REFUGE: {stats['refuge_samples']}")

    # Test loading a sample
    print("\n" + "="*50)
    print("Testing sample loading...")
    image, mask = train_dataset[5]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
