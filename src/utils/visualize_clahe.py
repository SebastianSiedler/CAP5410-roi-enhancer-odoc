"""
Visualize the effect of CLAHE on fundus images.

Shows original image vs CLAHE-enhanced image side by side.
"""

from data_loader.dataset import GlaucomaDataset
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
    """
    Apply CLAHE to RGB image.

    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


def visualize_clahe_effect(
    root_dir: str,
    num_samples: int = 4,
    save_path: str = None,
    seed: int = 42,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
):
    """
    Visualize CLAHE effect on sample images.

    Args:
        root_dir: Project root directory
        num_samples: Number of samples to visualize
        save_path: Path to save visualization (optional)
        seed: Random seed
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
    """
    # Load dataset (without transforms to get raw images)
    dataset = GlaucomaDataset(
        root_dir=root_dir,
        split='test',
        transform=None,  # No transforms - we want raw images
        seed=seed,
        filter_incomplete=True
    )

    print(f"Test dataset size: {len(dataset)}")

    # Select random samples
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=min(
        num_samples, len(dataset)), replace=False)

    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Get image (it will be a tensor from the dataset)
        image_tensor, _ = dataset[idx]

        # Convert tensor to numpy array (H, W, C) in range [0, 255]
        if isinstance(image_tensor, Image.Image):
            image_np = np.array(image_tensor)
        else:
            # It's already a tensor
            image_np = (image_tensor.permute(
                1, 2, 0).numpy() * 255).astype(np.uint8)

        # Apply CLAHE
        enhanced = apply_clahe(
            image_np, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Sample {idx}\nOriginal', fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(enhanced)
        axes[i, 1].set_title(
            f'With CLAHE\n(clip_limit={clip_limit}, grid={tile_grid_size})', fontsize=12)
        axes[i, 1].axis('off')

    plt.suptitle('CLAHE Effect on Fundus Images', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()

    return fig
