"""
Visualize model predictions on test images.

Shows side-by-side comparison of:
- Left: Original cropped image
- Middle: Ground truth segmentation overlay
- Right: Predicted segmentation overlay
"""

from data_loader.transforms import get_validation_transforms
from data_loader.dataset import GlaucomaDataset
from models.unet import UNet
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from PIL import Image

# Add src to path if needed
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    # Extract model parameters from checkpoint
    model = UNet(
        n_channels=3,
        n_classes=3,
        base_channels=checkpoint.get('base_channels', 64)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def create_colored_mask(mask: np.ndarray, alpha: float = 0.5):
    """
    Create colored overlay for segmentation mask.

    Args:
        mask: Segmentation mask (H, W) with values 0, 1, 2
        alpha: Transparency for overlay

    Returns:
        RGBA colored mask
    """
    # Define colors for each class
    colors = {
        0: [0, 0, 0, 0],        # Background: transparent
        1: [255, 0, 0, 255],     # Disc: red
        2: [0, 255, 0, 255]      # Cup: green
    }

    h, w = mask.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)

    for class_id, color in colors.items():
        colored[mask == class_id] = color

    # Apply alpha to non-background pixels
    colored[:, :, 3] = np.where(mask > 0, int(alpha * 255), 0)

    return colored


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """
    Overlay colored segmentation mask on image.

    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        mask: Segmentation mask (H, W) with values 0, 1, 2
        alpha: Transparency for overlay

    Returns:
        Image with overlay (H, W, 3)
    """
    colored_mask = create_colored_mask(mask, alpha)

    # Convert image to float for blending
    image_float = image.astype(float)

    # Blend where mask is not background
    mask_alpha = colored_mask[:, :, 3:4] / 255.0
    blended = image_float * (1 - mask_alpha) + \
        colored_mask[:, :, :3] * mask_alpha

    return blended.astype(np.uint8)


def predict_on_image(model, image: np.ndarray, device: str = 'cuda'):
    """
    Run model prediction on a single image.

    Args:
        model: Trained model
        image: Input image tensor (C, H, W)
        device: Device to run on

    Returns:
        Predicted mask (H, W)
    """
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    return pred_mask


def visualize_predictions(
    checkpoint_path: str,
    root_dir: str,
    num_samples: int = 4,
    save_path: str = None,
    device: str = 'cuda',
    image_size: int = 256,
    seed: int = 42
):
    """
    Visualize model predictions on test images.

    Args:
        checkpoint_path: Path to model checkpoint
        root_dir: Project root directory
        num_samples: Number of test samples to visualize
        save_path: Path to save visualization (optional)
        device: Device to run on
        image_size: Image size used during training
        seed: Random seed for reproducible sample selection
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = GlaucomaDataset(
        root_dir=root_dir,
        split='test',
        transform=get_validation_transforms(image_size=image_size),
        seed=seed,
        filter_incomplete=True
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Select random samples
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), size=min(
        num_samples, len(test_dataset)), replace=False)

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Process each sample
    for i, idx in enumerate(indices):
        # Get sample metadata
        sample_info = test_dataset.samples[idx]
        dataset_name = sample_info['dataset']
        image_filename = Path(sample_info['image_path']).name

        # Get image and ground truth
        image_tensor, gt_mask = test_dataset[idx]

        # Get original image (denormalize)
        # Reverse normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = image_tensor * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)

        # Convert to numpy for visualization
        image_np = (image_denorm.permute(
            1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask_np = gt_mask.numpy()

        # Get prediction
        pred_mask = predict_on_image(model, image_tensor, device)

        # Create overlays
        gt_overlay = overlay_mask_on_image(image_np, gt_mask_np, alpha=0.4)
        pred_overlay = overlay_mask_on_image(image_np, pred_mask, alpha=0.4)

        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'{dataset_name} | {image_filename}', fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title('Ground Truth', fontsize=12)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title('Prediction', fontsize=12)
        axes[i, 2].axis('off')

        # Calculate IoU for this sample
        iou_disc = calculate_iou(gt_mask_np, pred_mask, class_id=1)
        iou_cup = calculate_iou(gt_mask_np, pred_mask, class_id=2)

        print(
            f"{dataset_name} | {image_filename}: IoU Disc={iou_disc:.3f}, IoU Cup={iou_cup:.3f}")

    # Add legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.4, label='Optic Disc'),
        Patch(facecolor='green', alpha=0.4, label='Optic Cup')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.0), fontsize=12, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()

    return fig


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray, class_id: int):
    """Calculate IoU for a specific class."""
    mask1_class = (mask1 == class_id)
    mask2_class = (mask2 == class_id)

    intersection = np.logical_and(mask1_class, mask2_class).sum()
    union = np.logical_or(mask1_class, mask2_class).sum()

    if union == 0:
        return 0.0

    return intersection / union


def main():
    """Main function for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--root-dir', type=str, default='.',
                        help='Project root directory')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--save-path', type=str, default='results/predictions_visualization.png',
                        help='Path to save visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size used during training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Ensure save directory exists
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run visualization
    visualize_predictions(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        num_samples=args.num_samples,
        save_path=args.save_path,
        device=args.device,
        image_size=args.image_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
