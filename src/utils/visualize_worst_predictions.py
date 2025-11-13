"""
Visualize worst model predictions based on IoU scores.

Shows images where the model performed poorly to help identify failure cases.
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
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    model = UNet(
        n_channels=3,
        n_classes=3,
        base_channels=checkpoint.get('base_channels', 64)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def calculate_iou(pred: np.ndarray, target: np.ndarray, class_id: int):
    """Calculate IoU for a specific class."""
    pred_class = (pred == class_id)
    target_class = (target == class_id)

    intersection = np.logical_and(pred_class, target_class).sum()
    union = np.logical_or(pred_class, target_class).sum()

    if union == 0:
        return float('nan')

    return intersection / union


def create_colored_mask(mask: np.ndarray, alpha: float = 0.5):
    """
    Create colored overlay for segmentation mask.

    Args:
        mask: Segmentation mask (H, W) with values 0, 1, 2
        alpha: Transparency for overlay

    Returns:
        RGBA colored mask
    """
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


def find_worst_predictions(
    checkpoint_path: str,
    root_dir: str,
    num_worst: int = 8,
    metric: str = 'cup',  # 'cup', 'disc', or 'mean'
    device: str = 'cuda',
    image_size: int = 256,
    seed: int = 42
):
    """
    Find worst predictions based on IoU.

    Args:
        checkpoint_path: Path to model checkpoint
        root_dir: Project root directory
        num_worst: Number of worst samples to return
        metric: Which metric to use ('cup', 'disc', or 'mean')
        device: Device to run on
        image_size: Image size used during training
        seed: Random seed

    Returns:
        List of tuples (index, iou_score, image, gt_mask, pred_mask)
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

    # Evaluate all samples and store results
    print(f"\nEvaluating all samples to find worst {metric} predictions...")
    sample_results = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            # Get sample metadata
            sample_info = test_dataset.samples[idx]
            dataset_name = sample_info['dataset']
            image_filename = Path(sample_info['image_path']).name

            # Get sample
            image, gt_mask = test_dataset[idx]

            # Predict
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            gt_mask_np = gt_mask.numpy()

            # Calculate IoUs
            iou_disc = calculate_iou(pred_mask, gt_mask_np, 1)
            iou_cup = calculate_iou(pred_mask, gt_mask_np, 2)

            # Select metric
            if metric == 'cup':
                score = iou_cup
            elif metric == 'disc':
                score = iou_disc
            else:  # mean
                score = (iou_disc + iou_cup) / 2

            # Denormalize image for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            image_np = (image_denorm.permute(
                1, 2, 0).numpy() * 255).astype(np.uint8)

            sample_results.append({
                'idx': idx,
                'dataset': dataset_name,
                'filename': image_filename,
                'score': score,
                'iou_disc': iou_disc,
                'iou_cup': iou_cup,
                'image': image_np,
                'gt_mask': gt_mask_np,
                'pred_mask': pred_mask
            })

    # Sort by score (ascending - worst first)
    sample_results.sort(key=lambda x: x['score'] if not np.isnan(
        x['score']) else float('inf'))

    # Return worst samples
    return sample_results[:num_worst]


def visualize_worst_predictions(
    checkpoint_path: str,
    root_dir: str,
    num_worst: int = 8,
    metric: str = 'cup',
    save_path: str = None,
    device: str = 'cuda',
    image_size: int = 256,
    seed: int = 42
):
    """
    Visualize worst model predictions.

    Args:
        checkpoint_path: Path to model checkpoint
        root_dir: Project root directory
        num_worst: Number of worst samples to visualize
        metric: Which metric to use ('cup', 'disc', or 'mean')
        save_path: Path to save visualization (optional)
        device: Device to run on
        image_size: Image size used during training
        seed: Random seed
    """
    # Find worst predictions
    worst_samples = find_worst_predictions(
        checkpoint_path=checkpoint_path,
        root_dir=root_dir,
        num_worst=num_worst,
        metric=metric,
        device=device,
        image_size=image_size,
        seed=seed
    )

    # Create figure
    fig, axes = plt.subplots(num_worst, 3, figsize=(15, 5 * num_worst))
    if num_worst == 1:
        axes = axes.reshape(1, -1)

    # Plot each sample
    for i, sample in enumerate(worst_samples):
        image_np = sample['image']
        gt_mask = sample['gt_mask']
        pred_mask = sample['pred_mask']

        # Create overlays
        gt_overlay = overlay_mask_on_image(image_np, gt_mask, alpha=0.4)
        pred_overlay = overlay_mask_on_image(image_np, pred_mask, alpha=0.4)

        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(
            f'{sample["dataset"]} | {sample["filename"]}', fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title('Ground Truth', fontsize=12)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title(f'Prediction\nDisc IoU: {sample["iou_disc"]:.3f}, Cup IoU: {sample["iou_cup"]:.3f}',
                             fontsize=12)
        axes[i, 2].axis('off')

        print(
            f'{sample["dataset"]} | {sample["filename"]}: Disc IoU={sample["iou_disc"]:.3f}, Cup IoU={sample["iou_cup"]:.3f}')

    # Add legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.4, label='Optic Disc'),
        Patch(facecolor='green', alpha=0.4, label='Optic Cup')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.0), fontsize=12, frameon=True)

    # Add title
    metric_name = {'cup': 'Cup', 'disc': 'Disc', 'mean': 'Mean'}[metric]
    fig.suptitle(f'Worst {num_worst} Predictions (by {metric_name} IoU)',
                 fontsize=16, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()

    return fig, worst_samples


def main():
    """Main function for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize worst model predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--root-dir', type=str, default='.',
                        help='Project root directory')
    parser.add_argument('--num-worst', type=int, default=8,
                        help='Number of worst samples to visualize')
    parser.add_argument('--metric', type=str, default='cup', choices=['cup', 'disc', 'mean'],
                        help='Metric to use for finding worst samples')
    parser.add_argument('--save-path', type=str, default='results/worst_predictions.png',
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
    visualize_worst_predictions(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        num_worst=args.num_worst,
        metric=args.metric,
        save_path=args.save_path,
        device=args.device,
        image_size=args.image_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
