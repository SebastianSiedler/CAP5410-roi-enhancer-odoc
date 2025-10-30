"""
Evaluate model on entire test dataset and compute metrics.

Computes comprehensive metrics including:
- Mean IoU per class
- Dice score per class
- Pixel accuracy
- Per-sample statistics
"""

from data_loader.transforms import get_validation_transforms
from data_loader.dataset import GlaucomaDataset
from models.unet import UNet
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json

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
        return float('nan')  # No ground truth or prediction

    return intersection / union


def calculate_dice(pred: np.ndarray, target: np.ndarray, class_id: int):
    """Calculate Dice score for a specific class."""
    pred_class = (pred == class_id)
    target_class = (target == class_id)

    intersection = np.logical_and(pred_class, target_class).sum()
    total = pred_class.sum() + target_class.sum()

    if total == 0:
        return float('nan')

    return (2 * intersection) / total


def calculate_pixel_accuracy(pred: np.ndarray, target: np.ndarray):
    """Calculate overall pixel accuracy."""
    return (pred == target).sum() / pred.size


def evaluate_test_set(
    checkpoint_path: str,
    root_dir: str,
    device: str = 'cuda',
    image_size: int = 256,
    seed: int = 42,
    save_path: str = None
):
    """
    Evaluate model on entire test dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        root_dir: Project root directory
        device: Device to run on
        image_size: Image size used during training
        seed: Random seed
        save_path: Path to save results JSON (optional)

    Returns:
        Dictionary with evaluation metrics
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

    # Initialize metrics storage
    ious_bg = []
    ious_disc = []
    ious_cup = []
    dice_bg = []
    dice_disc = []
    dice_cup = []
    pixel_accs = []

    # Evaluate each sample
    print("\nEvaluating test set...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            # Get sample
            image, gt_mask = test_dataset[idx]

            # Predict
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            gt_mask = gt_mask.numpy()

            # Calculate metrics
            iou_bg = calculate_iou(pred_mask, gt_mask, 0)
            iou_disc = calculate_iou(pred_mask, gt_mask, 1)
            iou_cup = calculate_iou(pred_mask, gt_mask, 2)

            dice_bg_ = calculate_dice(pred_mask, gt_mask, 0)
            dice_disc_ = calculate_dice(pred_mask, gt_mask, 1)
            dice_cup_ = calculate_dice(pred_mask, gt_mask, 2)

            pixel_acc = calculate_pixel_accuracy(pred_mask, gt_mask)

            # Store (filter out NaN values)
            if not np.isnan(iou_bg):
                ious_bg.append(iou_bg)
            if not np.isnan(iou_disc):
                ious_disc.append(iou_disc)
            if not np.isnan(iou_cup):
                ious_cup.append(iou_cup)

            if not np.isnan(dice_bg_):
                dice_bg.append(dice_bg_)
            if not np.isnan(dice_disc_):
                dice_disc.append(dice_disc_)
            if not np.isnan(dice_cup_):
                dice_cup.append(dice_cup_)

            pixel_accs.append(pixel_acc)

    # Compute statistics
    results = {
        'test_set_size': len(test_dataset),
        'metrics': {
            'iou': {
                'background': {
                    'mean': float(np.mean(ious_bg)),
                    'std': float(np.std(ious_bg)),
                    'min': float(np.min(ious_bg)),
                    'max': float(np.max(ious_bg)),
                    'median': float(np.median(ious_bg))
                },
                'optic_disc': {
                    'mean': float(np.mean(ious_disc)),
                    'std': float(np.std(ious_disc)),
                    'min': float(np.min(ious_disc)),
                    'max': float(np.max(ious_disc)),
                    'median': float(np.median(ious_disc))
                },
                'optic_cup': {
                    'mean': float(np.mean(ious_cup)),
                    'std': float(np.std(ious_cup)),
                    'min': float(np.min(ious_cup)),
                    'max': float(np.max(ious_cup)),
                    'median': float(np.median(ious_cup))
                },
                'mean_iou': float(np.mean([np.mean(ious_bg), np.mean(ious_disc), np.mean(ious_cup)]))
            },
            'dice': {
                'background': {
                    'mean': float(np.mean(dice_bg)),
                    'std': float(np.std(dice_bg)),
                    'min': float(np.min(dice_bg)),
                    'max': float(np.max(dice_bg)),
                    'median': float(np.median(dice_bg))
                },
                'optic_disc': {
                    'mean': float(np.mean(dice_disc)),
                    'std': float(np.std(dice_disc)),
                    'min': float(np.min(dice_disc)),
                    'max': float(np.max(dice_disc)),
                    'median': float(np.median(dice_disc))
                },
                'optic_cup': {
                    'mean': float(np.mean(dice_cup)),
                    'std': float(np.std(dice_cup)),
                    'min': float(np.min(dice_cup)),
                    'max': float(np.max(dice_cup)),
                    'median': float(np.median(dice_cup))
                },
                'mean_dice': float(np.mean([np.mean(dice_bg), np.mean(dice_disc), np.mean(dice_cup)]))
            },
            'pixel_accuracy': {
                'mean': float(np.mean(pixel_accs)),
                'std': float(np.std(pixel_accs)),
                'min': float(np.min(pixel_accs)),
                'max': float(np.max(pixel_accs)),
                'median': float(np.median(pixel_accs))
            }
        },
        'config': {
            'checkpoint': checkpoint_path,
            'image_size': image_size,
            'device': device
        }
    }

    # Print results
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTest Set Size: {results['test_set_size']} images")

    print("\n--- IoU Scores ---")
    print(
        f"Background:  {results['metrics']['iou']['background']['mean']:.4f} ± {results['metrics']['iou']['background']['std']:.4f}")
    print(
        f"Optic Disc:  {results['metrics']['iou']['optic_disc']['mean']:.4f} ± {results['metrics']['iou']['optic_disc']['std']:.4f}")
    print(
        f"Optic Cup:   {results['metrics']['iou']['optic_cup']['mean']:.4f} ± {results['metrics']['iou']['optic_cup']['std']:.4f}")
    print(f"Mean IoU:    {results['metrics']['iou']['mean_iou']:.4f}")

    print("\n--- Dice Scores ---")
    print(
        f"Background:  {results['metrics']['dice']['background']['mean']:.4f} ± {results['metrics']['dice']['background']['std']:.4f}")
    print(
        f"Optic Disc:  {results['metrics']['dice']['optic_disc']['mean']:.4f} ± {results['metrics']['dice']['optic_disc']['std']:.4f}")
    print(
        f"Optic Cup:   {results['metrics']['dice']['optic_cup']['mean']:.4f} ± {results['metrics']['dice']['optic_cup']['std']:.4f}")
    print(f"Mean Dice:   {results['metrics']['dice']['mean_dice']:.4f}")

    print("\n--- Pixel Accuracy ---")
    print(
        f"Overall:     {results['metrics']['pixel_accuracy']['mean']:.4f} ± {results['metrics']['pixel_accuracy']['std']:.4f}")

    print("\n--- Detailed Statistics ---")
    print("\nOptic Disc IoU:")
    print(f"  Min:    {results['metrics']['iou']['optic_disc']['min']:.4f}")
    print(f"  Median: {results['metrics']['iou']['optic_disc']['median']:.4f}")
    print(f"  Max:    {results['metrics']['iou']['optic_disc']['max']:.4f}")

    print("\nOptic Cup IoU:")
    print(f"  Min:    {results['metrics']['iou']['optic_cup']['min']:.4f}")
    print(f"  Median: {results['metrics']['iou']['optic_cup']['median']:.4f}")
    print(f"  Max:    {results['metrics']['iou']['optic_cup']['max']:.4f}")

    print("\n" + "=" * 80)

    # Save results if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")

    return results


def main():
    """Main function for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--root-dir', type=str, default='.',
                        help='Project root directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size used during training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-path', type=str, default='results/test_evaluation.json',
                        help='Path to save results JSON')

    args = parser.parse_args()

    # Run evaluation
    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        device=args.device,
        image_size=args.image_size,
        seed=args.seed,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
