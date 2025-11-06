"""
Compare UNet and ResNet-UNet models side-by-side.

This script evaluates and compares the performance of the vanilla UNet
and ResNet-based UNet on the test set.
"""

from models.resnet_unet import ResNetUNet, ResNetUNetLite
from models.unet import UNet
from data_loader.dataset import get_dataloaders
from data_loader.transforms import get_validation_transforms
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def calculate_metrics(pred, target, num_classes=3):
    """
    Calculate comprehensive metrics.

    Args:
        pred: [B, C, H, W] - predicted logits
        target: [B, H, W] - target labels

    Returns:
        dict with IoU, Dice, Precision, Recall for each class
    """
    pred = torch.argmax(pred, dim=1)  # [B, H, W]

    metrics = {
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': []
    }

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        tn = (~pred_cls & ~target_cls).sum().float()

        # IoU
        union = (pred_cls | target_cls).sum().float()
        iou = (tp / union).item() if union > 0 else 0.0

        # Dice
        dice = (2 * tp / (2 * tp + fp + fn)
                ).item() if (2 * tp + fp + fn) > 0 else 0.0

        # Precision
        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0

        # Recall
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0

        metrics['iou'].append(iou)
        metrics['dice'].append(dice)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

    return metrics


@torch.no_grad()
def evaluate_model(model, test_loader, device, model_name='Model'):
    """Evaluate model on test set"""
    model.eval()

    all_metrics = {
        'iou': np.zeros(3),
        'dice': np.zeros(3),
        'precision': np.zeros(3),
        'recall': np.zeros(3)
    }

    num_batches = 0

    pbar = tqdm(test_loader, desc=f'Evaluating {model_name}')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate metrics
        batch_metrics = calculate_metrics(outputs, masks)

        for key in all_metrics.keys():
            all_metrics[key] += np.array(batch_metrics[key])

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'iou_disc': f"{batch_metrics['iou'][1]:.4f}",
            'iou_cup': f"{batch_metrics['iou'][2]:.4f}"
        })

    # Average metrics
    for key in all_metrics.keys():
        all_metrics[key] /= num_batches

    return all_metrics


def load_model(checkpoint_path, model_type='unet', device='cuda'):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model_type: 'unet', 'resnet34', or 'resnet18'
        device: Device to load model on
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    # Create model
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=3, base_channels=64)
    elif model_type == 'resnet18':
        model = ResNetUNetLite(n_channels=3, n_classes=3, pretrained=False)
    else:  # resnet34
        model = ResNetUNet(n_channels=3, n_classes=3, pretrained=False)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def compare_models(
    root_dir,
    unet_checkpoint,
    resnet_checkpoint,
    resnet_type='resnet34',
    batch_size=8,
    image_size=512,
    num_workers=4,
    use_clahe=False,
    save_dir='results'
):
    """
    Compare UNet and ResNet-UNet models.

    Args:
        root_dir: Project root directory
        unet_checkpoint: Path to UNet checkpoint
        resnet_checkpoint: Path to ResNet-UNet checkpoint
        resnet_type: 'resnet34' or 'resnet18'
        batch_size: Batch size for evaluation
        image_size: Input image size
        num_workers: Number of data loading workers
        use_clahe: Use CLAHE preprocessing
        save_dir: Directory to save results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_train=None,
        transform_val=get_validation_transforms(
            image_size=image_size, use_clahe=use_clahe
        ),
        filter_incomplete=True
    )

    print(f"Test set size: {len(test_loader.dataset)} images")

    # Load models
    print("\nLoading UNet model...")
    unet_model, unet_ckpt = load_model(unet_checkpoint, 'unet', device)
    print(f"  Loaded from epoch {unet_ckpt['epoch']}")
    print(f"  Val Loss: {unet_ckpt['val_loss']:.4f}")

    print(f"\nLoading ResNet-UNet ({resnet_type}) model...")
    resnet_model, resnet_ckpt = load_model(
        resnet_checkpoint, resnet_type, device)
    print(f"  Loaded from epoch {resnet_ckpt['epoch']}")
    print(f"  Val Loss: {resnet_ckpt['val_loss']:.4f}")

    # Evaluate models
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)

    unet_metrics = evaluate_model(unet_model, test_loader, device, 'UNet')
    resnet_metrics = evaluate_model(
        resnet_model, test_loader, device, f'ResNet-UNet ({resnet_type})')

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    class_names = ['Background', 'Optic Disc', 'Optic Cup']

    print("\n{:<15} {:<12} {:<12} {:<12}".format(
        'Metric', 'UNet', f'ResNet-UNet', 'Improvement'))
    print("-"*60)

    for metric_name in ['iou', 'dice', 'precision', 'recall']:
        print(f"\n{metric_name.upper()}:")
        for i, class_name in enumerate(class_names):
            unet_val = unet_metrics[metric_name][i]
            resnet_val = resnet_metrics[metric_name][i]
            improvement = ((resnet_val - unet_val) /
                           unet_val * 100) if unet_val > 0 else 0

            print(f"  {class_name:<13} {unet_val:.4f}       {resnet_val:.4f}       "
                  f"{improvement:+.2f}%")

    # Calculate mean metrics (weighted by importance)
    # Cup and Disc are more important than background
    weights = np.array([0.1, 0.5, 0.4])  # Background, Disc, Cup

    print("\n" + "-"*60)
    print("WEIGHTED AVERAGE (Disc: 50%, Cup: 40%, BG: 10%):")
    print("-"*60)

    for metric_name in ['iou', 'dice']:
        unet_avg = np.average(unet_metrics[metric_name], weights=weights)
        resnet_avg = np.average(resnet_metrics[metric_name], weights=weights)
        improvement = ((resnet_avg - unet_avg) /
                       unet_avg * 100) if unet_avg > 0 else 0

        print(f"{metric_name.upper():<14} {unet_avg:.4f}       {resnet_avg:.4f}       "
              f"{improvement:+.2f}%")

    # Save results
    results = {
        'unet': {k: v.tolist() for k, v in unet_metrics.items()},
        'resnet_unet': {k: v.tolist() for k, v in resnet_metrics.items()},
        'class_names': class_names,
        'resnet_type': resnet_type
    }

    results_path = save_dir / 'model_comparison.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot comparison
    plot_comparison(unet_metrics, resnet_metrics, class_names,
                    save_path=save_dir / 'comparison_plot.png',
                    resnet_type=resnet_type)

    return results


def plot_comparison(unet_metrics, resnet_metrics, class_names, save_path=None, resnet_type='resnet34'):
    """Plot comparison between models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics_to_plot = ['iou', 'dice', 'precision', 'recall']
    titles = ['IoU (Intersection over Union)',
              'Dice Coefficient', 'Precision', 'Recall']

    x = np.arange(len(class_names))
    width = 0.35

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 2, idx % 2]

        unet_vals = unet_metrics[metric]
        resnet_vals = resnet_metrics[metric]

        bars1 = ax.bar(x - width/2, unet_vals, width, label='UNet', alpha=0.8)
        bars2 = ax.bar(x + width/2, resnet_vals, width,
                       label=f'ResNet-UNet ({resnet_type})', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    return fig


if __name__ == '__main__':
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare UNet and ResNet-UNet models')
    parser.add_argument('--root_dir', type=str,
                        default='/home/robolab/dev/CAP5410-roi-enhancer-odoc',
                        help='Project root directory')
    parser.add_argument('--unet_checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to UNet checkpoint')
    parser.add_argument('--resnet_checkpoint', type=str,
                        default='checkpoints_resnet/best_model.pth',
                        help='Path to ResNet-UNet checkpoint')
    parser.add_argument('--resnet_type', type=str, default='resnet34',
                        choices=['resnet34', 'resnet18'],
                        help='ResNet backbone type')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--clahe', action='store_true',
                        help='Use CLAHE preprocessing')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Compare models
    results = compare_models(
        root_dir=args.root_dir,
        unet_checkpoint=args.unet_checkpoint,
        resnet_checkpoint=args.resnet_checkpoint,
        resnet_type=args.resnet_type,
        batch_size=args.batch_size,
        use_clahe=args.clahe,
        save_dir=args.save_dir
    )
