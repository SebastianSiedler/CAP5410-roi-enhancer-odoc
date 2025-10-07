"""
Test script for trained U-Net model on REFUGE test dataset
Evaluates model performance and generates visualizations
"""
import os
import sys
from pathlib import Path
import argparse
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.unet import UNet
from data_loader.dataset import RetinaDatasetTest
from utils.metrics import batch_metrics, MetricsTracker


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test trained U-Net model')
    
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/baseline_unet/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/REFUGE',
                        help='Path to REFUGE dataset directory')
    parser.add_argument('--test_csv', type=str,
                        default='datasets/REFUGE/REFUGE1Test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str,
                        default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of predictions')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Get model configuration from checkpoint
    args = checkpoint.get('args', {})
    base_features = args.get('base_features', 64)
    bilinear = args.get('bilinear', False)
    
    print(f"Model config: base_features={base_features}, bilinear={bilinear}")
    
    # Create model
    model = UNet(
        n_channels=3,
        n_classes=2,
        bilinear=bilinear,
        base_features=base_features
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def calculate_cdr(cup_mask, disc_mask):
    """Calculate Cup-to-Disc Ratio"""
    cup_area = cup_mask.sum()
    disc_area = disc_mask.sum()
    
    if disc_area > 0:
        return (cup_area / disc_area).item()
    return 0.0


def test_model(model, test_loader, device):
    """Test model on test dataset"""
    print("\nTesting model...")
    
    metrics_tracker = MetricsTracker()
    all_predictions = []
    all_masks = []
    all_cdrs_pred = []
    all_cdrs_gt = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Calculate batch metrics
            metrics = batch_metrics(outputs, masks)
            metrics_tracker.update(metrics)
            
            # Store predictions and masks for visualization
            all_predictions.append(predictions.cpu())
            all_masks.append(masks.cpu())
            
            # Calculate CDR for each sample
            pred_binary = (predictions > 0.5).float()
            masks_binary = (masks > 0.5).float()
            
            for i in range(predictions.shape[0]):
                cdr_pred = calculate_cdr(pred_binary[i, 1], pred_binary[i, 0])
                cdr_gt = calculate_cdr(masks_binary[i, 1], masks_binary[i, 0])
                all_cdrs_pred.append(cdr_pred)
                all_cdrs_gt.append(cdr_gt)
            
            # Update progress bar
            pbar.set_postfix({
                'dice': f"{metrics['dice_mean']:.4f}",
                'cdr_mae': f"{metrics['cdr_mae']:.4f}"
            })
    
    # Get average metrics
    avg_metrics = metrics_tracker.get_average()
    
    # Concatenate all predictions and masks
    all_predictions = torch.cat(all_predictions, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    return avg_metrics, all_predictions, all_masks, all_cdrs_pred, all_cdrs_gt


def print_results(metrics, cdrs_pred, cdrs_gt):
    """Print detailed test results"""
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    print(f"\nSegmentation Metrics:")
    print(f"  Overall Dice Score:     {metrics['dice_mean']:.4f}")
    print(f"  Disc Dice Score:        {metrics['dice_disc']:.4f}")
    print(f"  Cup Dice Score:         {metrics['dice_cup']:.4f}")
    
    print(f"\nCup-to-Disc Ratio (CDR):")
    print(f"  Mean Absolute Error:    {metrics['cdr_mae']:.4f}")
    
    # Calculate additional CDR statistics
    cdr_errors = [abs(p - g) for p, g in zip(cdrs_pred, cdrs_gt)]
    print(f"  Max Error:              {max(cdr_errors):.4f}")
    print(f"  Min Error:              {min(cdr_errors):.4f}")
    print(f"  Std Error:              {np.std(cdr_errors):.4f}")
    
    print(f"\nCDR Distribution:")
    print(f"  Ground Truth Mean:      {np.mean(cdrs_gt):.4f} ± {np.std(cdrs_gt):.4f}")
    print(f"  Predicted Mean:         {np.mean(cdrs_pred):.4f} ± {np.std(cdrs_pred):.4f}")
    
    print("\n" + "="*60)


def visualize_predictions(test_dataset, predictions, masks, output_dir, num_samples=10):
    """Create visualization of predictions"""
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    for idx in tqdm(indices, desc="Creating visualizations"):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Get image and predictions
        image, mask_gt = test_dataset[idx]
        pred = predictions[idx]
        
        # Convert image from tensor to numpy (denormalize)
        img_np = image.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # Ground truth masks
        disc_gt = mask_gt[0].numpy()
        cup_gt = mask_gt[1].numpy()
        
        # Predicted masks
        disc_pred = pred[0].numpy()
        cup_pred = pred[1].numpy()
        
        # Binary predictions (threshold at 0.5)
        disc_pred_bin = (disc_pred > 0.5).astype(float)
        cup_pred_bin = (cup_pred > 0.5).astype(float)
        
        # Calculate individual dice scores
        disc_dice = 2 * (disc_pred_bin * disc_gt).sum() / (disc_pred_bin.sum() + disc_gt.sum() + 1e-8)
        cup_dice = 2 * (cup_pred_bin * cup_gt).sum() / (cup_pred_bin.sum() + cup_gt.sum() + 1e-8)
        
        # Calculate CDR
        cdr_gt = cup_gt.sum() / (disc_gt.sum() + 1e-8)
        cdr_pred = cup_pred_bin.sum() / (disc_pred_bin.sum() + 1e-8)
        
        # Row 1: Ground Truth
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(disc_gt, cmap='gray')
        axes[0, 1].set_title(f'GT Disc')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cup_gt, cmap='gray')
        axes[0, 2].set_title(f'GT Cup')
        axes[0, 2].axis('off')
        
        # Overlay GT
        overlay_gt = img_np.copy()
        overlay_gt[disc_gt > 0.5] = overlay_gt[disc_gt > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
        overlay_gt[cup_gt > 0.5] = overlay_gt[cup_gt > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5
        axes[0, 3].imshow(overlay_gt)
        axes[0, 3].set_title(f'GT Overlay (CDR={cdr_gt:.3f})')
        axes[0, 3].axis('off')
        
        # Row 2: Predictions
        axes[1, 0].imshow(img_np)
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(disc_pred, cmap='gray')
        axes[1, 1].set_title(f'Pred Disc (Dice={disc_dice:.3f})')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cup_pred, cmap='gray')
        axes[1, 2].set_title(f'Pred Cup (Dice={cup_dice:.3f})')
        axes[1, 2].axis('off')
        
        # Overlay predictions
        overlay_pred = img_np.copy()
        overlay_pred[disc_pred_bin > 0.5] = overlay_pred[disc_pred_bin > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
        overlay_pred[cup_pred_bin > 0.5] = overlay_pred[cup_pred_bin > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5
        axes[1, 3].imshow(overlay_pred)
        axes[1, 3].set_title(f'Pred Overlay (CDR={cdr_pred:.3f})')
        axes[1, 3].axis('off')
        
        plt.suptitle(f'Test Sample {idx} - Avg Dice: {(disc_dice + cup_dice)/2:.3f}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f'test_sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}")


def create_summary_plot(metrics_history, cdrs_pred, cdrs_gt, output_dir):
    """Create summary plots of test results"""
    print("\nCreating summary plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Dice score distribution
    dice_disc = [metrics_history[i]['dice_disc'] for i in range(len(metrics_history))]
    dice_cup = [metrics_history[i]['dice_cup'] for i in range(len(metrics_history))]
    
    axes[0].hist(dice_disc, bins=30, alpha=0.6, label='Disc', color='green')
    axes[0].hist(dice_cup, bins=30, alpha=0.6, label='Cup', color='red')
    axes[0].axvline(np.mean(dice_disc), color='green', linestyle='--', linewidth=2, label=f'Disc Mean: {np.mean(dice_disc):.3f}')
    axes[0].axvline(np.mean(dice_cup), color='red', linestyle='--', linewidth=2, label=f'Cup Mean: {np.mean(dice_cup):.3f}')
    axes[0].set_xlabel('Dice Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Dice Score Distribution on Test Set')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: CDR scatter plot
    axes[1].scatter(cdrs_gt, cdrs_pred, alpha=0.5, s=20)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Ground Truth CDR')
    axes[1].set_ylabel('Predicted CDR')
    axes[1].set_title(f'CDR Prediction (MAE={np.mean([abs(p-g) for p, g in zip(cdrs_pred, cdrs_gt)]):.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'test_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary plot saved to: {save_path}")


def save_results(metrics, cdrs_pred, cdrs_gt, output_dir):
    """Save test results to JSON file"""
    results = {
        'metrics': {
            'dice_mean': float(metrics['dice_mean']),
            'dice_disc': float(metrics['dice_disc']),
            'dice_cup': float(metrics['dice_cup']),
            'cdr_mae': float(metrics['cdr_mae']),
        },
        'cdr_statistics': {
            'ground_truth_mean': float(np.mean(cdrs_gt)),
            'ground_truth_std': float(np.std(cdrs_gt)),
            'predicted_mean': float(np.mean(cdrs_pred)),
            'predicted_std': float(np.std(cdrs_pred)),
            'mae': float(np.mean([abs(p - g) for p, g in zip(cdrs_pred, cdrs_gt)])),
            'max_error': float(max([abs(p - g) for p, g in zip(cdrs_pred, cdrs_gt)])),
            'min_error': float(min([abs(p - g) for p, g in zip(cdrs_pred, cdrs_gt)])),
        },
        'num_samples': len(cdrs_gt)
    }
    
    output_file = os.path.join(output_dir, 'test_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main testing function"""
    args = get_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = RetinaDatasetTest(
        csv_file=args.test_csv,
        root_dir=args.data_dir,
        use_cropped=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Test model
    metrics, predictions, masks, cdrs_pred, cdrs_gt = test_model(model, test_loader, device)
    
    # Print results
    print_results(metrics, cdrs_pred, cdrs_gt)
    
    # Save results
    save_results(metrics, cdrs_pred, cdrs_gt, args.output_dir)
    
    # Create visualizations if requested
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        visualize_predictions(test_dataset, predictions, masks, vis_dir, args.num_vis_samples)
        
        # Note: Summary plot needs per-sample metrics, which we'd need to store during testing
        # For now, we'll skip it or calculate it from stored predictions
    
    print("\n✅ Testing completed!")


if __name__ == '__main__':
    main()
