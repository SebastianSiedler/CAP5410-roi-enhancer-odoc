"""
Test script for EE-TransUNet model
Loads checkpoint, evaluates on test set, creates visualizations and metrics
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import argparse

from src.models.ee_transunet import VisionTransformer, CONFIGS
from src.data_loader.dataset import RetinaDatasetTest
from torch.utils.data import DataLoader
from src.utils.metrics import dice_coefficient, calculate_cdr


def get_args():
    parser = argparse.ArgumentParser(description='Test EE-TransUNet model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default='datasets/REFUGE',
                       help='Path to dataset directory')
    parser.add_argument('--test_csv', type=str,
                       default='datasets/REFUGE/REFUGE1Test.csv',
                       help='Path to test CSV file')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str,
                       default='experiments/ee_transunet/checkpoint_epoch_39.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='ViT-Tiny',
                       help='Model variant')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str,
                       default='test_results_ee_transunet',
                       help='Directory to save test results')
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                       help='Save visualization images')
    parser.add_argument('--num_visualizations', type=int, default=10,
                       help='Number of test samples to visualize')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for testing')
    
    return parser.parse_args()


def load_model(checkpoint_path, model_name, img_size, device):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Get model configuration
    config = CONFIGS[model_name]
    
    # Create model
    model = VisionTransformer(config, img_size=img_size, num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch: {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (no epoch info)")
    
    model = model.to(device)
    model.eval()
    
    return model


def create_test_dataloader(data_dir, test_csv, img_size, batch_size=1):
    """Create test dataloader"""
    print("Creating test dataloader...")
    
    test_dataset = RetinaDatasetTest(
        root_dir=data_dir,
        csv_file=test_csv,
        target_size=(img_size, img_size),
        use_cropped=True,
        cropped_masks_dir='datasets/REFUGE_cropped_masks_test',  # ← Korrekter Test-Ordner
        use_clahe=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader, test_dataset  # ← Return dataset too for image names


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    print("\nEvaluating model...")
    
    results = {
        'per_sample': [],
        'overall': {}
    }
    
    all_dice_scores = []
    all_disc_dice = []
    all_cup_dice = []
    all_cdr_maes = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)
            
            # Get predictions (threshold at 0.5)
            preds = (probs > 0.5).float()
            
            # Calculate metrics for each sample in batch
            for i in range(images.size(0)):
                pred_disc = preds[i, 0]
                pred_cup = preds[i, 1]
                true_disc = masks[i, 0]
                true_cup = masks[i, 1]
                
                # Dice scores
                disc_dice = dice_coefficient(pred_disc, true_disc)
                if isinstance(disc_dice, torch.Tensor):
                    disc_dice = disc_dice.item()
                cup_dice = dice_coefficient(pred_cup, true_cup)
                if isinstance(cup_dice, torch.Tensor):
                    cup_dice = cup_dice.item()
                avg_dice = (disc_dice + cup_dice) / 2.0
                
                # CDR calculation
                pred_cdr = calculate_cdr(pred_disc, pred_cup)
                true_cdr = calculate_cdr(true_disc, true_cup)
                cdr_mae = abs(pred_cdr - true_cdr)
                
                # Store results
                sample_result = {
                    'sample_idx': batch_idx * test_loader.batch_size + i,
                    'disc_dice': disc_dice,
                    'cup_dice': cup_dice,
                    'avg_dice': avg_dice,
                    'pred_cdr': pred_cdr,
                    'true_cdr': true_cdr,
                    'cdr_mae': cdr_mae
                }
                results['per_sample'].append(sample_result)
                
                # Accumulate for overall stats
                all_dice_scores.append(avg_dice)
                all_disc_dice.append(disc_dice)
                all_cup_dice.append(cup_dice)
                all_cdr_maes.append(cdr_mae)
    
    # Calculate overall statistics
    results['overall'] = {
        'avg_dice': float(np.mean(all_dice_scores)),
        'std_dice': float(np.std(all_dice_scores)),
        'avg_disc_dice': float(np.mean(all_disc_dice)),
        'avg_cup_dice': float(np.mean(all_cup_dice)),
        'avg_cdr_mae': float(np.mean(all_cdr_maes)),
        'std_cdr_mae': float(np.std(all_cdr_maes)),
        'num_samples': len(all_dice_scores)
    }
    
    return results


def create_overlay(img_np, disc_mask, cup_mask, alpha=0.4):
    """Create overlay with green disc and orange cup"""
    overlay = img_np.copy()
    
    # Create color masks
    green_mask = np.zeros_like(img_np)
    green_mask[disc_mask > 0.5] = [0, 1, 0]  # Green for disc
    
    orange_mask = np.zeros_like(img_np)
    orange_mask[cup_mask > 0.5] = [1, 0.5, 0]  # Orange for cup
    
    # Combine masks (cup overwrites disc where they overlap)
    combined_mask = green_mask.copy()
    combined_mask[cup_mask > 0.5] = orange_mask[cup_mask > 0.5]
    
    # Blend with original image
    overlay = img_np * (1 - alpha) + combined_mask * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def create_visualizations(model, test_loader, test_dataset, device, output_dir, num_samples=10):
    """Create visualization images"""
    print(f"\nCreating visualizations for {num_samples} samples...")
    
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    samples_saved = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            if samples_saved >= num_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Process each sample in batch
            for i in range(images.size(0)):
                if samples_saved >= num_samples:
                    break
                
                # Get sample index and image name
                sample_idx = batch_idx * test_loader.batch_size + i
                row = test_dataset.df.iloc[sample_idx]
                
                # Extract image name from the row (just the filename without path/extension)
                if 'ImgName' in row:
                    img_path = row['ImgName']
                elif 'imgName' in row:
                    img_path = row['imgName']
                else:
                    # Try to extract from other columns
                    img_path = str(sample_idx).zfill(4)
                
                # Extract just the filename without path and extension
                import os
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Extract the test number from the filename (e.g., T0227 -> 227)
                import re
                test_number_match = re.search(r'T(\d+)', img_name)
                test_number = int(test_number_match.group(1)) if test_number_match else sample_idx + 1
                
                # Get sample data
                img = images[i].cpu()
                pred_disc = preds[i, 0].cpu().numpy()
                pred_cup = preds[i, 1].cpu().numpy()
                true_disc = masks[i, 0].cpu().numpy()
                true_cup = masks[i, 1].cpu().numpy()
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img.permute(1, 2, 0).numpy()
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                
                # Calculate CDR
                pred_cdr = calculate_cdr(torch.from_numpy(pred_disc), torch.from_numpy(pred_cup))
                true_cdr = calculate_cdr(torch.from_numpy(true_disc), torch.from_numpy(true_cup))
                
                # Calculate Dice
                disc_dice = dice_coefficient(torch.from_numpy(pred_disc), torch.from_numpy(true_disc))
                if isinstance(disc_dice, torch.Tensor):
                    disc_dice = disc_dice.item()
                cup_dice = dice_coefficient(torch.from_numpy(pred_cup), torch.from_numpy(true_cup))
                if isinstance(cup_dice, torch.Tensor):
                    cup_dice = cup_dice.item()
                avg_dice = (disc_dice + cup_dice) / 2.0
                
                # Create overlays
                gt_overlay = create_overlay(img_np, true_disc, true_cup)
                pred_overlay = create_overlay(img_np, pred_disc, pred_cup)
                
                # Create visualization (2 rows x 4 columns)
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                
                # Row 1: Ground Truth
                axes[0, 0].imshow(img_np)
                axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(true_disc, cmap='gray')
                axes[0, 1].set_title('GT Disc', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(true_cup, cmap='gray')
                axes[0, 2].set_title('GT Cup', fontsize=12, fontweight='bold')
                axes[0, 2].axis('off')
                
                axes[0, 3].imshow(gt_overlay)
                axes[0, 3].set_title(f'GT Overlay (CDR={true_cdr:.3f})', fontsize=12, fontweight='bold')
                axes[0, 3].axis('off')
                
                # Row 2: Predictions
                axes[1, 0].imshow(img_np)
                axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(pred_disc, cmap='gray')
                axes[1, 1].set_title(f'Pred Disc (Dice={disc_dice:.3f})', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(pred_cup, cmap='gray')
                axes[1, 2].set_title(f'Pred Cup (Dice={cup_dice:.3f})', fontsize=12, fontweight='bold')
                axes[1, 2].axis('off')
                
                axes[1, 3].imshow(pred_overlay)
                axes[1, 3].set_title(f'Pred Overlay (CDR={pred_cdr:.3f})', fontsize=12, fontweight='bold')
                axes[1, 3].axis('off')
                
                # Add overall title with metrics and image name
                fig.suptitle(
                    f'Test Sample {test_number} - Avg Dice: {avg_dice:.3f} | Image: {img_name}',
                    fontsize=16, fontweight='bold', y=0.98
                )
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                # Save figure
                save_path = vis_dir / f'test_sample_{sample_idx:04d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                samples_saved += 1
                
    print(f"Saved {samples_saved} visualizations to {vis_dir}")


def save_results(results, output_dir):
    """Save results to JSON file"""
    output_path = Path(output_dir) / 'test_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_path}")


def print_results(results):
    """Print test results"""
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    overall = results['overall']
    
    print(f"\nNumber of test samples: {overall['num_samples']}")
    print(f"\nDice Score:")
    print(f"  Average: {overall['avg_dice']:.4f} ± {overall['std_dice']:.4f}")
    print(f"  Disc:    {overall['avg_disc_dice']:.4f}")
    print(f"  Cup:     {overall['avg_cup_dice']:.4f}")
    
    print(f"\nCDR (Cup-to-Disc Ratio):")
    print(f"  MAE:     {overall['avg_cdr_mae']:.4f} ± {overall['std_cdr_mae']:.4f}")
    
    print("\n" + "="*60)


def main():
    args = get_args()
    
    print("="*60)
    print("EE-TransUNet Test Script")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, args.model_name, args.img_size, args.device)
    
    # Create test dataloader
    test_loader, test_dataset = create_test_dataloader(
        args.data_dir,
        args.test_csv,
        args.img_size,
        batch_size=1
    )
    
    # Evaluate model
    results = evaluate_model(model, test_loader, args.device)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.output_dir)
    
    # Create visualizations
    if args.save_visualizations:
        create_visualizations(
            model,
            test_loader,
            test_dataset,  # ← Pass dataset for image names
            args.device,
            args.output_dir,
            num_samples=args.num_visualizations
        )
    
    print(f"\n✅ Testing complete! Results saved to: {args.output_dir}")
    
    print(f"\n✅ Testing complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
