"""
Generate cropped BMP masks by finding crop region using template matching
"""
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import pandas as pd


def find_disc_bounding_box(disc_mask: np.ndarray, cup_mask: np.ndarray, 
                           padding: int = 50) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the bounding box of the optic disc region in the BMP mask.
    
    Args:
        disc_mask: Full resolution disc mask (H, W) with 128=disc, 255=background
        cup_mask: Full resolution cup mask (H, W) with 0=cup, 255=background
        padding: Extra pixels to add around the bounding box
        
    Returns:
        Tuple of (x, y, w, h) indicating the crop region, or None if no disc found
    """
    # Find disc pixels (value == 128)
    disc_pixels = (disc_mask == 128)
    
    if not disc_pixels.any():
        print("  No disc pixels found in mask!")
        return None
    
    # Get coordinates of disc pixels
    rows, cols = np.where(disc_pixels)
    
    # Calculate bounding box
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    # Add padding
    h, w = disc_mask.shape
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    
    # Calculate width and height
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    
    # Make it roughly square by taking the larger dimension
    size = max(crop_w, crop_h)
    
    # Center the square crop on the disc
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min = max(0, center_x - size // 2)
    y_min = max(0, center_y - size // 2)
    x_max = min(w, x_min + size)
    y_max = min(h, y_min + size)
    
    # Adjust if we hit boundaries
    if x_max - x_min < size:
        x_min = max(0, x_max - size)
    if y_max - y_min < size:
        y_min = max(0, y_max - size)
    
    return (x_min, y_min, crop_w, crop_h)


def crop_mask(mask: np.ndarray, region: Tuple[int, int, int, int], 
              resize_to: Optional[int] = None) -> np.ndarray:
    """
    Extract the crop region from a mask and optionally resize.
    
    Args:
        mask: Full resolution mask (H, W)
        region: Tuple of (x, y, w, h) indicating the crop region
        resize_to: Optional size to resize the cropped mask to (for matching upscaled crops)
        
    Returns:
        Cropped mask (h, w) or (resize_to, resize_to) if resized
    """
    x, y, w, h = region
    cropped = mask[y:y+h, x:x+w]
    
    if resize_to is not None:
        # Resize using nearest neighbor to preserve binary values
        cropped = cv2.resize(cropped, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
    
    return cropped


def process_sample(sample_folder: str, base_dir: str, output_dir: str, 
                   save_visualization: bool = False, padding: int = 50) -> bool:
    """
    Process a single sample to generate cropped masks by finding the disc bounding box.
    
    Instead of trying to match the existing cropped image, we:
    1. Find the bounding box of the disc in the BMP mask
    2. Crop both the full image and BMP masks using this bounding box
    3. Save the perfectly aligned cropped versions
    
    Args:
        sample_folder: Folder name (e.g., '0853')
        base_dir: Base directory containing Training-400/
        output_dir: Directory to save cropped masks
        save_visualization: Whether to save visualization images
        padding: Extra pixels around the disc bounding box (default: 50)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Processing {sample_folder}...")
    
    try:
        # Construct paths
        sample_dir = os.path.join(base_dir, 'Training-400', sample_folder)
        full_img_path = os.path.join(sample_dir, f'{sample_folder}.jpg')
        disc_mask_path = os.path.join(sample_dir, f'{sample_folder}_disc.bmp')
        cup_mask_path = os.path.join(sample_dir, f'{sample_folder}_cup.bmp')
        
        # Load images
        full_img = cv2.imread(full_img_path)
        disc_mask = np.array(Image.open(disc_mask_path))
        cup_mask = np.array(Image.open(cup_mask_path))
        
        # Find bounding box based on disc mask
        bbox = find_disc_bounding_box(disc_mask, cup_mask, padding=padding)
        
        if bbox is None:
            return False
        
        x, y, w, h = bbox
        
        print(f"  Disc bounding box: ({x}, {y}, {w}, {h})")
        
        # Crop image and masks
        img_cropped = full_img[y:y+h, x:x+w]
        disc_cropped = disc_mask[y:y+h, x:x+w]
        cup_cropped = cup_mask[y:y+h, x:x+w]
        
        # Create output directory for this sample
        output_sample_dir = os.path.join(output_dir, sample_folder)
        os.makedirs(output_sample_dir, exist_ok=True)
        
        # Save cropped image and masks
        img_out_path = os.path.join(output_sample_dir, f'{sample_folder}_cropped.jpg')
        disc_out_path = os.path.join(output_sample_dir, f'{sample_folder}_disc_cropped.bmp')
        cup_out_path = os.path.join(output_sample_dir, f'{sample_folder}_cup_cropped.bmp')
        
        cv2.imwrite(img_out_path, img_cropped)
        Image.fromarray(disc_cropped).save(disc_out_path)
        Image.fromarray(cup_cropped).save(cup_out_path)
        
        print(f"  Saved cropped data: {h}×{w}")
        print(f"  Disc coverage: {(disc_cropped == 128).sum() / (h*w) * 100:.1f}%")
        print(f"  Cup coverage: {(cup_cropped == 0).sum() / (h*w) * 100:.1f}%")
        
        if save_visualization:
            # Save visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Images
            axes[0, 0].imshow(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB))
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            axes[0, 0].add_patch(rect)
            axes[0, 0].set_title(f'Full Image with Bounding Box')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f'Cropped Image ({h}×{w})')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(disc_mask, cmap='gray')
            rect2 = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            axes[0, 2].add_patch(rect2)
            axes[0, 2].set_title('Full Disc Mask')
            axes[0, 2].axis('off')
            
            # Row 2: Cropped masks
            axes[1, 0].imshow(disc_cropped, cmap='gray')
            axes[1, 0].set_title(f'Cropped Disc Mask ({h}×{w})')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(cup_cropped, cmap='gray')
            axes[1, 1].set_title(f'Cropped Cup Mask ({h}×{w})')
            axes[1, 1].axis('off')
            
            # Overlay visualization
            overlay = cv2.cvtColor(img_cropped.copy(), cv2.COLOR_BGR2RGB)
            disc_overlay = np.zeros_like(overlay)
            disc_overlay[disc_cropped == 128] = [255, 0, 0]  # Red for disc
            disc_overlay[cup_cropped == 0] = [0, 255, 0]     # Green for cup
            overlay = cv2.addWeighted(overlay, 0.7, disc_overlay, 0.3, 0)
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Cropped with Overlay')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Sample {sample_folder} - 100% Accurate Alignment')
            plt.tight_layout()
            viz_path = os.path.join(output_sample_dir, f'{sample_folder}_visualization.png')
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  Saved visualization to {viz_path}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {sample_folder}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_cropped_masks(root_dir='datasets/REFUGE', splits=['Training-400', 'Validation-400']):
    """
    Generate cropped masks for all samples in the dataset
    
    Args:
        root_dir: REFUGE dataset root directory
        splits: List of splits to process
    """
    results = []
    
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            continue
        
        # Get all sample folders
        sample_ids = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        sample_ids.sort()
        
        print(f"\nProcessing {split}: {len(sample_ids)} samples")
        
        # Process each sample
        for sample_id in tqdm(sample_ids, desc=f"Processing {split}"):
            result = process_sample(sample_id, root_dir, split, save_cropped_masks=True)
            results.append(result)
            
            if not result['success']:
                print(f"  ✗ {sample_id}: {result['error']}")
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(results)} successful")
    print(f"{'='*60}")
    
    # Check confidence distribution
    confidences = [r['confidence'] for r in results if r['success']]
    if confidences:
        print(f"Template matching confidence:")
        print(f"  Min: {min(confidences):.3f}")
        print(f"  Max: {max(confidences):.3f}")
        print(f"  Mean: {np.mean(confidences):.3f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate cropped BMP masks for REFUGE dataset')
    parser.add_argument('--root_dir', type=str, default='datasets/REFUGE',
                       help='REFUGE dataset root directory')
    parser.add_argument('--output_dir', type=str, default='datasets/REFUGE_cropped_masks',
                       help='Output directory for cropped masks')
    parser.add_argument('--test_sample', type=str, default=None,
                       help='Test on single sample (e.g., "0853")')
    parser.add_argument('--viz', action='store_true',
                       help='Save visualization images')
    
    args = parser.parse_args()
    
    if args.test_sample:
        # Test on single sample
        print(f"Testing on sample: {args.test_sample}\n")
        success = process_sample(
            args.test_sample, 
            args.root_dir, 
            args.output_dir,
            save_visualization=args.viz
        )
        print(f"\nResult: {'✓ Success' if success else '✗ Failed'}")
    else:
        # Process all samples in Training-400
        print(f"Processing all Training-400 samples...")
        print(f"Root dir: {args.root_dir}")
        print(f"Output dir: {args.output_dir}\n")
        
        train_dir = os.path.join(args.root_dir, 'Training-400')
        if not os.path.exists(train_dir):
            print(f"Error: Training directory not found: {train_dir}")
            exit(1)
        
        # Get all sample folders
        sample_folders = sorted([d for d in os.listdir(train_dir) 
                                if os.path.isdir(os.path.join(train_dir, d))])
        
        print(f"Found {len(sample_folders)} samples\n")
        
        # Process all samples
        success_count = 0
        failed_samples = []
        
        for i, sample_folder in enumerate(sample_folders, 1):
            print(f"[{i}/{len(sample_folders)}] ", end='')
            success = process_sample(
                sample_folder,
                args.root_dir,
                args.output_dir,
                save_visualization=args.viz
            )
            
            if success:
                success_count += 1
            else:
                failed_samples.append(sample_folder)
            print()  # Empty line between samples
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Summary: {success_count}/{len(sample_folders)} successful")
        if failed_samples:
            print(f"\nFailed samples: {', '.join(failed_samples)}")
        print(f"{'='*60}")
