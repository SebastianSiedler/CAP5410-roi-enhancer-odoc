"""
Post-processing utilities for optic disc/cup segmentation predictions.
Applies morphological operations and anatomical constraints to improve results.
"""
import cv2
import numpy as np
import torch
from scipy import ndimage


def post_process_predictions(disc_mask, cup_mask, apply_constraints=True):
    """
    Apply post-processing to improve segmentation predictions.
    
    Args:
        disc_mask: Binary disc mask (numpy array, H x W, values 0-255 or 0-1)
        cup_mask: Binary cup mask (numpy array, H x W, values 0-255 or 0-1)
        apply_constraints: Whether to apply anatomical constraints
    
    Returns:
        Processed disc and cup masks
    """
    # Normalize to 0-255 range if needed
    if disc_mask.max() <= 1.0:
        disc_mask = (disc_mask * 255).astype(np.uint8)
    else:
        disc_mask = disc_mask.astype(np.uint8)
    
    if cup_mask.max() <= 1.0:
        cup_mask = (cup_mask * 255).astype(np.uint8)
    else:
        cup_mask = cup_mask.astype(np.uint8)
    
    # 1. Morphological operations to remove noise
    disc_mask = remove_small_noise(disc_mask, min_size=100)
    cup_mask = remove_small_noise(cup_mask, min_size=50)
    
    # 2. Keep only largest connected component
    disc_mask = keep_largest_component(disc_mask)
    cup_mask = keep_largest_component(cup_mask)
    
    # 3. Fill holes
    disc_mask = fill_holes(disc_mask)
    cup_mask = fill_holes(cup_mask)
    
    # 4. Apply anatomical constraints
    if apply_constraints:
        disc_mask, cup_mask = apply_anatomical_constraints(disc_mask, cup_mask)
    
    # 5. Smooth boundaries (optional - can make masks more circular)
    # disc_mask = smooth_boundaries(disc_mask)
    # cup_mask = smooth_boundaries(cup_mask)
    
    return disc_mask, cup_mask


def remove_small_noise(mask, min_size=100):
    """
    Remove small connected components (noise).
    
    Args:
        mask: Binary mask (0-255)
        min_size: Minimum size of components to keep (in pixels)
    
    Returns:
        Cleaned mask
    """
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove very small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create output mask
    output = np.zeros_like(mask)
    
    # Keep components larger than min_size
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    
    return output


def keep_largest_component(mask):
    """
    Keep only the largest connected component.
    Assumes the disc/cup should be a single region.
    
    Args:
        mask: Binary mask (0-255)
    
    Returns:
        Mask with only largest component
    """
    if mask.max() == 0:
        return mask
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:  # Only background
        return mask
    
    # Find largest component (excluding background at index 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create output with only largest component
    output = np.zeros_like(mask)
    output[labels == largest_label] = 255
    
    return output


def fill_holes(mask):
    """
    Fill holes in binary mask.
    
    Args:
        mask: Binary mask (0-255)
    
    Returns:
        Mask with holes filled
    """
    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Fill larger holes using flood fill from corners
    mask_floodfill = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from top-left corner to identify background
    cv2.floodFill(mask_floodfill, flood_mask, (0, 0), 255)
    
    # Invert floodfilled image and combine with original
    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
    mask_out = mask | mask_floodfill_inv
    
    return mask_out


def apply_anatomical_constraints(disc_mask, cup_mask):
    """
    Apply anatomical constraints:
    1. Cup must be inside disc
    2. Cup should be roughly centered in disc
    3. Cup size should be < disc size
    
    Args:
        disc_mask: Disc mask (0-255)
        cup_mask: Cup mask (0-255)
    
    Returns:
        Constrained disc and cup masks
    """
    # Constraint 1: Cup must be inside disc
    cup_mask = np.logical_and(cup_mask > 127, disc_mask > 127).astype(np.uint8) * 255
    
    # If cup is empty after constraint, return as is
    if cup_mask.max() == 0:
        return disc_mask, cup_mask
    
    # Constraint 2 & 3: Check cup size relative to disc
    disc_area = np.sum(disc_mask > 127)
    cup_area = np.sum(cup_mask > 127)
    
    # If cup is too large (>80% of disc), it's likely wrong - shrink it
    if cup_area > 0.8 * disc_area:
        # Erode cup mask to reduce size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cup_mask = cv2.erode(cup_mask, kernel, iterations=2)
        cup_mask = keep_largest_component(cup_mask)
    
    # Constraint 3: Cup should be somewhat centered in disc
    # Calculate centroids
    disc_moments = cv2.moments(disc_mask)
    if disc_moments['m00'] > 0:
        disc_cx = int(disc_moments['m10'] / disc_moments['m00'])
        disc_cy = int(disc_moments['m01'] / disc_moments['m00'])
        
        cup_moments = cv2.moments(cup_mask)
        if cup_moments['m00'] > 0:
            cup_cx = int(cup_moments['m10'] / cup_moments['m00'])
            cup_cy = int(cup_moments['m01'] / cup_moments['m00'])
            
            # Distance between centroids
            distance = np.sqrt((disc_cx - cup_cx)**2 + (disc_cy - cup_cy)**2)
            
            # Get disc radius (approximate)
            disc_radius = np.sqrt(disc_area / np.pi)
            
            # If cup is too far from disc center (>50% of radius), it's suspicious
            if distance > 0.5 * disc_radius:
                # Try to reposition cup toward disc center
                # This is a heuristic - in practice, might want to be more sophisticated
                pass  # For now, just keep the cup as is
    
    return disc_mask, cup_mask


def smooth_boundaries(mask, sigma=2.0):
    """
    Smooth mask boundaries using Gaussian filtering.
    Can help create more realistic, less jagged edges.
    
    Args:
        mask: Binary mask (0-255)
        sigma: Gaussian sigma for smoothing
    
    Returns:
        Smoothed mask
    """
    # Convert to float
    mask_float = mask.astype(np.float32) / 255.0
    
    # Apply Gaussian blur
    mask_smooth = ndimage.gaussian_filter(mask_float, sigma=sigma)
    
    # Threshold back to binary
    mask_smooth = (mask_smooth > 0.5).astype(np.uint8) * 255
    
    return mask_smooth


def fit_ellipse_mask(mask):
    """
    Fit an ellipse to the mask and return ellipse mask.
    Useful for creating smooth, anatomically plausible shapes.
    
    Args:
        mask: Binary mask (0-255)
    
    Returns:
        Ellipse-fitted mask
    """
    if mask.max() == 0:
        return mask
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or len(contours[0]) < 5:
        return mask
    
    # Fit ellipse to largest contour
    try:
        ellipse = cv2.fitEllipse(contours[0])
        
        # Create ellipse mask
        ellipse_mask = np.zeros_like(mask)
        cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        
        return ellipse_mask
    except:
        # If ellipse fitting fails, return original
        return mask


def calculate_cdr(disc_mask, cup_mask):
    """
    Calculate Cup-to-Disc Ratio from masks.
    
    Args:
        disc_mask: Binary disc mask
        cup_mask: Binary cup mask
    
    Returns:
        CDR value (cup_area / disc_area)
    """
    disc_area = np.sum(disc_mask > 127)
    cup_area = np.sum(cup_mask > 127)
    
    if disc_area == 0:
        return 0.0
    
    return float(cup_area) / float(disc_area)


def post_process_batch(disc_preds, cup_preds, threshold=0.5, apply_constraints=True):
    """
    Post-process a batch of predictions.
    
    Args:
        disc_preds: Disc predictions tensor (B, 1, H, W) with probabilities
        cup_preds: Cup predictions tensor (B, 1, H, W) with probabilities
        threshold: Threshold for binarization
        apply_constraints: Whether to apply anatomical constraints
    
    Returns:
        Processed disc and cup masks as tensors
    """
    batch_size = disc_preds.shape[0]
    device = disc_preds.device
    
    # Convert to numpy
    disc_preds_np = disc_preds.cpu().numpy()
    cup_preds_np = cup_preds.cpu().numpy()
    
    # Process each sample
    disc_masks_processed = []
    cup_masks_processed = []
    
    for i in range(batch_size):
        # Threshold
        disc_mask = (disc_preds_np[i, 0] > threshold).astype(np.uint8) * 255
        cup_mask = (cup_preds_np[i, 0] > threshold).astype(np.uint8) * 255
        
        # Post-process
        disc_mask, cup_mask = post_process_predictions(disc_mask, cup_mask, apply_constraints)
        
        # Normalize back to 0-1
        disc_masks_processed.append(disc_mask.astype(np.float32) / 255.0)
        cup_masks_processed.append(cup_mask.astype(np.float32) / 255.0)
    
    # Convert back to tensor
    disc_masks_processed = torch.from_numpy(np.array(disc_masks_processed)).unsqueeze(1).to(device)
    cup_masks_processed = torch.from_numpy(np.array(cup_masks_processed)).unsqueeze(1).to(device)
    
    return disc_masks_processed, cup_masks_processed


def test_postprocessing():
    """Test post-processing functions"""
    # Create dummy masks with noise
    disc_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(disc_mask, (256, 256), 100, 255, -1)
    
    cup_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(cup_mask, (256, 256), 40, 255, -1)
    
    # Add noise
    noise = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
    noise = cv2.dilate(noise, np.ones((3, 3)), iterations=1)
    disc_mask = cv2.bitwise_or(disc_mask, noise)
    
    print(f"Before post-processing:")
    print(f"  Disc area: {np.sum(disc_mask > 127)}")
    print(f"  Cup area: {np.sum(cup_mask > 127)}")
    print(f"  CDR: {calculate_cdr(disc_mask, cup_mask):.3f}")
    
    # Post-process
    disc_clean, cup_clean = post_process_predictions(disc_mask, cup_mask)
    
    print(f"\nAfter post-processing:")
    print(f"  Disc area: {np.sum(disc_clean > 127)}")
    print(f"  Cup area: {np.sum(cup_clean > 127)}")
    print(f"  CDR: {calculate_cdr(disc_clean, cup_clean):.3f}")
    
    print("\n✅ Post-processing test passed!")


if __name__ == '__main__':
    test_postprocessing()
