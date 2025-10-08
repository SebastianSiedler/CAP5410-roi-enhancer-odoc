"""
Inference script for trained U-Net model
Load a trained model and make predictions on new images
"""
import os
import argparse
from pathlib import Path
import json

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from src.models.unet import UNet
from src.utils.metrics import calculate_cdr
from src.data_loader.clahe_preprocessing import CLAHEPreprocessor


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to saved checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
        config: Training configuration
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False for backward compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration
    if 'args' in checkpoint:
        config = checkpoint['args']
    else:
        # Default configuration
        config = {
            'base_features': 64,
            'bilinear': False
        }
    
    # Create model
    model = UNet(
        n_channels=3,
        n_classes=2,
        bilinear=config.get('bilinear', False),
        base_features=config.get('base_features', 64)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  Validation Dice: {metrics.get('dice_mean', 'N/A'):.4f}")
        print(f"  Validation CDR MAE: {metrics.get('cdr_mae', 'N/A'):.4f}")
    
    return model, config


def preprocess_image(image_path, target_size=(512, 512), use_clahe=True, clahe_mode='LAB'):
    """
    Load and preprocess image for inference
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing
        use_clahe: Whether to apply CLAHE preprocessing
        clahe_mode: CLAHE mode ('LAB', 'RGB', 'HSV')
    
    Returns:
        tensor: Preprocessed image tensor (1, 3, H, W)
        original: Original PIL image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Store original for visualization
    original = image.copy()
    
    # Apply CLAHE if requested
    if use_clahe:
        clahe_preprocessor = CLAHEPreprocessor(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            apply_to=clahe_mode
        )
        # Convert PIL to numpy for CLAHE
        image_np = np.array(image)
        enhanced_np = clahe_preprocessor(image_np)
        image = Image.fromarray(enhanced_np)
    
    # Preprocessing pipeline (same as training)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor, original


def predict(model, image_tensor, device='cuda', threshold=0.5):
    """
    Make prediction on image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        device: Device to run on
        threshold: Threshold for binary segmentation
    
    Returns:
        disc_mask: Binary mask for optic disc (H, W)
        cup_mask: Binary mask for optic cup (H, W)
        cdr: Calculated cup-to-disc ratio
        disc_probs: Probability map for disc (H, W)
        cup_probs: Probability map for cup (H, W)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        
        # Apply threshold to get binary masks
        masks = (probs > threshold).float()
        
        # Extract disc and cup masks
        disc_mask = masks[0, 0].cpu().numpy()  # Channel 0: Optic Disc
        cup_mask = masks[0, 1].cpu().numpy()   # Channel 1: Optic Cup
        
        # Extract probability maps
        disc_probs = probs[0, 0].cpu().numpy()
        cup_probs = probs[0, 1].cpu().numpy()
        
        # Calculate CDR (keep as tensors for the function)
        disc_tensor = masks[0, 0]
        cup_tensor = masks[0, 1]
        cdr = calculate_cdr(cup_tensor, disc_tensor)
    
    return disc_mask, cup_mask, cdr, disc_probs, cup_probs


def visualize_prediction(original_image, disc_mask, cup_mask, cdr, disc_probs, cup_probs, save_path=None):
    """
    Visualize prediction results
    
    Args:
        original_image: Original PIL image
        disc_mask: Predicted optic disc mask
        cup_mask: Predicted optic cup mask
        cdr: Calculated cup-to-disc ratio
        disc_probs: Probability map for disc
        cup_probs: Probability map for cup
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Optic Disc probabilities
    axes[0, 1].imshow(disc_probs, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Optic Disc Probability')
    axes[0, 1].axis('off')
    axes[0, 1].figure.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
    
    # Optic Cup probabilities
    axes[0, 2].imshow(cup_probs, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Optic Cup Probability')
    axes[0, 2].axis('off')
    axes[0, 2].figure.colorbar(axes[0, 2].images[0], ax=axes[0, 2], fraction=0.046)
    
    # Disc mask
    axes[1, 0].imshow(disc_mask, cmap='gray')
    axes[1, 0].set_title(f'Disc Mask (Area: {disc_mask.sum():.0f})')
    axes[1, 0].axis('off')
    
    # Cup mask
    axes[1, 1].imshow(cup_mask, cmap='gray')
    axes[1, 1].set_title(f'Cup Mask (Area: {cup_mask.sum():.0f})')
    axes[1, 1].axis('off')
    
    # Overlay
    # Resize original to match mask size
    original_resized = original_image.resize((disc_mask.shape[1], disc_mask.shape[0]))
    overlay = np.array(original_resized).copy().astype(np.float32) / 255.0
    
    # Create colored overlays (disc in red, cup in green)
    overlay_colored = overlay.copy()
    overlay_colored[disc_mask > 0.5, 0] += 0.5  # Red channel for disc
    overlay_colored[cup_mask > 0.5, 1] += 0.5   # Green channel for cup
    overlay_colored = np.clip(overlay_colored, 0, 1)
    
    axes[1, 2].imshow(overlay_colored)
    axes[1, 2].set_title(f'Overlay (CDR: {cdr:.3f})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Inference with trained U-Net model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--use_clahe', action='store_true', default=True,
                       help='Apply CLAHE preprocessing (default: True)')
    parser.add_argument('--no_clahe', action='store_false', dest='use_clahe',
                       help='Disable CLAHE preprocessing')
    parser.add_argument('--clahe_mode', type=str, default='LAB',
                       choices=['LAB', 'RGB', 'HSV'],
                       help='CLAHE mode (default: LAB)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load and preprocess image
    print(f"\nProcessing image: {args.image}")
    print(f"  CLAHE: {'Enabled' if args.use_clahe else 'Disabled'}")
    if args.use_clahe:
        print(f"  CLAHE mode: {args.clahe_mode}")
    image_tensor, original_image = preprocess_image(args.image, use_clahe=args.use_clahe, clahe_mode=args.clahe_mode)
    
    # Make prediction
    print("Running inference...")
    disc_mask, cup_mask, cdr, disc_probs, cup_probs = predict(model, image_tensor, device, args.threshold)
    
    # Print results
    print(f"\nResults:")
    print(f"  Optic Disc area: {np.sum(disc_mask > 0.5)} pixels")
    print(f"  Optic Cup area: {np.sum(cup_mask > 0.5)} pixels")
    print(f"  Cup-to-Disc Ratio (CDR): {cdr:.4f}")
    print(f"  Disc probability range: {disc_probs.min():.3f} - {disc_probs.max():.3f}")
    print(f"  Cup probability range: {cup_probs.min():.3f} - {cup_probs.max():.3f}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_prediction(original_image, disc_mask, cup_mask, cdr, disc_probs, cup_probs, args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
