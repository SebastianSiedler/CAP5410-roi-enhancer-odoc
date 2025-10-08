"""
Quick visualization of augmentation impact on training data.
Run this to see what strong augmentation does to a sample image.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_loader.strong_augmentation import StrongAugmentation
from data_loader.clahe_preprocessing import CLAHEPreprocessor


def visualize_augmentation():
    """Show before/after examples of strong augmentation."""
    
    # Load sample image
    sample_path = project_root / 'datasets' / 'REFUGE' / 'Training-400' / 'Images' / 'V0001.jpg'
    
    if not sample_path.exists():
        print(f"❌ Sample image not found: {sample_path}")
        print("Please make sure REFUGE dataset is downloaded.")
        return
    
    # Load and preprocess
    img = np.array(Image.open(sample_path))
    
    # Apply CLAHE
    clahe = CLAHEPreprocessor(clip_limit=2.0, mode='LAB')
    img_clahe = clahe.apply(img)
    
    # Convert to tensor
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(to_tensor(img_clahe))
    
    # Create dummy mask
    mask_tensor = torch.rand(2, 512, 512)
    
    # Create augmentor
    augmentor = StrongAugmentation(p=1.0)  # Always augment for demo
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Strong Augmentation Examples - Experiment 04', fontsize=18, fontweight='bold')
    
    # Denormalize function
    def denormalize(tensor):
        img_np = tensor.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return np.clip(img_np, 0, 1)
    
    # Original
    axes[0, 0].imshow(denormalize(img_tensor))
    axes[0, 0].set_title('Original\n(CLAHE preprocessed)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.1, 'Baseline', ha='center', transform=axes[0, 0].transAxes, fontsize=12)
    
    # 7 augmented versions
    aug_labels = [
        'Rotation + Flip',
        'Color Jitter',
        'Affine Transform',
        'Blur',
        'All Combined',
        'Different Seed 1',
        'Different Seed 2'
    ]
    
    positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    
    for idx, (row, col) in enumerate(positions):
        aug_img, aug_mask = augmentor(img_tensor.clone(), mask_tensor.clone())
        
        axes[row, col].imshow(denormalize(aug_img))
        axes[row, col].set_title(f'Augmented #{idx+1}', fontsize=14)
        axes[row, col].axis('off')
        axes[row, col].text(0.5, -0.1, aug_labels[idx], ha='center', 
                           transform=axes[row, col].transAxes, fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_path = project_root / 'experiments' / '04_resnet34_augmented' / 'augmentation_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Augmentation demo saved to: {output_path}")
    
    plt.show()
    
    # Print info
    print("\n" + "="*60)
    print("Strong Augmentation Pipeline (p=0.8)")
    print("="*60)
    print("\nGeometric Transforms (p=0.8):")
    print("  ✓ RandomHorizontalFlip(p=0.5)")
    print("  ✓ RandomVerticalFlip(p=0.5)")
    print("  ✓ RandomRotation(degrees=30)")
    print("  ✓ RandomAffine(translate=0.1, scale=(0.9, 1.1), shear=10)")
    print("\nColor Transforms (p=0.5):")
    print("  ✓ ColorJitter(brightness=0.2, contrast=0.2)")
    print("  ✓ ColorJitter(saturation=0.2, hue=0.05)")
    print("\nBlur/Sharpness Transforms (p=0.3):")
    print("  ✓ GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))")
    print("  ✓ RandomAdjustSharpness(factor=2, p=0.5)")
    print("\nExpected Impact:")
    print("  • Effective dataset size: 400 → ~2000 samples")
    print("  • Overfitting gap: 10-15% → 2-5%")
    print("  • Validation Dice: 66-70% → 73-78%")
    print("="*60)


if __name__ == '__main__':
    visualize_augmentation()
