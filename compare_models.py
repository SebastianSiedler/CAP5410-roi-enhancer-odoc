#!/usr/bin/env python3
"""
Quick comparison of model architectures.
Shows parameter counts and expected performance.
"""

import sys
sys.path.append('src')

from models.unet import UNet
from models.resnet_unet import ResNetUNet
import torch

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    print("="*80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*80)
    
    models = [
        ("Small U-Net (32 features)", 
         lambda: UNet(n_channels=3, n_classes=2, base_features=32, bilinear=False),
         "76.38%", "67.04%", "7.8M", "140ms", "✅ Current best"),
        
        ("Standard U-Net (64 features)",
         lambda: UNet(n_channels=3, n_classes=2, base_features=64, bilinear=False),
         "69.75%", "53.70%", "31M", "160ms", "⚠️ Baseline (overfitted)"),
        
        ("ResNet18-UNet (pretrained)",
         lambda: ResNetUNet(n_classes=2, backbone='resnet18', pretrained=False),
         "77-79%", "68-71%", "14.4M", "120ms", "🎯 Fast + pretrained"),
        
        ("ResNet34-UNet (pretrained)",
         lambda: ResNetUNet(n_classes=2, backbone='resnet34', pretrained=False),
         "78-80%", "69-72%", "24.5M", "150ms", "🎯 Recommended"),
        
        ("ResNet50-UNet (pretrained)",
         lambda: ResNetUNet(n_classes=2, backbone='resnet50', pretrained=False),
         "78-81%", "70-73%", "72.0M", "200ms", "⚠️ May overfit"),
    ]
    
    print(f"\n{'Model':<35} {'Params':>10} {'Overall':>10} {'Cup':>10} {'Speed':>8} {'Note':>25}")
    print("-"*110)
    
    for name, model_fn, overall, cup, params_expected, speed, note in models:
        try:
            model = model_fn()
            total, trainable = count_parameters(model)
            params_str = f"{total/1e6:.1f}M"
            
            # Check if frozen encoder (ResNet models)
            if hasattr(model, 'freeze_encoder'):
                model.freeze_encoder()
                _, frozen_trainable = count_parameters(model)
                params_str = f"{total/1e6:.1f}M ({frozen_trainable/1e6:.1f}M)"
            
            print(f"{name:<35} {params_str:>10} {overall:>10} {cup:>10} {speed:>8} {note:>25}")
        except Exception as e:
            print(f"{name:<35} {'ERROR':>10} {overall:>10} {cup:>10} {speed:>8} {note:>25}")
    
    print("-"*110)
    print("\nLegend:")
    print("  Params: Total parameters (frozen parameters) for two-stage training")
    print("  Overall: Expected overall Dice score")
    print("  Cup: Expected cup Dice score (hardest metric)")
    print("  Speed: Inference time per image (approximate)")
    print()
    
    print("="*80)
    print("RECOMMENDATIONS FOR 400 TRAINING IMAGES")
    print("="*80)
    print()
    print("1. SAFE CHOICE: ResNet18-UNet")
    print("   - Conservative: 14.4M params (2x current best)")
    print("   - Fast: Similar speed to current")
    print("   - Expected: +1-3% improvement")
    print()
    print("2. RECOMMENDED: ResNet34-UNet ⭐")
    print("   - Balanced: 24.5M params (3x current best)")
    print("   - Proven: Most commonly used ResNet variant")
    print("   - Expected: +2-4% improvement")
    print()
    print("3. AGGRESSIVE: ResNet50-UNet")
    print("   - Large: 72M params (9x current best)")
    print("   - Risk: May overfit on 400 images")
    print("   - Expected: +2-5% improvement (if doesn't overfit)")
    print()
    
    print("="*80)
    print("TRAINING STRATEGY")
    print("="*80)
    print()
    print("Two-Stage Training (reduces overfitting risk):")
    print("  Stage 1 (epochs 0-9):  Freeze encoder, train decoder only")
    print("                          → Only 3.3M trainable parameters")
    print("  Stage 2 (epochs 10-49): Unfreeze encoder, train full model")
    print("                          → Full model trainable")
    print()
    print("This approach:")
    print("  ✅ Prevents destroying pretrained features")
    print("  ✅ Adapts decoder to encoder first")
    print("  ✅ Then fine-tunes encoder for retinal images")
    print()
    
    print("="*80)
    print("QUICK START")
    print("="*80)
    print()
    print("Run training:")
    print("  ./train_resnet_unet.sh")
    print()
    print("Or customize:")
    print("  python src/main.py \\")
    print("      --model resnet_unet \\")
    print("      --backbone resnet34 \\")
    print("      --pretrained \\")
    print("      --freeze_encoder \\")
    print("      --use_clahe")
    print()

if __name__ == '__main__':
    main()
