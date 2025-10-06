# Enhancement Pipeline Architecture

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ Clean Image  │
│  (512×512)   │
└──────┬───────┘
       │
       │ Apply Defects (on-the-fly)
       ├─── Gaussian Noise (std: 0.01-0.1)
       ├─── Gaussian Blur (kernel: 3-9, sigma: 0.5-2.0)
       ├─── Contrast Reduction (factor: 0.3-0.7)
       └─── Brightness Adjustment (factor: 0.7-0.9)
       │
       ↓
┌──────────────┐
│   Degraded   │
│    Image     │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────────────┐
│         EnhancerNet (1.4M params)        │
│  ┌────────────────────────────────────┐  │
│  │  Encoder:                          │  │
│  │   - Conv + BN + ReLU               │  │
│  │   - Downsample (→ 256×256)         │  │
│  │   - Downsample (→ 128×128)         │  │
│  │                                    │  │
│  │  Bottleneck:                       │  │
│  │   - 4× Residual Blocks             │  │
│  │                                    │  │
│  │  Decoder:                          │  │
│  │   - Upsample + Skip (→ 256×256)    │  │
│  │   - Upsample + Skip (→ 512×512)    │  │
│  │                                    │  │
│  │  Output:                           │  │
│  │   - Residual: output = input + Δ  │  │
│  │   - Clamp to [0, 1]                │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │
                   ↓
          ┌────────────────┐
          │ Enhanced Image │
          └────────┬───────┘
                   │
                   ↓
┌──────────────────────────────────────────┐
│           U-Net (13.4M params)           │
│  ┌────────────────────────────────────┐  │
│  │  Encoder:                          │  │
│  │   - DoubleConv (64)                │  │
│  │   - Down + DoubleConv (128)        │  │
│  │   - Down + DoubleConv (256)        │  │
│  │   - Down + DoubleConv (512)        │  │
│  │   - Down + DoubleConv (1024)       │  │
│  │                                    │  │
│  │  Decoder:                          │  │
│  │   - Up + Concat + DoubleConv       │  │
│  │   - Up + Concat + DoubleConv       │  │
│  │   - Up + Concat + DoubleConv       │  │
│  │   - Up + Concat + DoubleConv       │  │
│  │                                    │  │
│  │  Output:                           │  │
│  │   - 1×1 Conv → 2 channels          │  │
│  │   - Channel 0: Optic Disc          │  │
│  │   - Channel 1: Optic Cup           │  │
│  └────────────────────────────────────┘  │
└──────────────────┬───────────────────────┘
                   │
                   ↓
       ┌───────────────────────┐
       │  Segmentation Masks   │
       │  ┌─────┐    ┌─────┐   │
       │  │Disc │    │ Cup │   │
       │  └─────┘    └─────┘   │
       └───────────────────────┘
```

## Loss Functions

### Option 1: Enhancer-Only Training

```
Loss = L1(Enhanced, Clean)
     = Mean(|Enhanced - Clean|)
```

### Option 2: Task-Aware Training (Recommended)

```
Total Loss = λ × L_enhancement + (1-λ) × L_segmentation
           = 0.3 × L1(Enhanced, Clean) + 0.7 × (Dice + BCE)

where:
  L_enhancement = L1(Enhanced, Clean)
  L_segmentation = 0.5 × Dice_Loss + 0.5 × BCE_Loss
```

## Data Flow

### Training:

```python
# EnhancedRetinaDataset returns:
degraded_img, clean_img, mask = dataset[i]

# Forward pass:
enhanced = enhancer(degraded_img)
seg_pred = segmenter(enhanced)

# Compute losses:
enh_loss = L1(enhanced, clean_img)
seg_loss = DiceBCE(seg_pred, mask)
total = 0.3 * enh_loss + 0.7 * seg_loss
```

### Inference:

```python
# For degraded test image:
degraded_img = load_image()
enhanced = enhancer(degraded_img)
seg_pred = segmenter(enhanced)
disc_mask, cup_mask = seg_pred[0], seg_pred[1]
```

## Model Comparison

### EnhancerNet Variants:

| Model           | Parameters | Memory | Speed | Use Case     |
| --------------- | ---------- | ------ | ----- | ------------ |
| **Standard**    | 1.4M       | ~200MB | 1×    | Best quality |
| **Lightweight** | 45K        | ~10MB  | 3×    | Deployment   |

### Architecture Features:

| Feature      | EnhancerNet     | U-Net                |
| ------------ | --------------- | -------------------- |
| Purpose      | Enhancement     | Segmentation         |
| Input/Output | RGB → RGB       | RGB → 2-channel mask |
| Depth        | 3 levels        | 5 levels             |
| Bottleneck   | Residual blocks | DoubleConv           |
| Learning     | Residual        | Direct               |
| Parameters   | 1.4M / 45K      | 13.4M                |

## Defect Simulation Parameters

### Default Settings:

```python
DefectSimulator(
    noise_prob=0.8,              # 80% chance
    noise_std_range=(0.01, 0.1), # Mild to strong

    blur_prob=0.8,               # 80% chance
    blur_kernel_range=(3, 9),    # Small to medium
    blur_sigma_range=(0.5, 2.0), # Soft to strong

    contrast_prob=0.8,           # 80% chance
    contrast_range=(0.3, 0.7),   # Keep 30-70% contrast

    num_defects=(1, 3)           # Apply 1-3 defects
)
```

### Intensity Levels:

```python
# Low intensity (subtle degradation):
simulate_defects(img, intensity='low')
# → noise_std=0.02, blur_k=3, contrast=0.2

# Medium intensity (moderate degradation):
simulate_defects(img, intensity='medium')
# → noise_std=0.05, blur_k=5, contrast=0.4

# High intensity (severe degradation):
simulate_defects(img, intensity='high')
# → noise_std=0.1, blur_k=7, contrast=0.6
```

## Expected Results

### Segmentation Performance:

| Configuration    | Training Data   | Test Data | Dice Score |
| ---------------- | --------------- | --------- | ---------- |
| Baseline U-Net   | Clean           | Clean     | ~0.90      |
| Baseline U-Net   | Clean           | Degraded  | ~0.70 ❌   |
| U-Net + Enhancer | Clean + Defects | Degraded  | ~0.85 ✅   |
| Task-Aware       | Clean + Defects | Degraded  | ~0.87 ✅✅ |

### Enhancement Quality:

| Metric | Before Enhancement | After Enhancement |
| ------ | ------------------ | ----------------- |
| PSNR   | ~20 dB             | ~28 dB            |
| SSIM   | ~0.65              | ~0.85             |
| MSE    | ~0.02              | ~0.005            |

## File Organization

```
src/
├── data_loader/
│   ├── dataset.py
│   │   ├── RetinaDataset                    # Baseline (clean)
│   │   ├── RetinaDatasetValidation          # Baseline validation
│   │   ├── EnhancedRetinaDataset           # With defects
│   │   └── EnhancedRetinaDatasetValidation # Val with defects
│   │
│   └── augmentation.py
│       ├── add_gaussian_noise()
│       ├── add_gaussian_blur()
│       ├── reduce_contrast()
│       ├── adjust_brightness()
│       ├── DefectSimulator                  # Configurable
│       └── simulate_defects()               # Simple interface
│
├── models/
│   ├── unet.py
│   │   └── UNet                             # Segmentation
│   │
│   └── enhancer_net.py
│       ├── ResidualBlock
│       ├── EnhancerNet                      # Standard (1.4M)
│       ├── LightweightEnhancerNet          # Lightweight (45K)
│       └── AttentionEnhancerNet            # With attention
│
└── utils/
    ├── loss_functions.py
    │   ├── DiceLoss
    │   ├── DiceBCELoss
    │   ├── CombinedSegmentationLoss
    │   └── TaskAwareLoss                    # Enhancement + Seg
    │
    └── metrics.py
        ├── dice_coefficient()
        ├── iou_score()
        ├── calculate_cdr()
        └── batch_metrics()
```

## Quick Reference Commands

```bash
# Test enhancement pipeline
python test_enhancement.py

# View visualization
open enhancement_visualization.png

# Train enhancer only
cd src
python train_enhancer.py --epochs 50

# Train task-aware pipeline
python train_pipeline.py --lambda_enhancement 0.3 --epochs 100

# Evaluate on degraded images
python evaluate.py --use_enhancer --defect_level medium
```
