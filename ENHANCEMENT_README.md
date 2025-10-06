# Enhancement Pipeline Implementation

## Overview

This implementation adds **image quality enhancement** to the baseline U-Net segmentation pipeline. The enhancement module learns to restore degraded fundus images before segmentation.

## Why Enhancement + Defect Simulation?

### The Problem

Real-world fundus images often suffer from quality issues:

- **Gaussian noise** from camera sensors
- **Motion blur** from patient movement
- **Low contrast** from poor lighting conditions
- **Brightness variations** from different acquisition systems

### The Solution

Our approach:

1. **Simulate defects** on clean training images
2. **Train EnhancerNet** to reverse the degradation
3. **Improve segmentation** on low-quality images

## Architecture Components

### 1. Defect Simulation (`src/data_loader/augmentation.py`)

#### Available Defects:

- **Gaussian Noise**: `add_gaussian_noise(image, std=0.05)`
- **Gaussian Blur**: `add_gaussian_blur(image, kernel_size=5, sigma=1.5)`
- **Contrast Reduction**: `reduce_contrast(image, reduction_factor=0.5)`
- **Brightness Adjustment**: `adjust_brightness(image, factor=0.8)`

#### DefectSimulator Class:

```python
from data_loader.augmentation import DefectSimulator

simulator = DefectSimulator(
    noise_prob=0.8,           # 80% chance of adding noise
    blur_prob=0.8,            # 80% chance of adding blur
    contrast_prob=0.8,        # 80% chance of reducing contrast
    num_defects=(1, 3)        # Apply 1-3 defects randomly
)

degraded_image = simulator(clean_image)
```

### 2. EnhancerNet Models (`src/models/enhancer_net.py`)

#### Standard EnhancerNet (1.4M params):

```python
from models.enhancer_net import EnhancerNet

enhancer = EnhancerNet(
    in_channels=3,
    out_channels=3,
    base_features=32,
    num_residual_blocks=4
)
```

**Architecture:**

- Encoder-decoder with skip connections
- Residual blocks in bottleneck
- Learns residual (difference from input)
- Output clamped to [0, 1]

#### Lightweight EnhancerNet (45K params - 96.8% reduction):

```python
from models.enhancer_net import LightweightEnhancerNet

enhancer_light = LightweightEnhancerNet(
    base_features=16
)
```

**Features:**

- Depthwise separable convolutions
- Faster training and inference
- Minimal performance trade-off

### 3. Enhanced Dataset (`src/data_loader/dataset.py`)

Returns triplets: `(degraded_image, clean_image, mask)`

```python
from data_loader.dataset import EnhancedRetinaDataset
from data_loader.augmentation import DefectSimulator

simulator = DefectSimulator(num_defects=(1, 2))

dataset = EnhancedRetinaDataset(
    root_dir='datasets/REFUGE',
    csv_file='datasets/REFUGE/REFUGETrain.csv',
    defect_simulator=simulator,
    target_size=(512, 512),
    use_cropped=True
)

degraded, clean, mask = dataset[0]
```

## Training Strategies

### Strategy 1: Enhancer-Only Training

Train enhancer to restore images without segmentation:

**Loss**: MSE or L1 between enhanced and clean images

```python
loss = nn.L1Loss()(enhanced_image, clean_image)
```

**Pros**: Simple, fast convergence
**Cons**: Not optimized for segmentation task

### Strategy 2: Task-Aware Training (Recommended)

Train enhancer + segmenter jointly:

**Loss**: Combined enhancement + segmentation

```python
from utils.loss_functions import TaskAwareLoss

loss_fn = TaskAwareLoss(
    lambda_enhancement=0.3,  # 30% enhancement loss
    # 70% segmentation loss
)

total_loss, enh_loss, seg_loss = loss_fn(
    enhanced_img, clean_img,
    seg_pred, seg_target
)
```

**Pros**: Enhancement optimized for segmentation
**Cons**: More complex training

## Test Results

```
✓ Defect Simulation.............. PASSED
✓ EnhancerNet Models............. PASSED
✓ Enhanced Dataset............... PASSED
✓ Full Pipeline.................. PASSED
```

### Model Statistics:

- **Standard EnhancerNet**: 1,409,955 parameters
- **Lightweight EnhancerNet**: 45,172 parameters (96.8% smaller)

### Dataset Statistics:

- **Training samples**: 400 (with defects applied on-the-fly)
- **Validation samples**: 400 (with same defects for consistency)

## Usage Examples

### 1. Quick Test

```bash
source .venv/bin/activate
python test_enhancement.py
```

### 2. Visualize Enhancement

```bash
python -c "from test_enhancement import visualize_enhancement; visualize_enhancement()"
```

Creates `enhancement_visualization.png` showing:

- Original clean image
- Degraded image (with simulated defects)
- Enhanced image (from untrained model)
- Ground truth segmentation mask

### 3. Train Enhancer-Only

```python
import torch
from torch.utils.data import DataLoader
from data_loader.dataset import EnhancedRetinaDataset
from data_loader.augmentation import DefectSimulator
from models.enhancer_net import EnhancerNet

# Setup
simulator = DefectSimulator()
dataset = EnhancedRetinaDataset(
    root_dir='datasets/REFUGE',
    csv_file='datasets/REFUGE/REFUGETrain.csv',
    defect_simulator=simulator
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = EnhancerNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()

# Training loop
for degraded, clean, mask in loader:
    optimizer.zero_grad()
    enhanced = model(degraded)
    loss = criterion(enhanced, clean)
    loss.backward()
    optimizer.step()
```

### 4. Train Full Pipeline (Task-Aware)

```python
from models.unet import UNet
from utils.loss_functions import TaskAwareLoss

enhancer = EnhancerNet()
segmenter = UNet(n_channels=3, n_classes=2)
task_loss = TaskAwareLoss(lambda_enhancement=0.3)

# Training loop
for degraded, clean, mask in loader:
    enhanced = enhancer(degraded)
    seg_pred = segmenter(enhanced)

    total_loss, enh_loss, seg_loss = task_loss(
        enhanced, clean,
        seg_pred, mask
    )

    total_loss.backward()
    # ... optimizer steps
```

## File Structure

```
src/
├── data_loader/
│   ├── dataset.py                # Enhanced dataset classes
│   └── augmentation.py           # Defect simulation functions
├── models/
│   ├── unet.py                   # Segmentation model
│   └── enhancer_net.py           # Enhancement models
└── utils/
    ├── loss_functions.py         # Task-aware loss
    └── metrics.py                # Evaluation metrics

test_enhancement.py               # Comprehensive test suite
enhancement_visualization.png     # Visual output
```

## Next Steps

### Immediate:

1. ✅ Implement defect simulation
2. ✅ Implement EnhancerNet
3. ✅ Create enhanced dataset
4. ✅ Test full pipeline

### To Do:

1. **Train enhancer-only** on defected images
2. **Evaluate enhancement quality** (PSNR, SSIM)
3. **Train task-aware pipeline** with combined loss
4. **Compare segmentation performance**:
   - Baseline U-Net on clean images
   - U-Net on degraded images (no enhancement)
   - U-Net on enhanced images

### Expected Results:

| Configuration       | Dice Score (Clean) | Dice Score (Degraded) |
| ------------------- | ------------------ | --------------------- |
| Baseline U-Net      | ~0.90              | ~0.70                 |
| U-Net + EnhancerNet | ~0.90              | ~0.85                 |

## Key Insights

### Why This Approach Works:

1. **Realistic degradation**: Simulates actual image quality issues
2. **Task-aware learning**: Enhancement optimized for segmentation
3. **Lightweight design**: Fast inference for real-time applications
4. **Flexible training**: Can train enhancer separately or jointly

### Design Decisions:

- **Residual learning**: Enhancer learns correction, not full image
- **Skip connections**: Preserve spatial information during enhancement
- **Normalized inputs**: Consistent preprocessing for stability
- **Clamped outputs**: Ensures valid image range [0, 1]

## Troubleshooting

### Issue: Enhancement makes images worse

**Solution**: Train with more epochs, adjust learning rate, or increase lambda_enhancement

### Issue: Out of memory

**Solution**: Use LightweightEnhancerNet or reduce batch size

### Issue: Defects too strong/weak

**Solution**: Adjust DefectSimulator parameters (std, kernel_size, reduction_factor)

## References

- Baseline U-Net: `experiments/baseline_unet/README.md`
- Quick Start: `QUICK_START.md`
- Main README: `README.md`
