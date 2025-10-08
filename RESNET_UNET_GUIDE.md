# ResNet-UNet Model Guide

## Overview

ResNet-UNet combines the power of ResNet encoders (pretrained on ImageNet) with U-Net's decoder architecture for improved segmentation performance.

## Architecture

```
Input (3x512x512)
    ↓
ResNet Encoder (pretrained)
    ├─ Conv1 + BN + ReLU → 64 channels, H/2
    ├─ MaxPool → 64 channels, H/4
    ├─ Layer1 → 64/256 channels, H/4
    ├─ Layer2 → 128/512 channels, H/8
    ├─ Layer3 → 256/1024 channels, H/16
    └─ Layer4 → 512/2048 channels, H/32
              ↓
U-Net Decoder (skip connections)
    ├─ Decoder5 + Skip4 → H/16
    ├─ Decoder4 + Skip3 → H/8
    ├─ Decoder3 + Skip2 → H/4
    ├─ Decoder2 + Skip1 → H/2
    └─ Decoder1 → H (original size)
              ↓
Final Conv → 2 channels (disc, cup)
    ↓
Output (2x512x512)
```

## Key Features

### 1. Transfer Learning
- **Pretrained weights**: Encoder initialized with ImageNet weights
- **Better features**: General visual features transfer well to medical imaging
- **Faster convergence**: Starts from good initialization

### 2. Deep Architecture
- **ResNet18**: 11M layers, 14.4M parameters
- **ResNet34**: 21M layers, 24.5M parameters
- **ResNet50**: 48M layers, 72.0M parameters

### 3. Two-Stage Training
- **Stage 1 (epochs 0-9)**: Freeze encoder, train decoder only
  - Adapts decoder to encoder features
  - Prevents destroying pretrained weights
  - Only ~3-4M trainable parameters
  
- **Stage 2 (epochs 10-49)**: Unfreeze encoder, train full model
  - Fine-tunes encoder for retinal images
  - End-to-end optimization
  - Full parameter training

## Model Variants

### ResNet18 (Recommended for 400 images)
```python
model = ResNetUNet(n_classes=2, backbone='resnet18', pretrained=True)
```
- **Parameters**: 14.4M total, 3.3M decoder
- **Speed**: Fast (100-120 ms/image)
- **Memory**: Low (~4 GB GPU)
- **Best for**: Limited data (400 images)

### ResNet34 (Balanced)
```python
model = ResNetUNet(n_classes=2, backbone='resnet34', pretrained=True)
```
- **Parameters**: 24.5M total, 3.3M decoder
- **Speed**: Medium (120-150 ms/image)
- **Memory**: Medium (~5 GB GPU)
- **Best for**: Standard datasets

### ResNet50 (Maximum capacity)
```python
model = ResNetUNet(n_classes=2, backbone='resnet50', pretrained=True)
```
- **Parameters**: 72.0M total, 48.5M decoder
- **Speed**: Slow (150-200 ms/image)
- **Memory**: High (~8 GB GPU)
- **Best for**: Large datasets (>1000 images)

## Training Strategy

### Quick Start
```bash
# Train ResNet34-UNet with CLAHE
./train_resnet_unet.sh
```

### Custom Training
```bash
python src/main.py \
    --model resnet_unet \
    --backbone resnet34 \
    --pretrained \
    --freeze_encoder \
    --unfreeze_epoch 10 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --use_clahe \
    --clahe_mode LAB \
    --save_dir experiments/my_resnet_unet
```

### Key Hyperparameters

#### Learning Rate
- **Stage 1 (frozen)**: 1e-3 to 1e-4
  - Higher LR OK since only training decoder
  - Converges faster
  
- **Stage 2 (unfrozen)**: 1e-4 to 1e-5
  - Lower LR to fine-tune pretrained weights
  - Prevents destroying learned features

#### Batch Size
- **ResNet18/34**: 8-16 samples
- **ResNet50**: 4-8 samples (larger model)

#### Weight Decay
- **Recommended**: 1e-4
- Prevents overfitting on small dataset

### Two-Stage Training Schedule

```
Epoch 0-9:   Encoder FROZEN   → Train decoder only
              └─ LR: 1e-4, Trainable: 3.3M params

Epoch 10-49: Encoder UNFROZEN → Train full model
              └─ LR: 1e-4, Trainable: 24.5M params
```

## Advantages vs Standard U-Net

### 1. Better Initialization
- **Standard U-Net**: Random weights
- **ResNet-UNet**: Pretrained on 1M ImageNet images
- **Result**: Faster convergence, better features

### 2. Deeper Features
- **Standard U-Net**: 4-5 encoder levels
- **ResNet-UNet**: Much deeper (18-50 layers)
- **Result**: Can learn more complex patterns

### 3. Residual Connections
- **Standard U-Net**: Only skip connections
- **ResNet-UNet**: Skip connections + residual blocks
- **Result**: Easier gradient flow, better training

### 4. Proven Architecture
- **Standard U-Net**: Custom design
- **ResNet-UNet**: Battle-tested ResNet + U-Net
- **Result**: More reliable, well-understood

## Expected Performance

### Baseline Comparison

| Model | Overall Dice | Cup Dice | Parameters | Speed |
|-------|--------------|----------|------------|-------|
| **Small U-Net (32)** | 76.38% | 67.04% | 7.8M | 140ms |
| **ResNet18-UNet** | 77-79% | 68-71% | 14.4M | 120ms |
| **ResNet34-UNet** | 78-80% | 69-72% | 24.5M | 140ms |
| **ResNet50-UNet** | 78-81% | 70-73% | 72.0M | 180ms |

### Why ResNet-UNet Should Improve

1. **Better features**: ImageNet pretraining provides robust low-level features
2. **Deeper network**: Can capture more complex anatomical patterns
3. **Residual learning**: Easier to optimize, less prone to vanishing gradients
4. **Transfer learning**: Leverages knowledge from millions of natural images

## Training Tips

### 1. Monitor Overfitting
```python
# Watch training vs validation gap
if train_dice > val_dice + 0.10:
    # Overfitting! Increase weight_decay or reduce model size
```

### 2. Unfreeze Timing
- **Too early**: Destroys pretrained features
- **Too late**: Misses fine-tuning benefits
- **Recommended**: Epoch 10 (after decoder converges)

### 3. Data Augmentation
- Less aggressive than U-Net (pretrained features are robust)
- Focus on geometric transforms (rotation, flip)
- Avoid heavy color augmentation (encoder expects natural colors)

### 4. CLAHE Preprocessing
- **Essential**: ResNet expects normalized images
- **CLAHE first**: Enhances contrast before normalization
- **Mode**: LAB (best for medical images)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 4  # Instead of 8

# Or use smaller backbone
--backbone resnet18  # Instead of resnet34
```

### Poor Performance
```bash
# Make sure pretrained weights are loaded
--pretrained  # Don't forget this!

# Check encoder is unfreezing
--unfreeze_epoch 10

# Verify CLAHE is enabled
--use_clahe --clahe_mode LAB
```

### Slow Training
```bash
# Use smaller backbone
--backbone resnet18

# Increase batch size (if memory allows)
--batch_size 16
```

## Advanced Usage

### Fine-tuning from Checkpoint
```python
# Load pretrained model
checkpoint = torch.load('experiments/resnet34_unet_clahe/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune with lower learning rate
optimizer = Adam(model.parameters(), lr=1e-5)
```

### Ensemble Prediction
```python
# Train multiple models with different backbones
models = [
    ResNetUNet(backbone='resnet18', pretrained=True),
    ResNetUNet(backbone='resnet34', pretrained=True),
    ResNetUNet(backbone='resnet50', pretrained=True)
]

# Average predictions
pred = sum(model(x) for model in models) / len(models)
```

### Mixed Precision Training
```python
# Faster training with AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Implementation Details

### File Structure
```
src/models/resnet_unet.py      # Model implementation
train_resnet_unet.sh            # Training script
experiments/resnet34_unet_clahe/ # Output directory
```

### Model Code
```python
from models.resnet_unet import ResNetUNet

# Create model
model = ResNetUNet(
    n_classes=2,
    backbone='resnet34',
    pretrained=True
)

# Freeze encoder for initial training
model.freeze_encoder()

# Later: unfreeze for fine-tuning
model.unfreeze_encoder()

# Forward pass
output = model(input)  # (B, 2, H, W)
```

## Testing

### Quick Test
```bash
# Test model architecture
python src/models/resnet_unet.py

# Should output:
# ✅ RESNET18: 14.4M params
# ✅ RESNET34: 24.5M params
# ✅ RESNET50: 72.0M params
```

### Full Training Test
```bash
# Train for 5 epochs to verify setup
python src/main.py \
    --model resnet_unet \
    --backbone resnet18 \
    --pretrained \
    --epochs 5 \
    --save_dir experiments/test_resnet
```

## Comparison with Current Best Model

### Current: Small U-Net + CLAHE
- ✅ **Pros**: Small (7.8M), fast, good performance (76.38%)
- ❌ **Cons**: Random initialization, limited depth

### New: ResNet34-UNet + CLAHE
- ✅ **Pros**: Pretrained features, deeper, proven architecture
- ✅ **Expected**: 2-4% improvement (78-80% overall Dice)
- ⚠️ **Cons**: Larger (24.5M), slightly slower

### When to Use Which

**Use Small U-Net if:**
- Speed is critical (<150ms per image)
- Memory is limited (<4GB GPU)
- Current performance (76%) is sufficient

**Use ResNet-UNet if:**
- Want maximum accuracy
- Have GPU with 6+ GB memory
- Can afford 150-180ms per image
- Want to leverage pretrained features

## Next Steps

1. **Run training**: `./train_resnet_unet.sh`
2. **Monitor progress**: `tail -f experiments/resnet34_unet_clahe/training_log.txt`
3. **Compare results**: Test both models and compare
4. **Ablation study**: Try different backbones (18, 34, 50)

## Expected Timeline

- **Stage 1** (epochs 0-9): ~2 hours
  - Fast (decoder only)
  - Should reach ~70% dice
  
- **Stage 2** (epochs 10-49): ~6 hours
  - Slower (full model)
  - Should improve to ~78-80% dice

**Total**: ~8 hours on GTX 1080 Ti / RTX 2070

---

**Pro Tip**: Start with ResNet18 for quick experiments, then scale up to ResNet34/50 once you're happy with the setup!
