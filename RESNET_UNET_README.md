# ResNet-UNet for Glaucoma Detection

This directory contains an implementation of ResNet-based U-Net optimized for optic disc and cup segmentation in fundus images.

## 🎯 Why ResNet-UNet for Medical Imaging?

**Standard UNet** uses simple convolutional blocks in the encoder, while **ResNet-UNet** leverages pretrained ResNet architectures:

### Key Benefits:
1. **Transfer Learning**: Pretrained on ImageNet (1M+ images) for better feature extraction
2. **Residual Connections**: Better gradient flow, enabling deeper networks
3. **Medical Imaging**: Well-suited for fine-grained anatomical features
4. **Proven Architecture**: ResNet backbones are widely used in medical imaging

### Performance Improvements:
- **Optic Disc IoU**: Expected 2-5% improvement
- **Optic Cup IoU**: Expected 3-7% improvement (cup is harder to segment)
- **Faster Convergence**: Pretrained weights accelerate training
- **Better Generalization**: More robust on test data

## 📁 Files

### Models
- `src/models/resnet_unet.py` - ResNet-UNet architecture
  - `ResNetUNet` - ResNet34-based (recommended)
  - `ResNetUNetLite` - ResNet18-based (lightweight)

### Training
- `src/training/train_resnet.py` - Training script for ResNet-UNet
- `notebooks/train_resnet_unet.ipynb` - Interactive training notebook **[START HERE]**

### Evaluation
- `src/utils/compare_models.py` - Compare UNet vs ResNet-UNet

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)

Open and run: `notebooks/train_resnet_unet.ipynb`

The notebook includes:
- Model architecture overview
- Training configurations
- VRAM monitoring
- Side-by-side comparison with vanilla UNet
- Visualization of predictions

### Option 2: Command Line

```bash
# Activate virtual environment
source .venv/bin/activate

# Train ResNet34-UNet with CLAHE (recommended)
python src/training/train_resnet.py \
  --root_dir /home/robolab/dev/CAP5410-roi-enhancer-odoc \
  --epochs 100 \
  --batch_size 8 \
  --model_type resnet34 \
  --save_dir checkpoints_resnet34_clahe \
  --clahe

# Train lighter ResNet18-UNet (for larger batch sizes)
python src/training/train_resnet.py \
  --root_dir /home/robolab/dev/CAP5410-roi-enhancer-odoc \
  --epochs 100 \
  --batch_size 16 \
  --model_type resnet18 \
  --save_dir checkpoints_resnet18 \
  --clahe
```

## ⚙️ Model Configurations

### ResNet34-UNet (Recommended)

**Specs:**
- Parameters: ~24M
- VRAM Usage: ~8-9 GB (batch_size=8, 512x512)
- Training Time: ~4-5 hours (100 epochs on GTX 2080 Ti)

**Best For:**
- Maximum accuracy
- Standard training on GTX 2080 Ti
- Clinical applications requiring highest precision

**Config:**
```python
config = {
    'batch_size': 8,
    'image_size': 512,
    'model_type': 'resnet34',
    'pretrained': True,
    'use_clahe': True
}
```

### ResNet18-UNet Lite

**Specs:**
- Parameters: ~14M
- VRAM Usage: ~5-6 GB (batch_size=16, 512x512)
- Training Time: ~3-4 hours (100 epochs on GTX 2080 Ti)

**Best For:**
- Faster training
- Larger batch sizes
- Limited VRAM
- Good balance of speed and accuracy

**Config:**
```python
config = {
    'batch_size': 16,
    'image_size': 512,
    'model_type': 'resnet18',
    'pretrained': True,
    'use_clahe': True
}
```

## 📊 Model Comparison

After training, compare models:

```python
from utils.compare_models import compare_models

results = compare_models(
    root_dir='/home/robolab/dev/CAP5410-roi-enhancer-odoc',
    unet_checkpoint='checkpoints_clahe/best_model.pth',
    resnet_checkpoint='checkpoints_resnet34_clahe/best_model.pth',
    resnet_type='resnet34',
    use_clahe=True,
    save_dir='results'
)
```

This generates:
- `results/model_comparison.json` - Detailed metrics (IoU, Dice, Precision, Recall)
- `results/comparison_plot.png` - Visual comparison charts

## 🎓 Architecture Details

### Encoder (ResNet34)
```
Input (3, 512, 512)
  ↓
Conv1 + BN + ReLU → (64, 256, 256)  [skip1]
  ↓
MaxPool → (64, 128, 128)
  ↓
Layer1 (ResBlock x3) → (64, 128, 128)   [skip2]
  ↓
Layer2 (ResBlock x4) → (128, 64, 64)    [skip3]
  ↓
Layer3 (ResBlock x6) → (256, 32, 32)    [skip4]
  ↓
Layer4 (ResBlock x3) → (512, 16, 16)    [skip5]
```

### Decoder (U-Net Style)
```
Bottleneck (512, 16, 16)
  ↓
UpConv + skip4 → (256, 32, 32)
  ↓
UpConv + skip3 → (128, 64, 64)
  ↓
UpConv + skip2 → (64, 128, 128)
  ↓
UpConv + skip1 → (64, 256, 256)
  ↓
UpConv → (64, 512, 512)
  ↓
Conv1x1 → (3, 512, 512)  [Output: BG, Disc, Cup]
```

## 🔧 Training Features

### Differential Learning Rates
- **Encoder** (pretrained): `lr * 0.1` (lower to preserve learned features)
- **Decoder** (random init): `lr * 1.0` (higher to learn task-specific features)

### Loss Function
Combined Cross-Entropy + Dice Loss:
- Cross-Entropy: Pixel-wise classification
- Dice Loss: Handles class imbalance
- Class Weights: [1.0, 1.0, 2.0] (Background, Disc, Cup)

### Data Augmentation
- Random horizontal/vertical flip
- Random rotation (±15°)
- Color jittering
- Optional: CLAHE contrast enhancement

## 💾 VRAM Optimization for GTX 2080 Ti (11GB)

### Tips for VRAM Management:

1. **Batch Size**
   - ResNet34: Use batch_size=8 (512x512 images)
   - ResNet18: Can use batch_size=16

2. **Image Size**
   - 512x512: Best accuracy, needs smaller batches
   - 256x256: Faster training, allows larger batches

3. **Gradient Accumulation** (if needed)
   ```python
   # Simulate batch_size=16 with batch_size=8
   accumulation_steps = 2
   for i, (images, masks) in enumerate(train_loader):
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Mixed Precision Training** (future enhancement)
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       outputs = model(images)
       loss = criterion(outputs, masks)
   ```

## 📈 Expected Results

### Validation Metrics (100 epochs, CLAHE enabled):

**Original UNet:**
- Optic Disc IoU: ~0.85-0.87
- Optic Cup IoU: ~0.75-0.78

**ResNet34-UNet:**
- Optic Disc IoU: ~0.88-0.90 (+3-4%)
- Optic Cup IoU: ~0.80-0.83 (+5-7%)

**ResNet18-UNet:**
- Optic Disc IoU: ~0.86-0.88 (+2-3%)
- Optic Cup IoU: ~0.78-0.81 (+3-5%)

## 🐛 Troubleshooting

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
1. Reduce batch_size: 8 → 4
2. Use ResNet18 instead of ResNet34
3. Reduce image_size: 512 → 256
4. Clear GPU cache: `torch.cuda.empty_cache()`

### Pretrained Weights Warning
```
UserWarning: Using pretrained=True
```
This is normal! PyTorch is downloading ImageNet weights.

### Import Errors
```
ModuleNotFoundError: No module named 'models.resnet_unet'
```
Make sure to run from project root and add src to path:
```python
sys.path.append(str(Path.cwd().parent / 'src'))
```

## 📚 References

1. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
2. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
3. **Medical Imaging**: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)

## 🤝 Contributing

To add new ResNet variants (ResNet50, ResNet101):

1. Add model class in `src/models/resnet_unet.py`
2. Update `train_resnet.py` to support new type
3. Test VRAM usage before committing
4. Update this README with specs

## 📝 License

Same as parent project.
