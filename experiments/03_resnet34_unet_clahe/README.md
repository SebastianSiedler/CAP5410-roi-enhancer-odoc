# Experiment 03: ResNet34-UNet + CLAHE 🏆

## 📋 Overview

**BEST MODEL SO FAR!** Combines ResNet34 pretrained encoder with U-Net decoder and CLAHE preprocessing for state-of-the-art performance on the REFUGE dataset.

## 🎯 Objective

Leverage transfer learning from ImageNet to extract more powerful features for improved optic disc and cup segmentation.

## ⚙️ Configuration

### Model Architecture
- **Type:** ResNet34-UNet
- **Encoder:** ResNet34 (ImageNet pretrained)
- **Decoder:** U-Net style with skip connections
- **Parameters:** 24.5M
- **Pretrained:** Yes (ImageNet) ✅

### Training Strategy
- **Two-Stage Training:**
  - **Phase 1 (Epochs 0-9):** Encoder frozen, train decoder only
  - **Phase 2 (Epochs 10-49):** Encoder unfrozen, fine-tune all

### Preprocessing
- **CLAHE Enabled:** Yes ✅
- **Color Space:** LAB mode
- **Clip Limit:** 2.0
- **Tile Grid Size:** 8x8

### Data
- **Dataset:** REFUGE (cropped masks)
- **Train:** 320 samples
- **Validation:** 80 samples  
- **Test:** 400 samples

### Training Hyperparameters
```python
epochs = 50
batch_size = 8
learning_rate = 1e-4
optimizer = Adam
weight_decay = 1e-4
loss = BCEWithLogitsLoss
freeze_encoder_epochs = 10
```

## 📊 Results

### Test Set Performance ⭐
- **Overall Dice:** **84.20%** ✅
- **Disc Dice:** **91.89%** ✅
- **Cup Dice:** **76.50%** ✅
- **CDR MAE:** **0.0542** ✅

### Training Details
- **Training Time:** ~8-10 hours (GPU)
- **Best Epoch:** 47
- **Best Validation Dice:** 84.94%

## 📈 Comparison with Previous Experiments

| Metric | Exp 01: Baseline | Exp 02: Small + CLAHE | **Exp 03: ResNet34 + CLAHE** | Improvement |
|--------|------------------|----------------------|------------------------------|-------------|
| **Overall Dice** | XX.XX% | 76.38% | **84.20%** | **+7.82%** |
| **Disc Dice** | XX.XX% | 85.72% | **91.89%** | **+6.17%** |
| **Cup Dice** | XX.XX% | 67.04% | **76.50%** | **+9.46%** |
| **CDR MAE** | X.XXX | 0.0710 | **0.0542** | **-23.8%** |
| **Parameters** | 11M | 2.7M | 24.5M | +9x |

### Key Improvements
- **+7.82% Overall Dice** vs. Small U-Net
- **+9.46% Cup Dice** - massive improvement on harder task
- **-23.8% CDR MAE** - more accurate clinical measurements

## 📁 Files

- `experiment.ipynb` - Complete experiment notebook
- `results/best_model.pth` - Best model checkpoint (epoch 47)
- `results/config.json` - Experiment configuration
- `results/training_curves.png` - Loss/dice plots with freeze/unfreeze marker
- `results/visualizations/predictions.png` - Sample predictions
- `../../test_results_resnet34_clahe_FIXED/` - Complete test results with 9 visualizations

## 🔍 Key Findings

### What Made the Difference?

1. **Deeper Encoder:** ResNet34 extracts more complex hierarchical features
   - Layer 1: 64 channels (low-level edges)
   - Layer 2: 128 channels (textures)
   - Layer 3: 256 channels (patterns)
   - Layer 4: 512 channels (semantic features)

2. **ImageNet Pretraining:** Transfer learning from 1M+ images
   - Better weight initialization
   - Faster convergence
   - Better generalization

3. **Two-Stage Training:** Smart fine-tuning strategy
   - Phase 1: Learn to use pretrained features
   - Phase 2: Adapt features to fundus images

4. **CLAHE Preprocessing:** Enhanced local contrast
   - Better boundary visibility
   - Improved cup detection

### Clinical Relevance

- **91.89% Disc Dice:** Excellent - suitable for clinical use ✅
- **76.50% Cup Dice:** Good - approaching clinical requirements ✅
- **0.0542 CDR MAE:** Acceptable error range for glaucoma screening ✅

### Visual Quality

Unlike the initial visualization bug (without CLAHE), predictions now show:
- ✅ Clean, circular disc boundaries
- ✅ Accurate cup sizes (not oversized)
- ✅ Minimal artifacts
- ✅ Anatomically correct CDR ratios

## ⚠️ Important Notes

### CLAHE Preprocessing is CRITICAL!

**Issue Discovered:** Initial visualizations looked terrible because CLAHE was not applied during inference.

**Solution:** Always use `--use_clahe` flag when:
- Testing the model
- Generating visualizations  
- Running inference

**Correct Command:**
```bash
python test_model.py \
    --checkpoint experiments/resnet34_unet_clahe/best_model.pth \
    --use_clahe \
    --clahe_mode LAB \
    --visualize
```

## 🚀 Future Work

### Potential Improvements

1. **Cup Segmentation Enhancement (Target: >78%)**
   - Shape-aware loss functions (Hausdorff distance)
   - Boundary refinement module
   - Attention mechanisms on cup region

2. **Model Efficiency**
   - Try ResNet18 or MobileNetV2 (faster inference)
   - Knowledge distillation
   - Quantization for deployment

3. **Data Augmentation**
   - Rotation, flipping, scaling
   - Color jittering
   - Elastic deformations

4. **Ensemble Methods**
   - Combine multiple models
   - Test-time augmentation

## 📚 References

- He et al. "Deep Residual Learning for Image Recognition" (ResNet)
- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- REFUGE Challenge 2018

---

**Status:** ✅ Complete - **BEST MODEL!** 🏆  
**Date:** 2025-10-08  
**Recommended for:** Clinical evaluation and deployment
