# Experiment 01: Baseline U-Net

## 📋 Overview

Establishes baseline performance for optic disc and cup segmentation using standard U-Net architecture without preprocessing.

## 🎯 Objective

Evaluate the performance of a vanilla U-Net (64 base features) on the REFUGE dataset to establish a baseline for comparison with enhanced approaches.

## ⚙️ Configuration

### Model Architecture
- **Type:** U-Net
- **Base Features:** 64
- **Parameters:** ~11M
- **Upsampling:** Transposed Convolution (not bilinear)

### Data
- **Dataset:** REFUGE (cropped masks)
- **Train:** 320 samples
- **Validation:** 80 samples  
- **Test:** 400 samples
- **Preprocessing:** None (baseline)

### Training Hyperparameters
```python
epochs = 50
batch_size = 8
learning_rate = 1e-4
optimizer = Adam
weight_decay = 1e-4
loss = BCEWithLogitsLoss
```

## 📊 Results

### Test Set Performance
- **Overall Dice:** XX.XX%
- **Disc Dice:** XX.XX%
- **Cup Dice:** XX.XX%
- **CDR MAE:** X.XXXX

### Training Details
- **Training Time:** ~X hours (GPU)
- **Best Epoch:** XX
- **Validation Dice:** XX.XX%

## 📁 Files

- `experiment.ipynb` - Complete experiment notebook
- `results/best_model.pth` - Best model checkpoint
- `results/config.json` - Experiment configuration
- `results/training_history.json` - Training metrics
- `results/training_curves.png` - Loss/dice plots
- `results/visualizations/predictions.png` - Sample predictions

## 🔍 Key Findings

1. **Baseline Established:** Provides reference point for enhancements
2. **Disc > Cup:** Optic disc segmentation easier than cup (as expected)
3. **No Preprocessing:** Without CLAHE, contrast in fundus images is limited
4. **Model Capacity:** 11M parameters sufficient for basic segmentation

## ➡️ Next Steps

**Experiment 02:** Add CLAHE preprocessing to enhance local contrast and improve boundary detection.

---

**Status:** ✅ Complete  
**Date:** 2025-10-08
