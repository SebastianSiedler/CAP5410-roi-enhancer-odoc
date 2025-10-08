# Experiment 04: ResNet34-UNet + CLAHE + Strong Augmentation

**Date:** 2025-01-08  
**Status:** 🚧 In Progress  
**Goal:** Combat overfitting using aggressive data augmentation

---

## 📋 Problem Statement

### Issue Identified in Experiment 03
Experiment 03 (ResNet34-UNet + CLAHE) showed clear signs of **overfitting**:

```
Training Dice:     75-85%
Validation Dice:   66-70%
Overfitting Gap:   10-15% ❌
```

**Root Cause:**
- Only **400 training samples**
- Model has **24.5M parameters**
- Ratio: **0.016 samples per 1000 parameters** (catastrophically low!)
- Rule of thumb: Need at least 10 samples per 1000 parameters

---

## 💡 Solution: Strong Data Augmentation

### Hypothesis
By aggressively augmenting training data, we can:
1. **Increase effective dataset size** (400 → ~2000 samples with variations)
2. **Force generalization** (model can't memorize specific images)
3. **Simulate real-world variance** (rotation, lighting, blur, etc.)
4. **Act as regularization** (similar to dropout for data)

### Expected Results
```
Training Dice:     70-80%  (slightly lower, harder samples)
Validation Dice:   73-78%  (significant improvement! ✅)
Overfitting Gap:   2-5%    (healthy range ✅)
```

---

## 🔧 Configuration

### Model Architecture
- **Base:** ResNet34-UNet
- **Encoder:** ResNet34 (pretrained on ImageNet)
- **Parameters:** 24.5M total (17.1M trainable)
- **Input:** 512x512 RGB images
- **Output:** 2-channel segmentation (disc + cup)

### Preprocessing
- **CLAHE:** LAB mode, clip_limit=2.0
- **Normalization:** ImageNet statistics

### **NEW: Strong Augmentation Pipeline**

#### Geometric Transforms (p=0.8)
- `RandomHorizontalFlip(p=0.5)` - Mirror retina left/right
- `RandomVerticalFlip(p=0.5)` - Vertical flip
- `RandomRotation(degrees=30)` - Rotate ±30°
- `RandomAffine(translate=0.1, scale=(0.9, 1.1), shear=10)` - Affine distortions

#### Color Transforms (p=0.5)
- `ColorJitter(brightness=0.2)` - ±20% brightness
- `ColorJitter(contrast=0.2)` - ±20% contrast
- `ColorJitter(saturation=0.2)` - ±20% saturation
- `ColorJitter(hue=0.05)` - ±5% hue shift

#### Blur/Sharpness Transforms (p=0.3)
- `GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))` - Random blur
- `RandomAdjustSharpness(factor=2, p=0.5)` - Sharpen

**Overall augmentation probability:** 80% (vs. 0% in Exp 03!)

### Training Protocol
```python
{
    'epochs': 50,
    'freeze_encoder_epochs': 10,  # Two-stage training
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'optimizer': 'Adam',
    'loss': 'BCEWithLogitsLoss',
}
```

### Dataset
- **Training:** 400 samples (effective: ~2000 with augmentation!)
- **Validation:** 400 samples (no augmentation)
- **Test:** 400 samples (no augmentation)

---

## 📊 Implementation Details

### Code Changes

#### 1. **New Module:** `src/data_loader/strong_augmentation.py`
```python
class StrongAugmentation:
    """
    Aggressive augmentation pipeline for combating overfitting.
    
    Features:
    - Synchronized transforms for image + mask
    - Probability-based application
    - Geometric, color, and blur transforms
    """
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # Apply synchronized transforms
        # Returns: (augmented_image, augmented_mask)
```

#### 2. **Modified:** `src/data_loader/dataset.py`
```python
class RetinaDataset:
    def __init__(self, ..., augmentation: Optional[Callable] = None):
        self.augmentation = augmentation
    
    def __getitem__(self, idx):
        # ... load image and mask ...
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        
        return image, mask
```

#### 3. **New Notebook:** `experiments/04_resnet34_augmented/experiment.ipynb`
- Complete experiment notebook with augmentation
- Includes augmentation visualization section
- Uses `get_augmentation('strong')` factory function

---

## 🎯 Expected Impact

### Quantitative Improvements
| Metric | Exp 03 (No Aug) | Exp 04 (Strong Aug) | Improvement |
|--------|-----------------|---------------------|-------------|
| **Training Dice** | 75-85% | 70-80% | -5% (expected) |
| **Validation Dice** | 66-70% | 73-78% | **+7-8%** 🎉 |
| **Overfitting Gap** | 10-15% | 2-5% | **-8-10%** 🎉 |
| **Effective Samples** | 400 | ~2000 | **5x increase** |

### Qualitative Improvements
- ✅ **More robust model** - Handles diverse inputs better
- ✅ **Better generalization** - Performs well on unseen data
- ✅ **Reduced memorization** - Learns features, not specific images
- ✅ **Simulates real-world variance** - Rotation, lighting, blur, etc.

---

## 🔬 Why Augmentation Works

### Theory
1. **Data Efficiency:** Transforms 400 samples into ~2000 effective samples
2. **Regularization:** Acts like dropout but for data (prevents memorization)
3. **Invariance Learning:** Forces model to learn rotation/scale/color invariant features
4. **Distribution Matching:** Augmented data better matches real-world variance

### Empirical Evidence
- **Medical imaging:** Augmentation is standard practice (limited data availability)
- **ImageNet:** Top models use aggressive augmentation
- **Previous research:** Data augmentation reduces overfitting by 30-50%

### Why Strong Instead of Medium?
- **Small dataset:** 400 samples is very small for 24.5M parameters
- **High overfitting:** 10-15% gap requires aggressive intervention
- **Medical images:** Retinal images are naturally variable (patient differences)
- **Retinal anatomy:** Rotation-invariant (no "up" direction in fundus)

---

## 📈 Training Strategy

### Two-Stage Training (Same as Exp 03)
**Stage 1 (Epochs 0-9):** Freeze encoder, train decoder only
- Why? Prevent catastrophic forgetting of ImageNet features
- Trainable: 7.4M parameters (decoder only)

**Stage 2 (Epochs 10-49):** Unfreeze encoder, full fine-tuning
- Why? Adapt encoder to retinal images
- Trainable: 24.5M parameters (full model)

### Augmentation Schedule
- **Training:** Strong augmentation (p=0.8)
- **Validation:** NO augmentation (evaluate true generalization)
- **Testing:** NO augmentation (fair comparison)

---

## 📁 Directory Structure

```
experiments/04_resnet34_augmented/
├── experiment.ipynb           # Main experiment notebook
├── README.md                  # This file
└── results/                   # Generated during training
    ├── config.json            # Experiment configuration
    ├── training_history.json  # Loss/dice per epoch
    ├── training_curves.png    # Training visualization
    ├── best_model.pth         # Best model checkpoint
    ├── test_results.json      # Test metrics
    └── visualizations/        # Sample predictions
        └── test_predictions.png
```

---

## 🚀 How to Run

### 1. Start Jupyter
```bash
cd experiments/04_resnet34_augmented
jupyter notebook experiment.ipynb
```

### 2. Run All Cells
- **Section 1:** Environment setup
- **Section 2:** Augmentation demo (see before/after examples!)
- **Section 3:** Configuration
- **Section 4:** Data loading with augmentation
- **Section 5:** Model definition
- **Section 6:** Training (two-stage with augmentation)
- **Section 7:** Testing and evaluation

### 3. Monitor Training
Watch for:
- **Decreasing overfitting gap** (train dice ≈ val dice)
- **Increasing validation dice** (should reach 73-78%)
- **Stable training** (no sudden spikes/crashes)

---

## 🔍 Analysis & Next Steps

### If Results Are Good (Val Dice > 73%)
- ✅ **Use as final model** - Strong augmentation solved overfitting!
- Consider adding **Dropout** (Strategie 2) for even better results
- Try **ensemble** with Exp 03 (no aug) + Exp 04 (aug)

### If Results Are Not Good Enough (Val Dice < 73%)
- Try **medium augmentation** (less aggressive, p=0.6)
- Combine with **other strategies:**
  - Dropout (p=0.3-0.5)
  - Higher weight decay (1e-3)
  - Early stopping (patience=10)
- Consider **smaller model** (base_features=16) with augmentation

### If Overfitting Persists (Gap > 5%)
- Check augmentation is actually applied (visualize samples)
- Increase augmentation probability (p=0.9 or p=1.0)
- Add **Cutout/Erasing** augmentation
- Reduce model capacity (fewer features)

---

## 📚 References

### Data Augmentation in Medical Imaging
- Shorten & Khoshgoftaar (2019): "A survey on Image Data Augmentation for Deep Learning"
- Perez & Wang (2017): "The Effectiveness of Data Augmentation in Image Classification using Deep Learning"
- Nalepa et al. (2019): "Data Augmentation for Brain-Tumor Segmentation: A Review"

### Strong Augmentation Strategies
- Cubuk et al. (2019): "AutoAugment: Learning Augmentation Strategies from Data"
- Zhang et al. (2017): "mixup: Beyond Empirical Risk Minimization"
- DeVries & Taylor (2017): "Improved Regularization of Convolutional Neural Networks with Cutout"

### Retinal Image Segmentation
- Fu et al. (2018): "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation"
- Orlando et al. (2020): "REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment"

---

## 🏆 Success Criteria

### Minimum Viable Results
- ✅ **Validation Dice:** > 73% (vs. 66-70% in Exp 03)
- ✅ **Overfitting Gap:** < 5% (vs. 10-15% in Exp 03)
- ✅ **Test Dice:** > 72% (demonstrating true generalization)

### Ideal Results
- 🎯 **Validation Dice:** > 78%
- 🎯 **Overfitting Gap:** < 3%
- 🎯 **Test Dice:** > 77%

### Comparison with Literature
- **REFUGE Challenge winners:** 83-87% Dice (ensemble models)
- **Single model SOTA:** 80-82% Dice
- **Our target:** 75-80% Dice (reasonable for single model)

---

**Author:** CAP5410 Project Team  
**Last Updated:** 2025-01-08  
**Status:** Ready to train! 🚀
