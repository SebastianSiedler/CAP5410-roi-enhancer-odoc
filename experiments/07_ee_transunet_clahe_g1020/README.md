# Experiment 07: EE-TransUNet + CLAHE on G1020 Dataset

**Date:** 2025-10-08  
**Status:** 🚧 Ready to Train  
**Goal:** Combine Vision Transformer architecture with CLAHE preprocessing on larger G1020 dataset

---

## 📋 Experiment Overview

### Hypothesis
By combining EE-TransUNet's global attention mechanism with CLAHE preprocessing and training on the larger G1020 dataset (714 samples vs 400 in REFUGE), we expect:
1. Better contrast and edge detection from CLAHE
2. Improved generalization from larger dataset
3. Reduced overfitting due to 78% more training data

### Key Changes from Experiment 06
1. **Preprocessing:** Added CLAHE (LAB mode, clip_limit=2.0) before model input
2. **Dataset:** Switched from REFUGE (400 train) to G1020 (714 train)
3. **Data Volume:** 78% increase in training samples
4. **Image Size:** Changed from 224×224 to 430×430 (native G1020 size)

### Configuration
- **Model:** EE-TransUNet ViT-Tiny
- **Preprocessing:** CLAHE (LAB mode)
- **Dataset:** G1020 (cropped ROI)
- **Training samples:** 714 (70% of 1020)
- **Validation samples:** 153 (15% of 1020)
- **Test samples:** 153 (15% of 1020)
- **Image size:** 430×430 (native G1020 size)

### Hyperparameters
```python
{
    'epochs': 50,
    'batch_size': 4,  # Reduced due to larger image size
    'learning_rate': 0.001,  # Reduced from 0.01 (lesson learned!)
    'optimizer': 'Adam',  # Changed from SGD
    'weight_decay': 1e-4,
    'img_size': 430,  # Native G1020 size
    'patch_size': 16,
    'use_clahe': True,  # NEW!
    'use_augmentation': False,  # Start simple, add later if needed
}
```

---

## 🏗️ Model Architecture

### EE-TransUNet ViT-Tiny (Same as Exp 06)
```
Vision Transformer Encoder:
  - 6 Transformer layers
  - 256 hidden dimensions
  - 4 attention heads
  - 512 MLP dimensions
  
U-Net Decoder:
  - 4 upsampling stages: (128, 64, 32, 16) channels
  - Standard Conv2d operations
  
Total Parameters: 5,674,114 (5.7M)
Input Size: 430×430 (upscaled from 224×224)
Output: 2 channels (disc, cup)
```

### CLAHE Preprocessing Pipeline
```
Input RGB Image (430×430×3)
    ↓
Convert to LAB color space
    ↓
Apply CLAHE to L channel (clip_limit=2.0)
    ↓
Convert back to RGB
    ↓
Normalize (ImageNet stats)
    ↓
Patch Embedding (26×26 patches of 16×16 pixels)
    ↓
Vision Transformer → Decoder → Output
```

---

## 📊 Dataset: G1020

### Overview
- **Source:** G1020 clinical fundus images
- **Total samples:** 1020
- **Training:** 714 samples (70%)
- **Validation:** 153 samples (15%)
- **Test:** 153 samples (15%)

### Image Characteristics
- **Size:** 430×430 (already cropped to ROI)
- **Format:** RGB fundus images
- **Mask format:** Grayscale with values [0, 1, 2]
  - 0: Background
  - 1: Optic Disc
  - 2: Optic Cup

### Label Distribution
- **Normal (0):** ~71% (726 samples)
- **Glaucoma (1):** ~29% (294 samples)

### Comparison with REFUGE

| Aspect | REFUGE (Exp 06) | G1020 (Exp 07) |
|--------|-----------------|----------------|
| Training samples | 400 | 714 (+78%) |
| Validation samples | 80 | 153 (+91%) |
| Test samples | 400 | 153 (-62%) |
| Image size | 224×224 (resized) | 430×430 (native) |
| Preprocessing | None | CLAHE |
| Samples per 1000 params | 70 | 125 (+78%) |

**Key Advantage:** Much better data-to-parameter ratio (125 vs 70 samples per 1000 params)

---

## 🎯 Expected Results

### Based on Experiment 06 (REFUGE without CLAHE)
```
Overall Dice:  87.21%
Disc Dice:     93.97%
Cup Dice:      80.45%
CDR MAE:       0.0869
```

### Expected Improvements with CLAHE + G1020
```
Overall Dice:  88-90%  (+1-3% from better contrast & more data)
Disc Dice:     94-95%  (Already excellent, slight improvement)
Cup Dice:      82-85%  (+2-5% from CLAHE edge enhancement)
CDR MAE:       0.065-0.080  (More accurate cup boundaries)
```

### Training Stability
With lower LR (0.001 vs 0.01) and Adam optimizer:
- **Expected:** Smooth validation curves without spikes
- **Overfitting gap:** 2-5% (healthy range)
- **Training time:** ~4-6 hours (larger images + more samples)

---

## 🔧 Implementation Details

### CLAHE Configuration
```python
from src.data_loader.clahe_preprocessing import apply_clahe_lab

# Applied in dataset loader
use_clahe=True
# clip_limit=2.0 (enhances contrast without over-amplification)
# tile_grid_size=(8,8) (standard for medical images)
```

### Dataset Paths
```python
CONFIG = {
    'data_dir': '../../datasets/G1020',
    'train_csv': '../../datasets/G1020/train.csv',
    'val_csv': '../../datasets/G1020/val.csv',
    'test_csv': '../../datasets/G1020/test.csv',
    'use_cropped': True,  # G1020 already cropped
    'use_clahe': True,    # Enable CLAHE preprocessing
}
```

### Model Configuration
Same ViT-Tiny config as Experiment 06:
```python
from src.models.configs import get_tiny_config

config_vit = get_tiny_config()
config_vit.n_classes = 2
config_vit.n_skip = 0
config_vit.img_size = 430  # Updated for G1020
```

---

## 🚀 How to Run

### Prerequisites
1. **G1020 dataset split:**
   ```bash
   python datasets/G1020/split_dataset.py
   ```
   This creates train.csv, val.csv, test.csv (70/15/15 split)

2. **Environment activated:**
   ```bash
   .venv\Scripts\Activate.ps1
   ```

### Option 1: Jupyter Notebook (Recommended)
```bash
cd experiments/07_ee_transunet_clahe_g1020
jupyter notebook experiment.ipynb
```

Run cells sequentially:
- Cells 1-4: Setup, config, data loading
- Cell 5: Model definition
- Cells 6-7: Training (will take 4-6 hours)
- Cells 8-9: Testing and visualizations

### Option 2: Training Script
```bash
cd experiments/07_ee_transunet_clahe_g1020
python train.py
```

Results saved to `./results/`

### Option 3: Command Line with Custom Args
```bash
python train.py \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --img_size 430 \
    --use_clahe
```

---

## 📈 Monitoring Training

### Expected Training Progression
```
Epoch 10: Train Dice: 0.75, Val Dice: 0.74, Val Loss: 0.18
Epoch 20: Train Dice: 0.83, Val Dice: 0.81, Val Loss: 0.13
Epoch 30: Train Dice: 0.87, Val Dice: 0.85, Val Loss: 0.10
Epoch 40: Train Dice: 0.89, Val Dice: 0.87, Val Loss: 0.09
Epoch 50: Train Dice: 0.90, Val Dice: 0.88, Val Loss: 0.08
```

### Signs of Good Training
- ✅ Smooth validation loss curve (no large spikes)
- ✅ Small overfitting gap (2-5%)
- ✅ Steadily increasing Dice scores
- ✅ CDR MAE decreasing below 0.08

### Warning Signs
- ⚠️ Validation loss spikes → Reduce learning rate
- ⚠️ Overfitting gap > 10% → Add augmentation
- ⚠️ Dice plateaus early → Check data quality

---

## 🆚 Comparison with Previous Experiments

| Experiment | Model | Preprocessing | Dataset | Samples | Test Dice |
|------------|-------|---------------|---------|---------|-----------|
| 01 Baseline | UNet | None | REFUGE | 400 | 83.6% |
| 02 Small CLAHE | Small UNet | CLAHE | REFUGE | 400 | 86.5% |
| 03 ResNet34 | ResNet34-UNet | CLAHE | REFUGE | 400 | 72.1% |
| 06 ViT-Tiny | EE-TransUNet | None | REFUGE | 400 | 87.2% |
| 05 ResNet34-G1020 | ResNet34-UNet | CLAHE | G1020 | 714 | TBD |
| **07 ViT-CLAHE-G1020** | **EE-TransUNet** | **CLAHE** | **G1020** | **714** | **TBD** |

### Hypothesis
**Exp 07 should outperform Exp 06 because:**
1. 78% more training data (714 vs 400)
2. Better contrast from CLAHE preprocessing
3. Improved learning rate (0.001 vs 0.01)
4. Better optimizer (Adam vs SGD with high momentum)

**Expected ranking:**
1. 🥇 **Exp 07 (ViT-CLAHE-G1020): 88-90%** ← Best overall
2. 🥈 Exp 06 (ViT-Tiny): 87.2%
3. 🥉 Exp 02 (Small UNet CLAHE): 86.5%

---

## 📝 Lessons Applied from Experiment 06

### Issue 1: High Learning Rate ✅ Fixed
- **Problem:** LR=0.01 caused validation spikes
- **Solution:** Reduced to 0.001 (10× lower)

### Issue 2: No Data Augmentation ⚠️ Deferred
- **Problem:** Overfitting gap of 6%
- **Solution:** Start without augmentation on larger dataset, add if needed

### Issue 3: SGD with High Momentum ✅ Fixed
- **Problem:** Unstable training with SGD momentum=0.9
- **Solution:** Switched to Adam optimizer (more stable)

### Issue 4: Small Dataset ✅ Fixed
- **Problem:** Only 400 training samples
- **Solution:** G1020 has 714 samples (+78%)

---

## 🔬 Post-Training Analysis

### Metrics to Compare
1. **Test Dice Score** (overall, disc, cup)
2. **CDR Mean Absolute Error**
3. **Training stability** (smooth curves?)
4. **Overfitting gap** (train vs val)
5. **Edge quality** (CLAHE impact on boundaries)

### Visualization Plan
1. **Best/Median/Worst samples** (4-sample grid)
2. **CLAHE comparison** (with vs without preprocessing)
3. **Attention maps** (where does ViT look?)
4. **Failure cases** (low Dice samples analysis)

### Questions to Answer
1. Does CLAHE improve cup segmentation? (Expected: +2-5% Cup Dice)
2. Does larger dataset reduce overfitting? (Expected: gap < 5%)
3. How does ViT-Tiny scale with more data? (Expected: linear improvement)
4. Is 430×430 too large for ViT-Tiny? (May need more patches/layers)

---

## 🎯 Next Steps After Training

### If Results are Good (Dice > 88%)
1. **Add augmentation** → Try to push to 90%
2. **Ensemble** → Combine with ResNet34-UNet
3. **ViT-Small** → Scale up model with more data
4. **Hyperparameter tuning** → Grid search LR, batch size

### If Results are Similar to Exp 06 (Dice ~87%)
1. **Analyze CLAHE impact** → Compare with/without
2. **Check G1020 data quality** → Are labels accurate?
3. **Try different image sizes** → 224×224 vs 430×430
4. **Add skip connections** → Hybrid architecture

### If Results are Worse (Dice < 85%)
1. **Debug preprocessing** → Verify CLAHE is working
2. **Check data loader** → G1020 format correct?
3. **Reduce image size** → Try 224×224 first
4. **Increase training time** → More epochs needed

---

## 📚 References

1. **EE-TransUNet:** [GitHub Repository](https://github.com/wangyunyuwyy/EE-TransUNet)
2. **CLAHE:** Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization" (1994)
3. **G1020 Dataset:** Clinical fundus image collection
4. **Experiment 06:** EE-TransUNet on REFUGE (baseline)
5. **Experiment 05:** ResNet34-UNet on G1020 (comparison)

---

## 📂 Output Files

After training, `results/` will contain:
- `best_model.pth` - Best validation Dice checkpoint
- `config.json` - Full experiment configuration
- `training_history.json` - Loss/Dice per epoch
- `training_curves.png` - Training visualization
- `test_results.json` - Test set metrics
- `visualizations/` - Sample predictions
  - `predictions.png` - Best/median/worst samples
  - `clahe_comparison.png` - With/without CLAHE

---

**Status:** 🚧 Ready to Train  
**Expected Training Time:** 4-6 hours  
**Expected Performance:** 88-90% Test Dice (3-5% improvement over Exp 06)

---

**Notes:**
- Run notebook cells 1-9 sequentially
- Training checkpoints saved every 10 epochs
- Early stopping if no improvement for 15 epochs
- GPU recommended (will use CPU if unavailable)
