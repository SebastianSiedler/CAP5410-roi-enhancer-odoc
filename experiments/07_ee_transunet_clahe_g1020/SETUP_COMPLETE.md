# Experiment 07 Setup Complete ✅

## 🎯 What Was Created

I've successfully created **Experiment 07: EE-TransUNet + CLAHE on G1020 Dataset** with the following changes from Experiment 06:

### ✅ Key Changes

1. **CLAHE Preprocessing** - Enabled contrast-limited adaptive histogram equalization (LAB mode)
2. **G1020 Dataset** - Switched from REFUGE (400 samples) to G1020 (714 training samples, +78%)
3. **Larger Image Size** - Changed from 224×224 to 430×430 (native G1020 size)
4. **Lower Learning Rate** - Reduced from 0.01 to 0.001 (10× lower, lesson from Exp 06)
5. **Adam Optimizer** - Changed from SGD to Adam for more stable training

---

## 📁 Folder Structure

```
experiments/07_ee_transunet_clahe_g1020/
├── README.md                    # Comprehensive documentation
├── experiment.ipynb             # Jupyter notebook (ready to run)
├── train.py                     # Standalone training script
├── SETUP_COMPLETE.md           # This file
└── results/                     # Output directory (will be populated during training)
    ├── best_model.pth          # Best checkpoint (created during training)
    ├── config.json             # Configuration (created during training)
    ├── training_history.json   # Training metrics (created during training)
    ├── training_curves.png     # Loss/Dice plots (created during training)
    └── visualizations/         # Sample predictions (created during testing)
```

---

## 🚀 How to Run

### Prerequisites
1. **G1020 Dataset Split** (if not already done):
   ```bash
   python datasets/G1020/split_dataset.py
   ```
   This creates `train.csv`, `val.csv`, `test.csv` with 70/15/15 split.

2. **Virtual Environment Activated**:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

### Option 1: Jupyter Notebook (Recommended)
```bash
cd experiments/07_ee_transunet_clahe_g1020
jupyter notebook experiment.ipynb
```

**Then run cells sequentially:**
- Cells 1-4: Setup, config, data loading, CLAHE visualization
- Cell 5: Model definition (5.7M params)
- Cells 6-7: Training loop (will take 4-6 hours)
- Cell 8: Training curves visualization
- Cells 9-10: Testing and sample predictions
- Cell 11: Final analysis and comparison with Exp 06

### Option 2: Training Script
```bash
cd experiments/07_ee_transunet_clahe_g1020
python train.py
```

### Option 3: Custom Parameters
```bash
python train.py \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 0.001 \
    --img_size 430 \
    --use_clahe \
    --early_stop_patience 15
```

---

## 📊 Expected Results

### Based on Experiment 06 (REFUGE without CLAHE)
```
Exp 06 Test Dice:  87.21%
Exp 06 Disc Dice:  93.97%
Exp 06 Cup Dice:   80.45%
Exp 06 CDR MAE:    0.0869
```

### Expected for Experiment 07 (G1020 + CLAHE)
```
Target Test Dice:  88-90%  (+1-3% from CLAHE + more data)
Target Disc Dice:  94-95%  (Already excellent, slight improvement)
Target Cup Dice:   82-85%  (+2-5% from CLAHE edge enhancement)
Target CDR MAE:    0.065-0.080  (Better cup boundary detection)
```

### Why We Expect Improvement
1. **78% More Training Data** (714 vs 400 samples) → Better generalization
2. **CLAHE Preprocessing** → Enhanced contrast and edge detection
3. **Lower Learning Rate** (0.001 vs 0.01) → Stable training without spikes
4. **Adam Optimizer** → More stable than SGD with high momentum

---

## 🔧 Configuration Details

### Model Architecture
- **Name:** EE-TransUNet ViT-Tiny
- **Parameters:** 5,674,114 (5.7M)
- **Encoder:** Vision Transformer (6 layers, 256 hidden, 4 heads)
- **Decoder:** U-Net style (4 upsampling stages)
- **Input Size:** 430×430 (native G1020 size)
- **Patch Size:** 16×16 (results in 26×26 patches)

### CLAHE Configuration
```python
use_clahe = True
mode = 'LAB'  # Apply to L channel in LAB color space
clip_limit = 2.0  # Contrast enhancement level
tile_grid_size = (8, 8)  # Standard for medical images
```

### G1020 Dataset
```
Total samples: 1020
├── Training:   714 (70%)
├── Validation: 153 (15%)
└── Test:       153 (15%)

Image size: 430×430 (already cropped to ROI)
Mask format: Grayscale [0, 1, 2] = [background, disc, cup]
Label distribution: 71% normal, 29% glaucoma
```

### Training Configuration
```python
epochs = 50
batch_size = 4  # Reduced due to larger images
learning_rate = 0.001  # 10× lower than Exp 06
weight_decay = 1e-4
optimizer = Adam  # More stable than SGD
early_stopping_patience = 15 epochs
```

---

## 📈 Training Progress Monitoring

### Expected Training Timeline
```
Epoch 10: Val Dice ~0.74 (74%)
Epoch 20: Val Dice ~0.81 (81%)
Epoch 30: Val Dice ~0.85 (85%)
Epoch 40: Val Dice ~0.87 (87%)
Epoch 50: Val Dice ~0.88 (88%)
```

### Signs of Good Training
- ✅ Smooth validation curves (no large spikes)
- ✅ Small overfitting gap (< 5%)
- ✅ Steadily increasing Dice scores
- ✅ CDR MAE decreasing below 0.08

### Warning Signs
- ⚠️ Validation spikes → Reduce LR further
- ⚠️ Large overfitting gap → Add data augmentation
- ⚠️ Early plateau → Check data loading or model config

---

## 📝 Key Files

### README.md
Comprehensive documentation including:
- Experiment hypothesis and methodology
- Architecture details (encoder + decoder)
- Dataset comparison (REFUGE vs G1020)
- Expected results and improvements
- Lessons learned from Experiment 06
- Post-training analysis plan

### experiment.ipynb
Interactive Jupyter notebook with:
- 20+ cells covering full workflow
- CLAHE visualization (see the preprocessing effect!)
- Training loop with progress bars
- Automatic checkpoint saving (every 10 epochs)
- Early stopping (patience=15 epochs)
- Training curves plotting
- Test set evaluation
- Best/median/worst sample visualizations
- Comparison analysis with Experiment 06

### train.py
Standalone Python script with:
- Command-line argument parsing
- Configurable hyperparameters
- Full training loop
- Checkpoint saving
- Training history logging
- Automatic plot generation

---

## 🔬 Analysis After Training

### Metrics to Compare
1. **Overall Dice** (target: 88-90%)
2. **Disc Dice** (target: 94-95%)
3. **Cup Dice** (target: 82-85%, CLAHE should help here)
4. **CDR MAE** (target: < 0.08)
5. **Overfitting Gap** (target: < 5%)

### Visualizations Generated
1. **Training curves** (loss and Dice over epochs)
2. **Best/median/worst samples** (4-sample grid)
3. **CLAHE effect** (with vs without preprocessing)
4. **Per-class performance** (disc vs cup segmentation)

### Questions to Answer
1. ✅ Does CLAHE improve cup segmentation? (Expected: +2-5%)
2. ✅ Does G1020 reduce overfitting? (Expected: gap < 5%)
3. ✅ How does ViT-Tiny scale with more data? (Expected: linear improvement)
4. ✅ Is performance better than Exp 06? (Expected: +1-3%)

---

## 🎯 Next Steps

### If Results Are Good (Dice > 88%)
1. **Add augmentation** → Try to push to 90%
2. **ViT-Small** → Scale up model with more data
3. **Ensemble** → Combine with ResNet34-UNet
4. **Hyperparameter tuning** → Grid search batch size, LR

### If Results Are Similar (Dice ~87%)
1. **Analyze CLAHE impact** → Compare with/without
2. **Check G1020 quality** → Verify labels
3. **Try different sizes** → 224×224 vs 430×430
4. **Add skip connections** → Hybrid architecture (n_skip > 0)

### If Results Are Worse (Dice < 85%)
1. **Debug preprocessing** → Verify CLAHE working
2. **Check data loader** → G1020 format correct?
3. **Reduce image size** → Try 224×224
4. **Increase epochs** → More training time needed

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

**Hypothesis:** Exp 07 should be the best performing model due to:
- ViT's global attention mechanism
- CLAHE preprocessing for better contrast
- 78% more training data
- Improved hyperparameters (lower LR, Adam optimizer)

---

## ⚡ Quick Start Commands

```bash
# Navigate to experiment folder
cd experiments/07_ee_transunet_clahe_g1020

# Option 1: Jupyter (interactive)
jupyter notebook experiment.ipynb

# Option 2: Python script (automated)
python train.py

# Option 3: Custom parameters
python train.py --epochs 100 --learning_rate 0.0005 --batch_size 2
```

---

## 📚 Dependencies

All required packages are already installed in the project environment:
- PyTorch 2.8.0
- ml-collections 0.1.1
- scipy 1.14.1
- matplotlib, numpy, tqdm, pandas

---

## 💡 Tips

1. **GPU Recommended:** Training will take 4-6 hours on GPU, much longer on CPU
2. **Monitor Closely:** Check validation curves after first few epochs
3. **Checkpoints:** Saved every 10 epochs, use them if training interrupted
4. **Early Stopping:** Automatically stops if no improvement for 15 epochs
5. **Comparison:** After testing, compare results with Exp 06 in the notebook

---

**Status:** ✅ Ready to Train  
**Expected Training Time:** 4-6 hours (GPU)  
**Expected Performance:** 88-90% Test Dice (+3-5% vs Exp 06)

**Good luck with training! 🚀**
