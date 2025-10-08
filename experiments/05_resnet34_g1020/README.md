# Experiment 05: ResNet34-UNet on G1020 Dataset

**Date:** 2025-10-08  
**Objective:** Train ResNet34-UNet with CLAHE and strong augmentation on G1020 dataset

---

## 📋 Experiment Overview

### Dataset: G1020
- **Total samples:** 1020 fundus images
- **Training:** 714 samples (70%)
- **Validation:** 153 samples (15%)
- **Test:** 153 samples (15%)
- **Image size:** 430×430 (already cropped to ROI)
- **Mask format:** Grayscale with values [0, 1, 2] for background, disc, cup
- **Label distribution:** 
  - Normal (0): ~71%
  - Glaucoma (1): ~29%

### Model Architecture
- **Model:** ResNet34-UNet (pretrained encoder)
- **Parameters:** 24.5M total, 3.25M trainable when encoder frozen
- **Preprocessing:** CLAHE (LAB mode, clip_limit=2.0)
- **Augmentation:** Strong (rotation, flipping, color jitter, blur)

### Training Strategy
- **Two-stage training:**
  - Stage 1: Train decoder only (10 epochs, encoder frozen)
  - Stage 2: Fine-tune full model (40 epochs)
- **Augmentation:** 80% probability during training
- **Batch size:** 4 (adjusted for augmentation overhead)
- **Learning rate:** 1e-4
- **Optimizer:** Adam with weight decay 1e-4

### Expected Results
Based on Experiment 04 (REFUGE with augmentation):
- **Training Dice:** 70-80%
- **Validation Dice:** 73-78%
- **Overfitting gap:** 2-5% (healthy range)

---

## 📁 Files

- `experiment.ipynb` - Main training notebook
- `README.md` - This file
- `results/` - Training outputs (created during training)
  - `best_model.pth` - Best model checkpoint
  - `training_history.json` - Training metrics
  - `training_curves.png` - Loss/Dice plots
  - `test_results.json` - Test set evaluation
  - `visualizations/` - Prediction visualizations

---

## 🚀 How to Run

1. Ensure G1020 dataset is split:
   ```bash
   python datasets/G1020/split_dataset.py
   ```

2. Open and run the notebook:
   ```bash
   jupyter notebook experiment.ipynb
   ```

3. Results will be saved to `./results/`

---

## 📊 Key Differences from REFUGE Experiments

| Aspect | REFUGE (Exp 04) | G1020 (Exp 05) |
|--------|-----------------|----------------|
| Dataset size | 400 train / 400 val / 400 test | 714 train / 153 val / 153 test |
| Image source | Clinical (REFUGE challenge) | Clinical (G1020) |
| Image size | 432×494 (cropped) | 430×430 (cropped) |
| Task | Disc/Cup segmentation | Disc/Cup segmentation + Glaucoma classification |
| Class balance | N/A | 71% normal, 29% glaucoma |

---
