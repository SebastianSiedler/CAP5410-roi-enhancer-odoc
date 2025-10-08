# Experiment 05: G1020 Dataset Setup Complete

**Date:** 2025-10-08  
**Status:** ✅ Ready for Training

---

## 📊 Dataset Summary

### Split Statistics
- **Total samples:** 1020
- **Training:** 714 samples (70.0%)
  - Normal (0): 507 samples
  - Glaucoma (1): 207 samples
- **Validation:** 153 samples (15.0%)
  - Normal (0): 108 samples
  - Glaucoma (1): 45 samples
- **Test:** 153 samples (15.0%)
  - Normal (0): 109 samples
  - Glaucoma (1): 44 samples

### Data Format
- **Image location:** `datasets/G1020/Images_Cropped/img/`
- **Mask location:** `datasets/G1020/Masks_Cropped/img/`
- **Image format:** JPG, 430×430 pixels, RGB
- **Mask format:** PNG, 430×430 pixels, grayscale
- **Mask values:**
  - 0: Background
  - 1: Optic disc
  - 2: Optic cup
- **Note:** Some samples may have cup=0 (no cup annotation)

---

## 🗂️ Files Created

### Dataset Files
- ✅ `datasets/G1020/G1020_train.csv` - Training split (714 samples)
- ✅ `datasets/G1020/G1020_val.csv` - Validation split (153 samples)
- ✅ `datasets/G1020/G1020_test.csv` - Test split (153 samples)
- ✅ `datasets/G1020/split_dataset.py` - Split generation script

### Code Files
- ✅ `src/data_loader/g1020_dataset.py` - G1020 dataset loader class
- ✅ `experiments/05_resnet34_g1020/experiment.ipynb` - Training notebook
- ✅ `experiments/05_resnet34_g1020/README.md` - Experiment documentation

---

## 🚀 How to Run the Experiment

1. **Open the notebook:**
   ```bash
   cd experiments/05_resnet34_g1020
   jupyter notebook experiment.ipynb
   ```

2. **Run all cells** in order:
   - Cell 1-4: Environment setup and imports
   - Cell 5: Configuration
   - Cell 6-8: Data loading and validation
   - Cell 9-10: Model creation
   - Cell 11-12: GPU memory check
   - Cell 13: Training (50 epochs, ~2-3 hours)
   - Cell 14-16: Evaluation and visualization

3. **Monitor training:**
   - Training progress will show in the notebook
   - Best model saved to `./results/best_model.pth`
   - Training curves saved to `./results/training_curves.png`

---

## 📈 Expected Results

Based on Experiment 04 (REFUGE with strong augmentation):
- **Training Dice:** 70-80%
- **Validation Dice:** 73-78%
- **Overfitting Gap:** 2-5% (healthy range)

G1020 has more training data (714 vs 400), so we might see:
- Similar or slightly better performance
- Better generalization due to larger dataset

---

## ⚙️ Training Configuration

```python
{
    'model': 'ResNet34-UNet',
    'encoder': 'ResNet34 (pretrained)',
    'preprocessing': 'CLAHE (LAB, clip=2.0)',
    'augmentation': 'Strong (p=0.8)',
    'batch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'freeze_encoder_epochs': 10,
}
```

---

## 🔍 Validation Performed

✅ Dataset loader tested successfully:
- All 714 training samples loadable
- Image shape: (3, 512, 512)
- Mask shape: (2, 512, 512) - [disc, cup]
- Disc and cup channels properly separated
- Label distribution verified

---

## 📝 Notes

1. **Cup annotations:** Some samples have no cup annotations (cup=0). This is normal and handled by the loss function.

2. **Image paths:** Images are in `Images_Cropped/img/` and masks in `Masks_Cropped/img/` subdirectories.

3. **Stratified split:** The train/val/test split maintains the same label distribution (~71% normal, ~29% glaucoma) in all sets.

4. **GPU memory:** Training uses ~3-4 GB GPU memory with batch_size=4. Adjust if needed.

---

**Ready to train! Open the notebook and run all cells to start the experiment.**
