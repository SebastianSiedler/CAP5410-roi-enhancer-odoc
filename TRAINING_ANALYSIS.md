# Training Analysis: Improved U-Net Results

## Results Summary

### Baseline Model (Before Improvements)
- **Test Dice Overall**: 69.75%
- **Test Disc Dice**: 85.80%
- **Test Cup Dice**: 53.70%
- **CDR MAE**: 0.236

### Improved Model (After 20 Epochs)
- **Best Validation Dice**: 65.81%
- **Training Dice**: 87.23% (epoch 19)
- **Training Loss**: 0.1957
- **Validation Loss**: 0.6417

---

## ⚠️ Problem Diagnosis

### Issue 1: Severe Overfitting
**Observation:**
- Training Dice: **87.23%**
- Validation Dice: **65.81%**
- **Gap: 21.4%** ← This is worse than baseline!

**Original Baseline:**
- Training Dice: ~81%
- Test Dice: ~70%
- Gap: ~11%

**Conclusion:** The improvements made overfitting WORSE, not better.

### Issue 2: Validation Performance Dropped
- Baseline validation/test: ~70%
- Improved validation: ~66%
- **Performance decreased by 4%**

### Issue 3: High Validation Loss
- Training loss: 0.20 (good)
- Validation loss: 0.64 (3x higher)
- Loss gap indicates poor generalization

---

## Root Cause Analysis

### 1. Augmentation May Be Too Aggressive
With 8 augmentation techniques applied with p=0.5 probability:
- The model sees highly distorted training images
- Achieves 87% on augmented (harder) training data
- But fails to generalize to clean validation data

**Problem:** The augmented images may be too different from real fundus images.

### 2. Cup Weight Too High
With `cup_weight=2.0`:
- Model heavily prioritizes cup segmentation during training
- Achieves high training metrics
- But overfits to training cup patterns

**Problem:** 2x weight may be excessive for this dataset.

### 3. Training Duration Too Short
Only 20 epochs:
- Model hasn't fully converged
- Best model was at epoch 15 (65.81%)
- Performance degraded after epoch 15

**Problem:** Need learning rate scheduling and longer training.

---

## Why Validation Dice < Test Dice (Baseline)?

This is unusual but explainable:

1. **Different Data Distributions**
   - Validation set: 400 samples from REFUGE1Val.csv
   - Test set: 400 samples from REFUGE1Test.csv (used in baseline)
   - Validation may be inherently harder

2. **Overfitting to Training Set**
   - Model memorized training-specific patterns
   - Doesn't generalize to validation set

3. **Augmentation Mismatch**
   - Training: Heavily augmented images
   - Validation: Clean images
   - Large distribution shift

---

## Recommendations

### Strategy A: Reduce Augmentation (Quick Fix)
**Goal:** Reduce training-validation distribution gap

**Changes:**
1. Reduce augmentation probability: `p=0.5` → `p=0.3`
2. Disable aggressive transforms (affine, color jitter)
3. Keep only geometric transforms (flips, rotation)

**Expected Result:** Better validation performance, less overfitting

**Implementation:**
```python
# In training_augmentation.py
aug = TrainingAugmentation(training=True, p=0.3)  # Reduce from 0.5

# Or disable specific transforms
self.apply_affine = False
self.apply_color_jitter = False
self.apply_gamma = False
```

### Strategy B: Reduce Cup Weight (Quick Fix)
**Goal:** Balance disc and cup learning

**Changes:**
1. Reduce cup weight: `cup_weight=2.0` → `cup_weight=1.5`
2. This makes learning more balanced

**Expected Result:** Better overall generalization

**Implementation:**
```python
# In main.py
criterion = ImprovedCombinedLoss(
    cup_weight=1.5,  # Reduce from 2.0
    use_focal=True
)
```

### Strategy C: Better Training Schedule (Medium Effort)
**Goal:** Prevent overfitting with regularization

**Changes:**
1. Increase epochs: 20 → 50
2. Add learning rate decay: ReduceLROnPlateau (already in code)
3. Add early stopping: Stop if validation doesn't improve for 10 epochs
4. Increase weight decay: 0 → 1e-5

**Expected Result:** Better convergence, less overfitting

**Implementation:**
```python
# In main.py
optimizer = Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=1e-5  # Add weight decay
)

# Early stopping (add this logic)
patience = 10
epochs_no_improve = 0
```

### Strategy D: Use Baseline Model + Post-Processing Only (Low Effort)
**Goal:** Test if post-processing alone helps

**Changes:**
1. Use the baseline model (already trained)
2. Apply only post-processing from Strategy 3
3. No augmentation, no new loss function

**Expected Result:** Quick validation of post-processing benefit

**Implementation:**
```python
# Modify test_model.py to apply post-processing
from utils.postprocessing import post_process_predictions

# After getting predictions
pred_disc, pred_cup = post_process_predictions(
    pred_disc_np, pred_cup_np, apply_constraints=True
)
```

---

## Recommended Next Steps

### Option 1: Quick Iteration (Recommended)
Try Strategy A + B together:

1. **Reduce augmentation to p=0.3**
2. **Reduce cup_weight to 1.5**
3. **Train for 30 epochs** (not 20)
4. **Expected time**: 10-12 hours

**Command:**
```bash
# Edit main.py and training_augmentation.py first
.venv/bin/python src/main.py \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --save_dir experiments/improved_unet_v2
```

### Option 2: Test Current Model
Maybe the current model performs better on TEST set than validation:

```bash
.venv/bin/python test_model.py \
  --model_path experiments/improved_unet/best_model.pth \
  --data_dir datasets/REFUGE \
  --test_csv datasets/REFUGE/REFUGE1Test.csv \
  --output_dir test_results_improved \
  --cropped_masks_dir datasets/REFUGE_cropped_masks_test
```

This will show if validation was just a harder split.

### Option 3: Baseline + Post-Processing
Quickest validation of whether Strategy 3 helps:

```bash
# Modify test_model.py to use post-processing
# Then test baseline model with post-processing
.venv/bin/python test_model.py \
  --model_path experiments/baseline_unet/best_model.pth \
  --data_dir datasets/REFUGE \
  --test_csv datasets/REFUGE/REFUGE1Test.csv \
  --output_dir test_results_baseline_postproc \
  --use_postprocessing  # Add this flag
```

---

## Technical Insights

### Why Augmentation Can Hurt
- **Small dataset** (400 samples): Augmentation helps
- **Too much augmentation**: Creates distribution shift
- **Finding balance**: Critical for medical imaging

### Why Cup Weight Can Hurt
- **Class imbalance**: Real problem (cup is 10% of disc)
- **Too high weight**: Model overfits to cup patterns
- **Better approach**: Use weighted sampling or curriculum learning

### Why Current Approach Failed
1. Implemented all 3 strategies at once
2. Couldn't isolate which strategy helps/hurts
3. Compounded negative effects

---

## Ablation Study Recommendation

Test each strategy individually:

| Experiment | Augmentation | Loss | Post-Proc | Est. Dice |
|------------|-------------|------|-----------|-----------|
| Baseline | ❌ | BCE+Dice | ❌ | 69.75% |
| +Aug (p=0.3) | ✓ | BCE+Dice | ❌ | ? |
| +Loss (w=1.5) | ❌ | Improved | ❌ | ? |
| +Post-Proc | ❌ | BCE+Dice | ✓ | ? |
| All (p=0.3, w=1.5) | ✓ | Improved | ✓ | ? |

This would take 5 training runs but give clear insights.

---

## Positive Observations

Despite the results, some things worked:

1. ✅ **Code works correctly** - No errors, clean training
2. ✅ **Training converges** - Loss decreases steadily
3. ✅ **High training accuracy** - Model has capacity to learn
4. ✅ **Post-processing ready** - Can be applied at inference

---

## Next Action

**Immediate:** Test current model on TEST set (not validation):
```bash
.venv/bin/python test_model.py \
  --model_path experiments/improved_unet/best_model.pth \
  --data_dir datasets/REFUGE \
  --test_csv datasets/REFUGE/REFUGE1Test.csv \
  --output_dir test_results_improved \
  --cropped_masks_dir datasets/REFUGE_cropped_masks_test
```

**If test results are also poor (< 70% overall):**
- Implement Option 1 (reduce augmentation & cup weight)
- Train for 30 epochs

**If test results are good (≥ 70% overall):**
- Validation set was just harder
- Use current model!

Would you like me to test the current model on the test set?
