# Test Results Summary

**Date**: October 7, 2025  
**Model**: U-Net (31M parameters, base_features=64)  
**Checkpoint**: experiments/baseline_unet/best_model.pth (Epoch 65)  
**Test Dataset**: REFUGE Test-400 (400 samples with cropped masks)

---

## Overall Performance

### Segmentation Metrics

| Metric | Score |
|--------|-------|
| **Overall Dice** | **69.75%** |
| Disc Dice | **85.80%** ✅ |
| Cup Dice | **53.70%** ⚠️ |

### Cup-to-Disc Ratio (CDR) Analysis

| Metric | Value |
|--------|-------|
| **Mean Absolute Error** | **0.2360** |
| Max Error | 0.6897 |
| Min Error | 0.0010 |
| Standard Deviation | 0.1508 |

### CDR Distribution

| | Ground Truth | Predicted | Bias |
|---|--------------|-----------|------|
| Mean | 0.2583 | 0.4910 | **+0.233** ⚠️ |
| Std Dev | 0.1065 | 0.1682 | - |

---

## Analysis

### ✅ Strengths

1. **Excellent Disc Segmentation**: 85.8% Dice score is very good for medical imaging
2. **Stable Predictions**: Model successfully runs on all 400 test samples
3. **Test Set Generalization**: 69.75% overall Dice on unseen test data shows the model learned meaningful features

### ⚠️ Areas for Improvement

1. **Cup Segmentation**: 53.7% Dice indicates difficulty with the smaller, more variable cup region
2. **CDR Overestimation**: Model predicts CDR ~0.49 vs ground truth ~0.26 (89% higher)
   - This is likely due to cup over-segmentation
   - Critical issue for glaucoma diagnosis (CDR > 0.5 indicates glaucoma risk)

3. **Performance Gap**: 
   - Disc: Train (92.8%) → Test (85.8%) = **7% gap** ✅ Reasonable
   - Cup: Train (81.0%) → Test (53.7%) = **27% gap** ⚠️ Significant overfitting

---

## Comparison: Train vs Validation vs Test

| Split | Overall Dice | Disc Dice | Cup Dice | CDR MAE |
|-------|--------------|-----------|----------|---------|
| **Training** | 86.9% | 92.8% | 81.0% | - |
| **Validation** | 55.2% | 68.2% | 42.2% | 0.236 |
| **Test** | **69.8%** | **85.8%** | **53.7%** | **0.236** |

**Key Observation**: Test performance (69.8%) is **significantly better** than validation (55.2%). This is unusual but can occur due to:
- Different dataset characteristics (Test-400 vs Validation-400)
- Possible data quality differences
- Random chance in dataset splits

---

## Clinical Implications

### For Glaucoma Screening:

1. **Disc Detection**: ✅ Model reliably identifies optic disc location (85.8% accuracy)
2. **Cup Detection**: ⚠️ Moderate accuracy (53.7%) - needs improvement for clinical use
3. **CDR Estimation**: ❌ **Overestimation bias is critical**
   - Model predicts ~90% higher CDR than ground truth
   - Would lead to many false positives (healthy patients flagged as glaucoma suspects)
   - **Not suitable for clinical deployment without correction**

### Recommendations for Clinical Use:

1. **Calibration Needed**: Apply post-processing to correct CDR overestimation
2. **Human Verification**: Use as screening tool only, require ophthalmologist review
3. **Threshold Adjustment**: If using for triage, adjust CDR threshold from 0.5 to ~0.65 to account for bias

---

## Technical Insights

### What Worked Well:

1. **Data Alignment**: Cropped masks generation solved the misalignment issue
2. **Architecture**: U-Net with 64 base features provides good capacity
3. **Training Strategy**: ReduceLROnPlateau scheduler helped optimization

### Known Issues:

1. **Small Dataset**: Only 400 training samples → overfitting on cup segmentation
2. **Class Imbalance**: Cup is much smaller than disc → harder to learn
3. **No Data Augmentation**: Model memorizes training patterns

### Suggested Improvements:

1. **Data Augmentation** (Priority 1):
   - Rotation, flipping, color jitter
   - Elastic deformations
   - Could improve cup dice by 10-15%

2. **Loss Function Tuning**:
   - Increase weight on cup loss
   - Try Focal Loss for hard examples
   - Add CDR-specific loss term

3. **Architecture Modifications**:
   - Add attention mechanisms for cup region
   - Try deeper networks (ResNet-UNet, UNet++)
   - Multi-scale feature fusion

4. **Post-Processing**:
   - Ensure cup is always inside disc
   - Apply morphological operations
   - CDR calibration curve

5. **Ensemble Methods**:
   - Train multiple models with different seeds
   - Average predictions to reduce variance

---

## Files Generated

- `test_results/test_results.json` - Detailed metrics in JSON format
- `test_results/visualizations/` - 10 sample prediction visualizations
  - Shows side-by-side comparison of ground truth vs predictions
  - Includes dice scores and CDR values per sample

---

## Conclusion

The model achieves **69.75% overall Dice** on the test set, with strong disc segmentation (85.8%) but weaker cup segmentation (53.7%). While these results demonstrate the model has learned meaningful features, the **CDR overestimation bias (89% higher than ground truth)** is a critical issue that prevents clinical deployment without correction.

The performance represents a solid baseline but requires:
1. Data augmentation to reduce overfitting
2. Loss function tuning to improve cup segmentation  
3. Post-processing calibration to fix CDR bias

For research purposes, this is a good starting point. For clinical use, further development is required to meet the accuracy standards needed for glaucoma screening (typically >80% Dice for both disc and cup).

---

**Next Steps**: See visualization samples in `test_results/visualizations/` to understand failure modes and guide improvements.
