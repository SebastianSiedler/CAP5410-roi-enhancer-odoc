# Quick Training Guide

## ✅ Pre-Training Checklist

All improvements are implemented and tested:
- [x] Data augmentation (8 techniques)
- [x] Improved loss function (cup_weight=2.0)
- [x] Post-processing utilities
- [x] Integration tests passed

## 🚀 Start Training

### Basic Command
```bash
python src/main.py \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --save_dir experiments/improved_unet
```

### Background Training (Recommended)
```bash
# Start training in background
nohup python src/main.py \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --save_dir experiments/improved_unet \
  > improved_training.log 2>&1 &

# Get process ID
echo $!

# Monitor progress
tail -f improved_training.log

# Check if still running
ps aux | grep main.py
```

## 📊 What to Expect

### Training Metrics
- **Training Dice**: 75-85% (augmentation makes it harder)
- **Validation Dice**: 65-75% (should be better than baseline 55%)
- **Training Time**: ~8-10 hours (50 epochs on GTX 1080 Ti)

### Signs of Success
✓ Validation dice improves over baseline (55% → 65%+)
✓ Training-validation gap reduced (less overfitting)
✓ Cup dice improves significantly (54% → 63%+)

### Signs of Issues
⚠ Validation dice doesn't improve after 10 epochs
⚠ Training dice drops below 70%
⚠ Loss becomes NaN

## 🧪 After Training

### Test the Model
```bash
python test_model.py \
  --model_path experiments/improved_unet/best_model.pth \
  --data_dir datasets/REFUGE \
  --test_csv datasets/REFUGE/REFUGE1Test.csv \
  --output_dir test_results_improved
```

### Compare Results
```bash
# View baseline results
cat test_results/test_results.json

# View improved results
cat test_results_improved/test_results.json
```

### Expected Improvements
| Metric | Baseline | Target |
|--------|----------|--------|
| Cup Dice | 53.7% | 63-73% |
| Disc Dice | 85.8% | 86-88% |
| Overall | 69.8% | 75-80% |

## 🔧 Hyperparameter Tuning (If Needed)

### If cup dice < 63%:
```python
# In main.py, increase cup weight
criterion = ImprovedCombinedLoss(
    cup_weight=2.5,  # Increase from 2.0
    use_focal=True
)
```

### If training is too slow:
```python
# In training_augmentation.py, reduce probability
aug = TrainingAugmentation(training=True, p=0.3)  # Reduce from 0.5
```

### If model is underfitting:
```bash
# Increase batch size and learning rate
python src/main.py --batch_size 8 --lr 2e-4 --epochs 75
```

### If model is overfitting:
```bash
# More augmentation, more regularization
python src/main.py --weight_decay 1e-4 --epochs 100
```

## 📈 Monitoring Training

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View Training Curves
Training curves are saved in `experiments/improved_unet/`:
- `training_history.png`
- `training_metrics.csv`

### Stop Training Early
```bash
# Find process ID
ps aux | grep main.py

# Kill gracefully
kill <PID>
```

## 🎯 Success Criteria

Minimum (Good):
- Cup Dice ≥ 63%
- Overall Dice ≥ 75%

Target (Great):
- Cup Dice ≥ 68%
- Overall Dice ≥ 78%

Excellent (Publish):
- Cup Dice ≥ 73%
- Overall Dice ≥ 82%

## 📝 Notes

- Model checkpoints saved every epoch in `experiments/improved_unet/`
- Best model saved based on validation Dice
- Training can be resumed from checkpoint if interrupted
- Post-processing can be applied during evaluation (not training)

---

**Current Status**: Ready to train! 🚀

**Estimated Time**: 8-10 hours

**Next Step**: Run the training command above
