# Enhancer V3 Training Plan

## V2 Results Summary (FAILED)

### What Happened:
- ✅ V2 improved over V1 on validation: 0.8301 vs 0.8228 (+0.73%)
- ❌ Still below baseline on validation: 0.8301 vs 0.8486 (-1.85%)
- ❌ **CATASTROPHIC on test set**: 0.8071 vs 0.8461 baseline (-3.91%)
- ❌ **Enhancement magnitude STILL too large**: 0.6251 (target: 0.01-0.1)

### Root Cause:
```python
# In enhancer.py line 177
delta = self.tanh(out) * 0.5  # Allows ±50% changes - WAY TOO LARGE!
```

This scale factor of **0.5** means the network can change pixel values by ±50%, which completely destroys images!

## V3 Fix Strategy

### PRIORITY 1: Reduce Enhancement Scale ⭐⭐⭐

**Change in `src/models/enhancer.py` line 177:**
```python
# OLD (V1/V2):
delta = self.tanh(out) * 0.5  # ±50% changes

# NEW (V3):
delta = self.tanh(out) * 0.05  # ±5% changes (10x reduction!)
```

**Why 0.05?**
- Target mean change: 0.01-0.1
- Current mean: 0.6251
- Reduction needed: 0.6251 → 0.05-0.10
- Factor: ~10x reduction
- New scale: 0.5 / 10 = 0.05

**Expected impact:**
- Mean absolute difference: 0.6251 → ~0.06-0.08 ✅
- Much subtler enhancements
- Should improve test performance significantly

### PRIORITY 2: Further Reduce Perceptual Weight

**Current V2 config:**
```python
'perceptual_weight': 0.05  # Still might be constraining
```

**New V3 config:**
```python
'perceptual_weight': 0.01  # 5x reduction to allow more freedom
```

### PRIORITY 3: Increase Training Epochs

V2 was still improving at epoch 50. Train longer for V3:
```python
'num_epochs': 100  # Up from 50
```

## V3 Training Configuration

```python
config_v3 = {
    # Architecture change (modify enhancer.py first!)
    'enhancement_scale': 0.05,  # In code: delta * 0.05 instead of 0.5
    
    # Loss weights
    'seg_weight': 1.0,
    'perceptual_weight': 0.01,  # ✅ REDUCED from 0.05
    'tv_weight': 0.001,  # Keep same
    
    # Training
    'num_epochs': 100,  # ✅ INCREASED from 50
    'batch_size': 16,
    'learning_rate': 1e-4,
    
    # Paths
    'segmentation_checkpoint': 'checkpoints/best_model.pth',  # Keep baseline
    'save_dir': 'checkpoints_enhancer_v3',
}
```

## Expected V3 Results

### Enhancement Magnitude:
- Mean: 0.05-0.08 (currently 0.6251) ✅
- Max: 0.15-0.25 (currently 2.12) ✅
- Status: "Good! Moderate changes (0.01-0.2)"

### Performance:
- **Conservative estimate**: Disc IoU 0.840-0.845 (neutral, not harmful)
- **Optimistic estimate**: Disc IoU 0.850-0.855 (+0.5-1% over baseline)
- **Minimum requirement**: Must beat 0.8461 baseline!

## Implementation Steps

### Step 1: Modify enhancer.py
```bash
# Edit line 177 in src/models/enhancer.py
# Change: delta = self.tanh(out) * 0.5
# To:     delta = self.tanh(out) * 0.05
```

### Step 2: Create V3 training notebook
Copy `train_enhancer_v2.ipynb` → `train_enhancer_v3.ipynb` with:
- Update title to V3
- Update config (perceptual 0.01, epochs 100)
- Add note about architecture change
- Update save directory

### Step 3: Train
```bash
# Should take ~4-5 hours for 100 epochs on GTX 2080 Ti
```

### Step 4: Evaluate
```bash
# Use evaluate_v2_results.ipynb (just change paths)
# Should see:
# - Enhancement magnitude: 0.05-0.08 ✅
# - Test Disc IoU: 0.845+ (target)
```

## Fallback Options (if V3 still fails)

### Option A: Even Smaller Scale
```python
delta = self.tanh(out) * 0.01  # ±1% changes - very conservative
```

### Option B: Remove Perceptual Loss Entirely
```python
'perceptual_weight': 0.0  # No L1 constraint, purely task-driven
```

### Option C: Switch to SimpleCNNEnhancer
```python
'enhancer_type': 'simple'  # Simpler architecture, might be easier to train
```

### Option D: Supervised Enhancement (Paired Data)
- Generate CLAHE versions of all images
- Train enhancer to match CLAHE (supervised)
- This guarantees it learns something useful
- Trade-off: Not fully task-aware, but should work

## Success Criteria

### Must Have:
- ✅ Enhancement magnitude 0.01-0.1 (not 0.6!)
- ✅ Test Disc IoU ≥ 0.8461 (baseline)

### Nice to Have:
- ✅ Test Disc IoU ≥ 0.850 (+1% over baseline)
- ✅ Visual improvements in enhancement patterns
- ✅ Beat CLAHE (0.8421)

### For Final Project:
Even if V3 only matches baseline (0.8461), you have a great story:
1. **Motivation**: Task-aware vs task-agnostic (CLAHE)
2. **V1 Failure**: Train/test mismatch + over-modification
3. **V2 Attempt**: Fixed frozen model, but still over-modifying
4. **V3 Solution**: Dramatically reduced enhancement scale
5. **Result**: Learned subtle, task-specific enhancements
6. **Learning**: Enhancement scale is critical hyperparameter!

This debugging process itself is valuable research contribution!
