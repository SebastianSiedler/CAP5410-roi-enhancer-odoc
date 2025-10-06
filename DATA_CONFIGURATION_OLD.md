# Data Configuration Decision: Full Images vs Cropped ROIs

## Current Understanding

### REFUGE Dataset Structure:
1. **`XXXX.jpg`** - Full fundus image (e.g., 2124×2056)
2. **`XXXX_cropped.jpg`** - ROI crop, rotated (e.g., 2056×2124)
3. **`XXXX_disc.bmp`** - **Ground truth disc mask** (consensus from 7 experts) ✅
4. **`XXXX_cup.bmp`** - **Ground truth cup mask** (consensus from 7 experts) ✅
5. **`XXXX_seg_disc_1.png`** through **`_7.png`** - Individual expert annotations
6. **`XXXX_seg_cup_1.png`** through **`_7.png`** - Individual expert annotations

### Key Facts:
- **BMP masks = GROUND TRUTH** (merged consensus) - This is what you should use!
- **PNG masks = Individual experts** (for research/uncertainty analysis)
- **BMP masks match FULL images** in size
- **Cropped images have different dimensions** (rotated/transformed)

## The Problem

You have two mismatched configurations:

### Current Training (WRONG):
```python
use_cropped=True  →  XXXX_cropped.jpg (2056×2124)
mask = XXXX_seg_disc_1.png (2124×2056)  # Individual expert, not ground truth!
```

Problems:
- ❌ Using individual expert annotation (not consensus)
- ❌ Size mismatch requires resize (potential misalignment)
- ❌ Not using the official ground truth BMP files

### Previous Training (ALSO WRONG):
```python
use_cropped=True  →  XXXX_cropped.jpg (2056×2124)
mask = XXXX_disc.bmp (2124×2056)  # Ground truth, but wrong size!
```

Problems:
- ❌ Size mismatch causes severe misalignment
- ❌ Resizing a mask from wrong-sized image creates garbage

## Solution Options

### ✅ Option A: Use Full Images (RECOMMENDED)

**Configure dataset to use full images with BMP ground truth:**

```python
RetinaDataset(
    root_dir='datasets/REFUGE',
    csv_file='datasets/REFUGE/REFUGETrain.csv',
    target_size=(512, 512),
    use_cropped=False  # ← Use full images
)
```

**Dataset changes needed:**
```python
# In dataset.py, change to use BMP masks
img_path = f'{folder_name}.jpg'  # Full image
disc_mask_path = f'{folder_name}_disc.bmp'  # Ground truth!
cup_mask_path = f'{folder_name}_cup.bmp'    # Ground truth!
```

**Pros:**
- ✅ Perfect alignment: image and masks are same size
- ✅ Using official ground truth (BMP consensus)
- ✅ No transformation/crop issues
- ✅ Standard approach for medical imaging

**Cons:**
- ⚠️ Full fundus images include non-ROI regions
- ⚠️ Model sees more context (could be good or bad)
- ⚠️ Larger input → more GPU memory (but you're resizing to 512×512 anyway)

### Option B: Fix Crop Transformation

**Figure out the exact crop/rotation and apply to masks:**

Need to:
1. Understand the transformation from full→cropped
2. Apply same transformation to BMP masks
3. Create new cropped BMP masks

**Pros:**
- ✅ Focuses on ROI only (true to project goals)
- ✅ Smaller input data
- ✅ Using official ground truth

**Cons:**
- ⚠️ Complex - need to reverse-engineer the crop
- ⚠️ Excel files may have crop coordinates (need openpyxl)
- ⚠️ More preprocessing work

## Recommended Approach

### Phase 1: Train on Full Images (Quick Start)

**Change your training command:**
```bash
python src/main.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --save_dir experiments/baseline_unet_full_images
```

**Modify dataset.py** to:
1. Set `use_cropped=False` by default
2. Use `_disc.bmp` and `_cup.bmp` (ground truth)
3. Both image and masks will be same size → perfect alignment

This will give you a working baseline immediately.

### Phase 2: Optional ROI Cropping (Later)

If you need ROI-specific training:
1. Install `openpyxl`: `pip install openpyxl`
2. Read Excel files to get crop coordinates
3. Apply same crop to BMP masks
4. Save cropped masks
5. Use cropped everything

## What's Currently Running?

**Your training is using:**
```python
use_cropped=True
masks = _seg_disc_1.png / _seg_cup_1.png
```

This is **NOT using ground truth**, but rather **expert #1's annotations**.

While the size mismatch is less severe (both 2124×2056 vs 2056×2124 = just rotated), you're still:
- Not using official BMP consensus
- Have potential 90° rotation issue

## Recommendation: STOP and RESTART

**Stop current training** (Ctrl+C) and:

1. **Modify `src/data_loader/dataset.py`** to use:
   - `use_cropped=False`
   - BMP ground truth masks

2. **Restart training:**
```bash
python src/main.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4
```

This will give you a properly trained model on correct ground truth data.

---

## Quick Fix Code

Change the dataset initialization in `src/main.py`:

```python
# In create_dataloaders function:
train_dataset = RetinaDataset(
    root_dir=args.data_dir,
    csv_file=args.train_csv,
    target_size=(args.target_size, args.target_size),
    use_cropped=False  # ← Add this to use full images
)
```

Or modify dataset.py defaults as shown above.
