# Cropped Masks Generation - Solution

## Problem
The REFUGE dataset provides:
- Full resolution images (2056×2124 pixels)
- BMP ground truth masks aligned to full images
- "Cropped" JPG files that are upscaled/rotated versions

**Issue**: The provided "cropped" images don't align perfectly with the BMP masks, making it impossible to use them for training.

## Solution
Instead of trying to match the existing cropped images, we **generate our own perfectly aligned crops**:

### Approach
1. **Find the optic disc bounding box** in the BMP mask (pixels == 128)
2. **Add padding** (default 50px) and make it roughly square
3. **Crop both the full image AND masks** using the same bounding box
4. **Save the cropped versions** with perfect alignment

### Key Benefits
- ✅ **100% accurate alignment** - no template matching errors
- ✅ **40-50% disc coverage** - much better than 1.7% in full images
- ✅ **Consistent approach** - works for all 400 training samples
- ✅ **Simple and robust** - no need for complex matching algorithms

## Usage

### Generate cropped masks for single sample (with visualization):
```bash
python src/utils/generate_cropped_masks.py --test_sample 0853 --viz
```

### Generate for all 400 training samples:
```bash
python src/utils/generate_cropped_masks.py
```

### Output structure:
```
datasets/REFUGE_cropped_masks/
├── 0826/
│   ├── 0826_cropped.jpg          # Cropped image
│   ├── 0826_disc_cropped.bmp     # Cropped disc mask
│   ├── 0826_cup_cropped.bmp      # Cropped cup mask
│   └── 0826_visualization.png    # Visualization (if --viz)
├── 0827/
│   └── ...
└── ...
```

## Results
- **Mean disc coverage**: ~45% (vs 1.7% in full images)
- **Mean cup coverage**: ~10-15%
- **Crop size**: 330-470 pixels (varies by sample)
- **Success rate**: 100% on valid samples

## Integration with Training
To use the cropped masks for training, update `src/data_loader/dataset.py`:

```python
# Option 1: Use the generated cropped files
use_cropped = True  # Enable cropped mode
cropped_masks_dir = 'datasets/REFUGE_cropped_masks'  # Point to cropped masks

# Option 2: Generate on-the-fly (not recommended - slower)
# Use find_disc_bounding_box() in dataset.py
```

## Technical Details

### BMP Encoding
- **Disc mask**: 128 = disc foreground, 255 = background
- **Cup mask**: 0 = cup foreground, 255 = background

### Bounding Box Algorithm
1. Find all pixels where `disc_mask == 128`
2. Calculate min/max coordinates: `(x_min, y_min, x_max, y_max)`
3. Add padding and make square by using `max(width, height)`
4. Center the square on the disc center
5. Handle boundary cases (image edges)

### Why This Works
The BMP masks are the ground truth annotations - they define exactly where the optic disc and cup are located. By using these masks to define the crop region, we ensure perfect alignment between images and labels.

## Files Generated
- **Script**: `src/utils/generate_cropped_masks.py`
- **Output**: `datasets/REFUGE_cropped_masks/*/`
- **Log**: `cropped_masks_generation.log` (if running in background)

## Next Steps
1. ✅ Generate all 400 cropped masks
2. Update dataset loader to use cropped versions
3. Retrain model with correctly aligned data
4. Expect much better performance (Dice >75%)
