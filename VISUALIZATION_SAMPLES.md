# Sample Visualizations - Small U-Net + CLAHE Model

## Overview
Generated 9 sample prediction visualizations from the test set to demonstrate the model's performance visually.

## Model Details
- **Model**: Small U-Net (32 base features, 7.8M parameters)
- **Preprocessing**: CLAHE (LAB mode, clip_limit=2.0)
- **Checkpoint**: experiments/small_unet_clahe/best_model.pth (epoch 51)
- **Test Performance**: 76.38% overall Dice, 67.04% cup Dice

## Generated Samples

### Sample Selection Strategy
Selected 9 diverse test images across the dataset:
- **Sample 000** (0401): First image
- **Sample 050** (0458): Early section
- **Sample 100** (0510): First quarter
- **Sample 150** (0565): Middle section
- **Sample 200** (0616): Center
- **Sample 250** (0668): Third quarter
- **Sample 300** (0718): Late section
- **Sample 350** (0775): Near end
- **Sample 399** (1281): Last image

### Visualization Layout
Each visualization contains 8 panels arranged in 2 rows × 4 columns:

#### Row 1: Input and Ground Truth
1. **Original Image**: Raw fundus image from REFUGE test set
2. **CLAHE Enhanced**: Image after CLAHE preprocessing (LAB mode)
3. **GT Disc Mask**: Ground truth optic disc segmentation
4. **GT Cup Mask**: Ground truth optic cup segmentation

#### Row 2: Predictions and Overlays
1. **Predicted Disc**: Model's disc segmentation
2. **Predicted Cup**: Model's cup segmentation
3. **GT Overlay**: Ground truth overlaid on original (Red=Disc, Green=Cup)
4. **Prediction Overlay**: Model prediction overlaid on original (Red=Disc, Green=Cup)

### File Locations
```
test_results_small_clahe/visualizations/
├── sample_000_0401.png  (1.2 MB)
├── sample_050_0458.png  (1.2 MB)
├── sample_100_0510.png  (1.2 MB)
├── sample_150_0565.png  (1.3 MB)
├── sample_200_0616.png  (1.2 MB)
├── sample_250_0668.png  (1.2 MB)
├── sample_300_0718.png  (1.2 MB)
├── sample_350_0775.png  (1.2 MB)
└── sample_399_1281.png  (1.2 MB)
```

Total size: ~11 MB (9 high-resolution visualizations)

## How to View

### Command Line
```bash
# List all visualizations
ls -lh test_results_small_clahe/visualizations/

# View with image viewer (Linux)
eog test_results_small_clahe/visualizations/sample_*.png

# View with default application
xdg-open test_results_small_clahe/visualizations/
```

### In Jupyter Notebook
```python
from IPython.display import Image, display
import glob

# Display all samples
for img_path in sorted(glob.glob('test_results_small_clahe/visualizations/*.png')):
    print(f"\n{img_path}")
    display(Image(filename=img_path, width=800))
```

### In Python Script
```python
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Display all in grid
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for idx, img_path in enumerate(sorted(glob.glob('test_results_small_clahe/visualizations/*.png'))):
    row, col = idx // 3, idx % 3
    img = Image.open(img_path)
    axes[row, col].imshow(img)
    axes[row, col].set_title(Path(img_path).name, fontsize=10)
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()
```

## What to Look For

### Good Performance Indicators
- ✅ **Disc**: Red overlay matches between GT and prediction
- ✅ **Cup**: Green overlay matches between GT and prediction
- ✅ **Smooth boundaries**: Clean, anatomically plausible shapes
- ✅ **CLAHE enhancement**: Visible improvement in contrast

### Common Challenges
- 🔍 **Cup boundary**: Often low contrast, harder to segment
- 🔍 **Irregular shapes**: Some cups are non-circular
- 🔍 **Image quality**: Varying brightness and contrast
- 🔍 **Disc-cup relationship**: Cup must be contained within disc

## Performance Notes

### Compared to Baseline
- **Baseline**: 69.75% overall, 53.70% cup
- **This Model**: 76.38% overall, 67.04% cup
- **Improvement**: +6.63% overall, +13.34% cup (25% relative improvement!)

### CLAHE Impact
The CLAHE enhanced images (panel 0,1) show:
- Enhanced contrast in optic disc region
- Better visibility of cup boundaries
- Preserved color information (LAB mode)
- Reduced illumination variations

### Visualization Quality
- Resolution: 512×512 pixels per mask
- Format: PNG (lossless)
- DPI: 150 (high quality for papers/presentations)
- File size: ~1.2 MB per visualization

## Usage in Documentation

These visualizations are perfect for:
1. **Project Reports**: Demonstrating model performance visually
2. **Presentations**: Showing input → processing → output pipeline
3. **Papers**: Qualitative results section
4. **Debugging**: Visual inspection of failure cases
5. **Comparison**: Side-by-side with other methods

## Generation Script

To regenerate or create more samples:
```bash
# Generate visualizations (as run)
cd /home/robolab/dev/CAP5410-roi-enhancer-odoc
.venv/bin/python -c "
# ... (script content as executed)
"

# Or modify sample_indices to select different images:
# sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 399]
# Change to any indices between 0-399
```

## Next Steps

### More Visualizations
To generate additional samples:
1. Modify `sample_indices` to include more images
2. Add failure case analysis (low Dice score images)
3. Create comparison with baseline model predictions
4. Generate video showing all 400 test predictions

### Analysis
1. Identify challenging cases (low Dice scores)
2. Analyze CLAHE impact on different image types
3. Compare disc vs cup prediction quality
4. Evaluate performance on glaucoma vs non-glaucoma cases

---

**Generated**: October 8, 2025  
**Model**: experiments/small_unet_clahe/best_model.pth  
**Test Set**: REFUGE Test-400 (400 images)  
**Performance**: 76.38% overall Dice, 85.71% disc, 67.04% cup
