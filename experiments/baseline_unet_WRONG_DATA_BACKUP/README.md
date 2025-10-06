# Baseline U-Net Experiment

## Overview

This experiment implements a baseline U-Net model for optic disc (OD) and optic cup (OC) segmentation on the REFUGE dataset. The baseline serves as a reference for comparing future enhancement-based approaches.

## Architecture

- **Model**: Standard U-Net with encoder-decoder structure
- **Input**: RGB fundus images (cropped ROI, 512x512)
- **Output**: 2-channel segmentation map (disc and cup)
- **Parameters**: ~31M (with base_features=64)

## Dataset

- **Training**: REFUGE Training-400 dataset
- **Validation**: REFUGE Validation-400 dataset
- **Preprocessing**:
  - Cropped ROI images
  - Resized to 512x512
  - Normalized with ImageNet statistics

## Training Configuration

```bash
python src/main.py \
    --data_dir datasets/REFUGE \
    --train_csv datasets/REFUGE/REFUGETrain.csv \
    --val_csv datasets/REFUGE/REFUGE1Val.csv \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --target_size 512 \
    --save_dir experiments/baseline_unet
```

## Loss Function

Combined loss with:

- **Dice Loss** (λ=0.5): Measures overlap between prediction and ground truth
- **BCE Loss** (λ=0.5): Binary cross-entropy for pixel-wise classification

Total Loss: `L = 0.5 * L_dice + 0.5 * L_bce`

## Evaluation Metrics

1. **Dice Score**: Overlap measure for disc and cup (higher is better)
2. **IoU**: Intersection over Union for disc and cup
3. **CDR (Cup-to-Disc Ratio)**:
   - Calculated as cup_area / disc_area
   - CDR MAE (Mean Absolute Error) compared to ground truth

## Expected Results

- **Dice Score (Disc)**: ~0.95
- **Dice Score (Cup)**: ~0.85
- **Mean Dice**: ~0.90
- **CDR MAE**: <0.05

## Files Structure

```
experiments/baseline_unet/
├── config.json              # Training configuration
├── training_log.txt         # Epoch-by-epoch metrics
├── best_model.pth          # Best model checkpoint (highest val Dice)
└── checkpoint_epoch_*.pth  # Periodic checkpoints
```

## Running the Baseline

### 1. Test Implementation

```bash
python test_baseline.py
```

### 2. Train Model

```bash
cd src
python main.py --epochs 100 --batch_size 8
```

### 3. Monitor Training

Training progress is saved in `experiments/baseline_unet/training_log.txt`:

```
epoch,train_loss,train_dice,val_loss,val_dice,val_cdr_mae
0,0.3521,0.8234,0.2987,0.8456,0.0234
...
```

## Next Steps

After establishing the baseline:

1. Implement defect simulation (noise, blur, contrast reduction)
2. Develop EnhancerNet for image quality improvement
3. Train task-aware pipeline with combined enhancement + segmentation loss
4. Compare enhanced pipeline vs. baseline on degraded images

## Notes

- The baseline is trained on clean, high-quality cropped images
- No data augmentation is applied in this baseline version
- Model checkpoints are saved every 10 epochs
- Best model is automatically saved based on validation Dice score
