# Quick Start Guide - Baseline U-Net Training

## Environment Setup

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Verify Installation

```bash
python test_baseline.py
```

## Training the Baseline U-Net

### 1. Quick Start (Recommended Settings)

```bash
cd src
python main.py --epochs 100 --batch_size 8 --lr 1e-4
```

### 2. Full Configuration Example

```bash
cd src
python main.py \
    --data_dir ../datasets/REFUGE \
    --train_csv ../datasets/REFUGE/REFUGETrain.csv \
    --val_csv ../datasets/REFUGE/REFUGE1Val.csv \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --target_size 512 \
    --base_features 64 \
    --lambda_dice 0.5 \
    --lambda_bce 0.5 \
    --save_dir ../experiments/baseline_unet \
    --save_freq 10 \
    --device cuda  # or 'mps' for Mac M1/M2, or 'cpu'
```

### 3. Resume Training from Checkpoint

```bash
cd src
python main.py --resume ../experiments/baseline_unet/checkpoint_epoch_50.pth
```

## Monitoring Training

### View Training Log

```bash
tail -f experiments/baseline_unet/training_log.txt
```

### Log Format

```
epoch,train_loss,train_dice,val_loss,val_dice,val_cdr_mae
0,0.3521,0.8234,0.2987,0.8456,0.0234
1,0.3012,0.8567,0.2654,0.8723,0.0198
...
```

## Expected Training Time

- **GPU (CUDA)**: ~2-3 hours for 100 epochs
- **Mac M1/M2 (MPS)**: ~4-5 hours for 100 epochs
- **CPU**: ~10-15 hours for 100 epochs

## Outputs

After training, you'll find in `experiments/baseline_unet/`:

- `best_model.pth` - Best model based on validation Dice score
- `checkpoint_epoch_*.pth` - Periodic checkpoints every 10 epochs
- `config.json` - Training configuration
- `training_log.txt` - Detailed training metrics

## Expected Performance

- **Dice Score (Disc)**: ~0.95
- **Dice Score (Cup)**: ~0.85
- **Mean Dice**: ~0.90
- **CDR MAE**: <0.05

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
python main.py --batch_size 4
```

### Slow Training

Reduce image size:

```bash
python main.py --target_size 256 --batch_size 16
```

### Use Different Device

```bash
# For Mac M1/M2
python main.py --device mps

# For CPU only
python main.py --device cpu --batch_size 4
```

## Next Steps

After baseline training:

1. Check `experiments/baseline_unet/README.md` for detailed results
2. Implement defect simulation in `src/data_loader/augmentation.py`
3. Develop EnhancerNet in `src/models/enhancer_net.py`
4. Train the full enhancement pipeline

## Quick Commands Reference

```bash
# Activate venv
source .venv/bin/activate

# Test implementation
python test_baseline.py

# Train baseline
cd src && python main.py --epochs 100 --batch_size 8

# Monitor training (in another terminal)
tail -f experiments/baseline_unet/training_log.txt
```
