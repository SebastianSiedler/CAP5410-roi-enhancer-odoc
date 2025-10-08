# EE-TransUNet Integration Guide

## Overview

Your codebase has been successfully updated to use **EE-TransUNet** (Edge-Enhanced TransUNet) instead of the baseline U-Net. EE-TransUNet is a state-of-the-art model specifically designed for optic disc and cup segmentation with enhanced edge detection capabilities.

## Changes Made

### 1. New Model Files Created

- **`src/models/configs.py`**: Configuration management for different EE-TransUNet variants
- **`src/models/skip.py`**: ResNetV2 backbone implementation for hybrid encoder
- **`src/models/ee_transunet.py`**: Complete EE-TransUNet implementation with:
  - Vision Transformer (ViT) encoder
  - DOConv2d (Depthwise Over-parameterized Convolution)
  - CCF (Cascaded Convolutional Fusion) blocks
  - CSMF (Channel Shuffling Multiple Expansion Fusion) blocks
  - Enhanced decoder with skip connections

### 2. Updated Files

- **`src/main.py`**: 
  - Replaced U-Net with EE-TransUNet
  - Updated command-line arguments for model configuration
  - Removed CLAHE preprocessing (not used in original EE-TransUNet)
  - Updated default hyperparameters (lr=0.01, epochs=150)

- **`requirements.txt`**:
  - Added `ml-collections==0.1.1` for configuration management
  - Added `scipy==1.14.1` for pretrained weight loading

## Model Variants Available

The following EE-TransUNet variants are available:

| Model Name | Description | Parameters | Best For |
|-----------|-------------|------------|----------|
| `R50-ViT-B_16` | ResNet50 + ViT-Base/16 (Default) | ~102M | Best balance of performance and speed |
| `ViT-B_16` | ViT-Base/16 | ~86M | Pure transformer without ResNet |
| `ViT-B_32` | ViT-Base/32 | ~88M | Faster inference, larger patches |
| `R50-ViT-L_16` | ResNet50 + ViT-Large/16 | ~307M | Highest accuracy, more compute |
| `ViT-L_16` | ViT-Large/16 | ~304M | Pure large transformer |
| `testing` | Minimal config | Tiny | For debugging only |

## Training the Model

### Basic Training (Recommended)

```bash
python src/main.py \
    --model_name R50-ViT-B_16 \
    --img_size 512 \
    --batch_size 8 \
    --epochs 150 \
    --lr 0.01 \
    --save_dir experiments/ee_transunet
```

### Training with Pretrained Weights

EE-TransUNet benefits greatly from ImageNet21k pretrained weights. Download the pretrained weights:

**For R50-ViT-B_16 (Recommended):**
- Download from: https://console.cloud.google.com/storage/browser/vit_models/imagenet21k
- File: `imagenet21k_R50+ViT-B_16.npz` (~375MB)
- Place in: `./imagenet21k_R50+ViT-B_16.npz`

```bash
python src/main.py \
    --model_name R50-ViT-B_16 \
    --pretrained_path ./imagenet21k_R50+ViT-B_16.npz \
    --img_size 512 \
    --batch_size 8 \
    --epochs 150 \
    --lr 0.01
```

**Alternative Pretrained Weights Sources:**
1. Original EE-TransUNet weights: https://pan.baidu.com/s/1IQsiUMnmRIYzJ2qI1o0OPQ?pwd=gwcy
2. Google Vision Transformer: https://github.com/google-research/vision_transformer

### Advanced Training Options

```bash
python src/main.py \
    --model_name R50-ViT-B_16 \
    --img_size 512 \
    --batch_size 8 \
    --epochs 150 \
    --lr 0.01 \
    --weight_decay 1e-4 \
    --n_skip 3 \
    --lambda_dice 0.5 \
    --lambda_bce 0.5 \
    --save_dir experiments/ee_transunet \
    --save_freq 10 \
    --num_workers 4 \
    --device cuda
```

## Command-Line Arguments

### Model Arguments
- `--model_name`: Model variant (default: `R50-ViT-B_16`)
- `--n_skip`: Number of skip connections (default: 3)
- `--vit_patches_size`: Vision Transformer patch size (default: 16)
- `--pretrained_path`: Path to pretrained weights (.npz file)

### Training Arguments
- `--epochs`: Number of training epochs (default: 150)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.01)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--img_size`: Input image size (default: 512)

### Loss Arguments
- `--lambda_dice`: Weight for Dice loss (default: 0.5)
- `--lambda_bce`: Weight for BCE loss (default: 0.5)

### Data Arguments
- `--data_dir`: REFUGE dataset directory
- `--train_csv`: Training CSV file path
- `--val_csv`: Validation CSV file path

## Key Differences from U-Net

### Architecture
1. **Hybrid Encoder**: Combines ResNet50 CNN with Vision Transformer
2. **Self-Attention**: Global context modeling via transformer blocks
3. **Enhanced Skip Connections**: CSMF blocks with channel shuffling
4. **Edge Enhancement**: CCF blocks for better boundary detection
5. **DOConv**: Over-parameterized convolutions for better feature extraction

### Training
1. **No CLAHE**: EE-TransUNet doesn't use CLAHE preprocessing
2. **Higher Learning Rate**: Default 0.01 vs 1e-4 for U-Net
3. **More Epochs**: 150 epochs recommended vs 100 for U-Net
4. **Pretrained Weights**: Can leverage ImageNet21k pretraining

### Performance
- Better edge detection for optic cup boundaries
- Improved CDR (Cup-to-Disc Ratio) estimation
- Higher Dice coefficients on validation data
- More parameters but better accuracy

## Installation

1. Install the new dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Download pretrained weights as described above

## Expected Results

Based on the original EE-TransUNet paper, you should expect:

- **Optic Disc Dice**: ~0.97-0.98
- **Optic Cup Dice**: ~0.88-0.90
- **Mean Dice**: ~0.92-0.94
- **CDR MAE**: <0.05

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 4 or 2
- Reduce `--img_size` to 384
- Use `ViT-B_16` instead of `R50-ViT-B_16`

### Slow Training
- Increase `--num_workers`
- Use smaller model: `ViT-B_32`
- Enable mixed precision training (requires code modification)

### Poor Performance
- Use pretrained weights
- Increase training epochs to 150+
- Ensure data augmentation is working
- Check learning rate (try 0.005 if 0.01 is too high)

## References

- **EE-TransUNet Repository**: https://github.com/wangyunyuwyy/EE-TransUNet
- **TransUNet Paper**: https://arxiv.org/abs/2102.04306
- **Vision Transformer**: https://arxiv.org/abs/2010.11929

## Support

For issues specific to the integration, check:
1. All dependencies installed: `pip list | grep -E "ml-collections|scipy"`
2. Model files exist in `src/models/`
3. CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

For EE-TransUNet-specific questions, contact the original authors or create an issue on their repository.
