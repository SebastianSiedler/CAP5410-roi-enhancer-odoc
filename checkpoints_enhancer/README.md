# Image Enhancer Training

This directory contains checkpoints for the two-phase Image Enhancer + UNet training.

## Architecture

The Image Enhancer is a lightweight encoder-decoder network (3-4 levels) designed to preprocess fundus images before segmentation. Key features:

- **Shallow architecture**: 3-4 levels vs UNet's 5 levels (~10-20% of UNet's parameters)
- **Residual connection**: Input → Output to preserve original content
- **L1 regularization**: Prevents over-manipulation of images

## Training Strategy

### Phase 1: Enhancer Only (UNet Frozen)
- Train enhancer with frozen pretrained UNet
- Learning rate: 1e-4
- Epochs: 30 (with early stopping)
- Loss: Segmentation loss + L1 regularization

### Phase 2: Joint Fine-tuning
- Unfreeze UNet and train both models together
- Learning rate: 1e-5 (10x lower)
- Epochs: 20 (with early stopping)
- Loss: Same as Phase 1

## Checkpoints

- `phase1_best_model.pth`: Best model from Phase 1
- `phase1_checkpoint_epoch_*.pth`: Phase 1 periodic checkpoints
- `phase2_best_model.pth`: Best model from Phase 2 (final model)
- `phase2_checkpoint_epoch_*.pth`: Phase 2 periodic checkpoints

## Data Pipeline

**Important**: Uses original images WITHOUT CLAHE preprocessing!
- Same augmentations as UNet training
- Image size: 256x256
- Batch size: 16

## Usage

See `notebooks/train_enhancer.ipynb` for complete training pipeline.

To load a trained model:

```python
from models.enhancer import ImageEnhancer
from models.unet import UNet
from training.train_enhancer import EnhancerUNetModel

# Create models
enhancer = ImageEnhancer(n_channels=3, base_channels=32, num_levels=3)
unet = UNet(n_channels=3, n_classes=3, base_channels=64)
model = EnhancerUNetModel(enhancer, unet)

# Load checkpoint
checkpoint = torch.load('checkpoints_enhancer/phase2_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
enhanced, segmentation = model(image)
```

## Results

See `results/enhancer_comparison.json` for quantitative comparison between:
- UNet only
- UNet + Enhancer

Both evaluated on validation set without CLAHE.
