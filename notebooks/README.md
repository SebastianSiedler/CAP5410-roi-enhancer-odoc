# Notebook Organization

This directory contains Jupyter notebooks for training and evaluating different approaches to optic disc/cup segmentation.

## Notebooks Overview

### 1. `train_enhancer.ipynb` - Standard Image Enhancer
**Purpose**: Train a simple encoder-decoder image enhancer with two-phase training

**What it does**:
- Creates a lightweight 3-level encoder-decoder enhancer
- Phase 1: Trains enhancer only (UNet frozen)
- Phase 2: Fine-tunes both enhancer and UNet jointly
- Compares UNet alone vs UNet + Enhancer

**Use this for**:
- Testing if learned image enhancement helps segmentation
- Baseline comparison for enhancement approaches
- Understanding two-phase training methodology

**Expected runtime**: ~1-2 hours (50 total epochs)

---

### 2. `train_atrous_enhancer.ipynb` - Multi-Scale Atrous Enhancer
**Purpose**: Train an ASPP-based image enhancer with multi-scale dilated convolutions

**What it does**:
- Creates atrous enhancer with ASPP module (dilation rates: [1, 3, 6])
- Multi-scale enhancement: fine vessels, disc boundaries, global illumination
- Two-phase training: enhancer only → joint fine-tuning
- Compares with baseline UNet

**Use this for**:
- Testing if multi-scale enhancement outperforms standard enhancement
- Evaluating atrous convolutions for preprocessing
- Understanding ASPP module behavior

**Expected runtime**: ~2-3 hours (80 total epochs)

**Key difference from standard enhancer**: Captures features at multiple scales simultaneously without losing resolution

---

### 3. `train_aspp_unet.ipynb` - ASPP-UNet Architecture
**Purpose**: Train UNet with ASPP bottleneck (architectural improvement)

**What it does**:
- Creates ASPP-UNet: UNet with Atrous Spatial Pyramid Pooling in bottleneck
- Trains from scratch (no pretrained UNet)
- Multi-scale feature extraction during segmentation
- Compares with baseline UNet

**Use this for**:
- **Recommended main approach** - likely best performance
- Direct architectural improvement vs preprocessing
- Testing if multi-scale features help during segmentation
- More efficient model (fewer parameters than standard UNet!)

**Expected runtime**: ~3-4 hours (100 epochs from scratch)

**Key advantage**: Multi-scale features learned specifically for segmentation task, not generic preprocessing

---

### 4. `train_unet.ipynb` - Baseline UNet Training
**Purpose**: Train standard UNet (baseline model)

**What it does**:
- Standard UNet architecture with 5 levels
- Combined CE + Dice loss
- Class weights for cup segmentation
- Basic augmentation pipeline

**Use this for**:
- Training baseline model for comparison
- Understanding basic UNet training
- Generating pretrained UNet for enhancer approaches

**Already trained**: Checkpoint available at `../checkpoints/best_model.pth`

---

## Recommended Execution Order

### Option A: Quick Comparison (Recommended)

1. **Verify baseline** (optional, if checkpoint exists):
   - Baseline UNet already trained at `checkpoints/best_model.pth`
   
2. **Train ASPP-UNet** (`train_aspp_unet.ipynb`):
   - Most promising approach
   - Direct architectural improvement
   - Run this first to get best results

3. **Compare results**: Check if ASPP-UNet improves over baseline

### Option B: Comprehensive Experiment

Run all notebooks to compare all approaches:

1. **`train_unet.ipynb`** (if needed):
   - Train baseline UNet
   - Get pretrained checkpoint

2. **`train_enhancer.ipynb`**:
   - Standard enhancement approach
   - Establishes enhancement baseline

3. **`train_atrous_enhancer.ipynb`**:
   - Multi-scale enhancement
   - Compare with standard enhancer

4. **`train_aspp_unet.ipynb`**:
   - Architectural improvement
   - Compare with all preprocessing approaches

5. **Final comparison**: Analyze all results together

---

## Expected Results Ranking

Based on similar experiments in medical imaging:

1. 🥇 **ASPP-UNet** (best expected)
   - Multi-scale features during segmentation
   - Task-specific optimization
   - Proven approach in semantic segmentation

2. 🥈 **Standard UNet** (strong baseline)
   - Well-optimized architecture
   - Good performance on original images

3. 🥉 **UNet + Atrous Enhancer** (preprocessing)
   - Multi-scale enhancement
   - May not align with segmentation needs

4. 🔻 **UNet + Standard Enhancer** (preprocessing)
   - Generic enhancement
   - Likely marginal or negative impact

---

## Outputs and Results

### Checkpoints Saved To:
- `../checkpoints/` - Standard UNet
- `../checkpoints_enhancer/` - Standard enhancer
- `../checkpoints_atrous_enhancer/` - Atrous enhancer
- `../checkpoints_aspp_unet/` - ASPP-UNet

### Results Saved To:
- `../results/enhancer_comparison.json` - Standard enhancer results
- `../results/atrous_enhancer_comparison.json` - Atrous enhancer results
- `../results/aspp_unet_comparison.json` - ASPP-UNet results
- Training history plots and JSONs for each approach

---

## Quick Start Guide

### To run ASPP-UNet (recommended):

```bash
cd /home/robolab/dev/CAP5410-roi-enhancer-odoc
source .venv/bin/activate
jupyter notebook notebooks/train_aspp_unet.ipynb
```

Then run all cells in order.

### To run Atrous Enhancer:

```bash
jupyter notebook notebooks/train_atrous_enhancer.ipynb
```

### To run Standard Enhancer:

```bash
jupyter notebook notebooks/train_enhancer.ipynb
```

---

## GPU Memory Requirements

All notebooks designed to run on single GPU with ~10GB VRAM:
- Batch size: 16
- Image size: 256×256
- Mixed precision available if needed

If GPU OOM occurs:
- Reduce batch size to 8
- Reduce num_workers to 0
- Clear GPU memory between notebooks

---

## Key Differences Summary

| Notebook | Approach | Model Type | Parameters | Training Time | Expected Performance |
|----------|----------|-----------|------------|---------------|---------------------|
| train_unet.ipynb | Baseline | Standard UNet | 31.0M | ~3-4h | Strong baseline |
| train_enhancer.ipynb | Preprocessing | Enhancer + UNet | 31.1M | ~1-2h | Similar to baseline |
| train_atrous_enhancer.ipynb | Preprocessing | Atrous Enh + UNet | 31.0M | ~2-3h | Similar to baseline |
| train_aspp_unet.ipynb | **Architecture** | **ASPP-UNet** | **21.5M** | **~3-4h** | **Best expected** |

---

## Scientific Insights

### Why separate notebooks?

1. **Modularity**: Each approach independently executable
2. **Debugging**: Easier to troubleshoot specific methods
3. **Comparison**: Clear separation for fair comparison
4. **Reproducibility**: Each experiment fully self-contained
5. **Flexibility**: Can run only the approaches you need

### What to report in paper?

**Minimal comparison**:
- Standard UNet (baseline)
- ASPP-UNet (your contribution)
- Show multi-scale features improve performance

**Full comparison**:
- All four approaches
- Show preprocessing doesn't help but architecture does
- Valuable negative results (enhancers don't improve)
- Demonstrates importance of task-specific design

---

## Tips for Success

1. **Start with ASPP-UNet**: Most likely to succeed
2. **Monitor training**: Check convergence, validation metrics
3. **Save frequently**: Checkpoints every 10 epochs
4. **Compare fairly**: Same data, same hyperparameters where possible
5. **Visualize results**: Look at predictions, not just numbers
6. **Document findings**: Both positive and negative results valuable

---

## Getting Help

If you encounter issues:

1. **Check GPU memory**: `nvidia-smi` to monitor usage
2. **Verify data paths**: Ensure datasets accessible
3. **Check dependencies**: All required packages installed
4. **Review logs**: Training output for errors
5. **Compare configs**: Ensure fair comparison settings

---

## Next Steps After Training

1. **Test set evaluation**: Best model on held-out test set
2. **Ablation studies**: Different dilation rates, architectures
3. **Hyperparameter tuning**: Learning rates, batch sizes
4. **Visualization**: Attention maps, feature visualizations
5. **Paper writing**: Document findings and insights

Good luck with your experiments! 🚀
