# ROI Enhancer for Optic Disc and Cup Segmentation
**Task-Aware Multi-Scale Feature Learning for Glaucoma Detection**

Sebastian Siedler & Steffen Ullmann  
CAP5410 - Computer Vision  
Florida Polytechnic University

---

## Presentation Outline

1. **Problem & Motivation**
2. **Research Question**
3. **Related Work & Gap**
4. **Dataset & Preprocessing**
5. **Methodology: 5 Approaches**
6. **Architecture Deep Dive**
7. **Training Strategy**
8. **Results & Analysis**
9. **Code Structure & Implementation**
10. **Discussion & Conclusions**

---

## 1. Problem & Motivation

### Glaucoma: A Silent Threat
- Leading cause of **irreversible blindness** worldwide
- Progresses **without noticeable symptoms**
- Early detection is critical for treatment

### Cup-to-Disc Ratio (CDR)
- Key diagnostic metric: ratio of Optic Cup (OC) to Optic Disc (OD)
- Requires accurate segmentation of both structures in fundus images

### Challenges
- **Low contrast** in fundus images
- **Uneven illumination**
- **Image blur and artifacts**
- **High anatomical variability** (especially cup shape)

---

## 2. Research Question

### Central Question
> **Which is more effective for OD/OC segmentation:**  
> **Learned preprocessing (image enhancement) vs. architectural improvements (multi-scale features)?**

### Comparison Axes
1. **Traditional preprocessing** (CLAHE)
2. **Learned enhancement** (Standard + Multi-scale enhancers)
3. **Architectural integration** (ASPP-UNet)

### Key Metrics
- **Performance**: mIoU, class-specific IoU
- **Efficiency**: Parameter count, training time
- **Convergence**: Training dynamics

---

## 3. Related Work & Gap

### State-of-the-Art Segmentation
- **UNet-based architectures** dominate medical imaging
- **Multi-GlaucNet** (Xiong et al., 2025): Multi-task learning, ResNet50
- **EE-TransUNet** (Liu et al., 2025): Transformer + edge enhancement
- **RMHA-Net** (Zedan et al., 2025): ASPP + Hybrid Attention → **Best SOTA**

### Image Enhancement Techniques
- **Traditional**: CLAHE (widely used, simple)
- **Learned**: CycleGAN variants (You et al., 2019 - Cycle-CBAM)
- Problem: Most enhancers trained **separately** from analysis task

### **The Gap**
> Enhancement methods optimize for **visual quality**, not **task-specific features**  
> → Our solution: **Joint end-to-end training** of enhancement + segmentation

---

## 4. Dataset & Preprocessing

### Combined Dataset (2,870 images)
1. **G1020**: 1,020 images (private clinical practice)
2. **ORIGA**: 650 images (482 normal, 168 glaucoma)
3. **REFUGE**: 1,200 images (challenge dataset)

### Preprocessing
- Images **pre-cropped** to ROI centered on optic disc
- Mask classes: 0=background, 1=disc, 2=cup
- **Filtered** 234 images with incomplete masks

### Data Split (70/15/15)
- **Training**: 1,845 images
- **Validation**: 395 images
- **Test**: 396 images

### Augmentation
- Random flip (H/V, p=0.5)
- Rotation (±20°)
- Brightness/contrast (±0.2)
- ImageNet normalization

---

## 5. Methodology: 5 Approaches

### Our Comparison Framework

| # | Approach | Type | Parameters | Key Feature |
|---|----------|------|------------|-------------|
| 1 | **Baseline UNet** | Single-phase | 31.0M | Standard encoder-decoder |
| 2 | **CLAHE + UNet** | Traditional preprocessing | 31.0M | Histogram equalization |
| 3 | **Std Enhancer + UNet** | Learned preprocessing | 31.1M | Simple encoder-decoder |
| 4 | **Atrous Enhancer + UNet** | Multi-scale preprocessing | 31.0M | ASPP-based enhancer |
| 5 | **ASPP-UNet** | Architectural integration | **23.5M** | Multi-scale bottleneck |

### Design Rationale
- Isolate **preprocessing** vs. **architecture** contributions
- Fair comparison: same base UNet for approaches 1-4
- Efficiency test: fewer parameters, better performance?

---

## 6. Architecture Deep Dive

### 6.1 Baseline UNet
```
Encoder: 64 → 128 → 256 → 512 → 1024 channels
         ↓     ↓     ↓     ↓      ↓
Decoder: Skip connections (concatenation)
         ↓     ↓     ↓     ↓      ↓
Output:  3 classes (background, disc, cup)

Bottleneck: Double Conv (1024 channels)
Parameters: 31.0M
```

### 6.2 CLAHE Preprocessing
- **C**ontrast **L**imited **A**daptive **H**istogram **E**qualization
- Traditional computer vision technique
- Applied **before** UNet input
- No additional parameters

---

### 6.3 Standard Enhancer (52K params)
```
Input (256×256×3)
    ↓
Encoder: 64 → 128 → 256 (max pooling)
    ↓
Decoder: 256 → 128 → 64 (transposed conv)
    ↓
Residual Connection: α·Enhanced + β·Original
    ↓
UNet (frozen in Phase 1, joint in Phase 2)
```

**Key Features:**
- Lightweight (52K parameters)
- Learnable residual weights (α, β)
- Two-phase training

---

### 6.4 Atrous Enhancer (26K params)
```
Input (256×256×3)
    ↓
ASPP Module (dilation rates: [1, 3, 6])
├─ 1×1 conv (local features)
├─ 3×3 atrous conv (rate=3)
├─ 3×3 atrous conv (rate=6)
└─ Global avg pooling
    ↓ (concatenate)
1×1 projection → Residual connection
    ↓
UNet (frozen in Phase 1, joint in Phase 2)
```

**Key Features:**
- Multi-scale without downsampling
- Fewer parameters than standard enhancer
- Preserves spatial resolution

---

### 6.5 ASPP-UNet (23.5M params)
```
Encoder: 64 → 128 → 256 → 512
         ↓     ↓     ↓     ↓
    ASPP Bottleneck (1024 channels)
    ├─ 1×1 conv
    ├─ 3×3 atrous (rate=6)
    ├─ 3×3 atrous (rate=12)
    ├─ 3×3 atrous (rate=18)
    └─ Global pooling
         ↓ (concatenate & project)
Decoder: Skip connections
         ↓
Output: 3 classes
```

**Why Fewer Parameters?**
- ASPP module more efficient than double conv bottleneck
- 24% reduction: 23.5M vs. 31.0M

---

## 7. Training Strategy

### Single-Phase Training (UNet Variants)
- **Baseline, CLAHE, ASPP-UNet**
- Adam optimizer: lr=1e-4, β=(0.9, 0.999)
- Batch size: 16
- ReduceLROnPlateau: factor=0.5, patience=5
- **Baseline/CLAHE**: 100 epochs (no early stopping)
- **ASPP-UNet**: 67 epochs (early stopping, patience=15)

### Two-Phase Training (Enhancer Variants)
**Phase 1: Train Enhancer Only**
- UNet frozen with pretrained weights
- Standard: 26 epochs (early stop from 30)
- Atrous: 50 epochs (no early stop)

**Phase 2: Joint Fine-tuning**
- Both modules trainable
- Lower learning rate: 1e-5
- Standard: 20 epochs
- Atrous: 25 epochs (early stop from 30)

---

### Loss Functions

**Segmentation Loss:**
$$L_{\text{seg}} = 0.5 \cdot L_{\text{CE}} + 0.5 \cdot L_{\text{Dice}}$$

**Cross-Entropy (weighted):**
$$L_{\text{CE}} = -\sum_{c=1}^C w_c \sum_i y_{i,c} \log(\hat{y}_{i,c})$$
- Class weights: [1.0, 1.0, 2.0] (emphasize cup)

**Dice Loss:**
$$L_{\text{Dice}} = 1 - \frac{2\sum_i y_i \hat{y}_i + \epsilon}{\sum_i y_i + \sum_i \hat{y}_i + \epsilon}$$
- ε = 1 for numerical stability

**Enhancement Loss (Enhancer approaches):**
$$L_{\text{total}} = L_{\text{seg}} + \lambda \cdot L_{\text{L1}}$$
$$L_{\text{L1}} = \frac{1}{N} \sum_i |I_{\text{enhanced}}^{(i)} - I_{\text{original}}^{(i)}|$$
- λ = 0.001 (minimal deviation)

---

## 8. Results & Analysis

### Quantitative Comparison (Test Set)

| Approach | mIoU | Δ mIoU | IoU BG | IoU Disc | IoU Cup | Parameters |
|----------|------|--------|--------|----------|---------|------------|
| **Baseline UNet** | 90.86% | --- | 97.85% | 90.74% | 84.00% | 31.0M |
| CLAHE | 90.79% | -0.07% | 97.83% | 90.65% | 83.89% | 31.0M |
| + Std Enhancer | 91.12% | **+0.26%** | 98.00% | 90.91% | 84.46% | 31.1M |
| + Atrous Enhancer | 90.85% | -0.01% | 97.83% | 90.79% | 83.93% | 31.0M |
| **ASPP-UNet** | **91.72%** | **+0.86%** | **98.14%** | **91.42%** | **85.59%** | **23.5M** |

### Key Findings
✅ **ASPP-UNet wins**: +0.86 pp improvement  
✅ **24% fewer parameters** (23.5M vs. 31.0M)  
✅ **Faster convergence**: 67 vs. 100 epochs (33% reduction)  
❌ **CLAHE hurts performance**: -0.07 pp  
⚠️ **Enhancers provide minimal benefit**: +0.26% (std) to -0.01% (atrous)

---

### Training Convergence Analysis

#### Single-Phase Convergence
- **Baseline UNet**: 100 epochs, convergence at epoch ~85
- **CLAHE-UNet**: 100 epochs, convergence at epoch ~95
- **ASPP-UNet**: **67 epochs** (early stop), convergence at epoch ~40

#### Two-Phase Convergence
- **Standard Enhancer**: 
  - Phase 1: 26 epochs (42% loss reduction)
  - Phase 2: 20 epochs (21% additional reduction)
  - **Total: 46 epochs**
  
- **Atrous Enhancer**:
  - Phase 1: 50 epochs (57% loss reduction)
  - Phase 2: 25 epochs (13% additional reduction)
  - **Total: 75 epochs**

**Insight**: Multi-scale atrous enhancer requires longer training but doesn't deliver better performance → architectural complexity ≠ better results

---

### Class-Specific Performance

**Improvement over Baseline (Δ IoU)**

| Approach | Δ BG | Δ Disc | Δ Cup (hardest) |
|----------|------|--------|------------------|
| CLAHE | -0.02% | -0.09% | **-0.11%** |
| + Std Enhancer | +0.15% | +0.17% | **+0.46%** |
| + Atrous Enhancer | -0.02% | +0.05% | **-0.07%** |
| **ASPP-UNet** | **+0.29%** | **+0.68%** | **+1.59%** |

**Key Observations:**
- **Cup segmentation** (most challenging): ASPP-UNet shows largest gain (+1.59%)
- All methods achieve >97% on background
- ASPP-UNet: consistent improvements across **all classes**

---

### Qualitative Results

#### Sample 160: Frayed Edges
![Sample 160](../results/sample_comparison_160.png)
- **ASPP-UNet**: Smoother, anatomically plausible boundaries
- **Baseline**: Frayed edges, irregular disc/cup contours
- Multi-scale features capture fine vessel structures

#### Sample 13: Low Contrast Challenge
![Sample 13](../results/sample_comparison_13.png)
- Both models handle poor visibility well
- ASPP-UNet edges slightly cleaner
- Robust to challenging imaging conditions

#### Sample 52 & 56: Extreme Cases
- **Very large OD/OC**: Both models struggle
- Potential issue: 256×256 resolution limitation
- Future work: higher resolution training

---

### Enhancement Visualization

#### Standard Enhancer (Sample 218)
![Std Enhancer](../results/std_enhancer_top5_sample_218.png)
- Lightens background and vessels
- Less modification on disc/cup regions
- **Checkerboard pattern** in difference heatmap → localized adjustments

#### Atrous Enhancer (Sample 391)
![Atrous Enhancer](../results/atrous_enhancer_top1_sample_391.png)
- Stronger background/vessel changes
- No checkerboard pattern → different enhancement strategy
- Despite multi-scale design, performance not better than standard

**Insight**: Enhancers modify task-irrelevant regions (background, vessels) more than critical disc/cup boundaries

---

## 9. Code Structure & Implementation

### Project Organization
```
CAP5410-roi-enhancer-odoc/
├── src/
│   ├── models/
│   │   ├── unet.py              # Baseline UNet
│   │   ├── aspp_unet.py         # ASPP-UNet
│   │   ├── enhancer.py          # Standard enhancer
│   │   └── atrous_enhancer.py   # Multi-scale enhancer
│   ├── data_loader/
│   │   ├── dataset.py           # GlaucomaDataset class
│   │   └── transforms.py        # Augmentation pipeline
│   ├── training/
│   │   ├── train.py             # Single-phase training
│   │   ├── train_enhancer.py    # Two-phase training
│   │   └── train_utils.py       # Loss functions, metrics
│   └── utils/
├── notebooks/
│   ├── train_unet.ipynb
│   ├── train_aspp_unet.ipynb
│   ├── train_enhancer.ipynb
│   └── test_comparison.ipynb
├── checkpoints/                  # Model weights
├── results/                      # Training histories, visualizations
└── datasets/                     # G1020, ORIGA, REFUGE
```

---

### Key Implementation Details

#### 1. Dataset Class (`dataset.py`)
```python
class GlaucomaDataset(Dataset):
    """
    Loads G1020, ORIGA, REFUGE datasets
    - 70/15/15 train/val/test split
    - Filters incomplete masks
    - Stratified split for glaucoma/normal cases
    """
```

**Features:**
- Unified interface for 3 datasets
- Cached splits (reproducible with seed=42)
- Optional incomplete mask filtering

---

#### 2. UNet Architecture (`unet.py`)
```python
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, base_channels=64):
        # Encoder: DoubleConv → MaxPool (4 levels)
        # Decoder: TransposeConv + Skip connections
        # Output: 1×1 Conv to n_classes
```

**Building Blocks:**
- `DoubleConv`: Conv → BatchNorm → ReLU (×2)
- `Down`: MaxPool → DoubleConv
- `Up`: TransposeConv → Concatenate → DoubleConv

---

#### 3. ASPP Module (`aspp_unet.py`)
```python
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 dilation_rates=[1, 6, 12, 18]):
        # Parallel branches:
        # - 1×1 conv (local)
        # - 3×3 atrous convs (multi-scale)
        # - Global average pooling
        # Final: Concatenate → 1×1 projection → Dropout
```

**Key Innovation:**
- Replaces UNet's double conv bottleneck
- Multi-scale features without spatial downsampling
- More parameter-efficient (23.5M vs. 31.0M)

---

#### 4. Enhancer Architectures

**Standard Enhancer (`enhancer.py`):**
```python
class ImageEnhancer(nn.Module):
    def __init__(self, n_channels=3, base_channels=32, num_levels=3):
        # Lightweight encoder-decoder
        # Residual connection: α·Enhanced + β·Original
```

**Atrous Enhancer (`atrous_enhancer.py`):**
```python
class AtrousImageEnhancer(nn.Module):
    def __init__(self, n_channels=3, base_channels=64):
        # ASPP-based enhancement
        # Dilation rates: [1, 3, 6]
        # Residual connection
```

---

#### 5. Training Loop (`train.py`, `train_enhancer.py`)

**Single-Phase:**
```python
def train_epoch(model, dataloader, criterion, optimizer):
    for images, masks in dataloader:
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
```

**Two-Phase (Enhancer):**
```python
# Phase 1: Freeze UNet
for param in unet.parameters():
    param.requires_grad = False
train_enhancer(enhancer, unet, ...)

# Phase 2: Joint fine-tuning
for param in unet.parameters():
    param.requires_grad = True
train_joint(enhancer, unet, ...)
```

---

#### 6. Loss Functions (`train.py`)

```python
class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation"""
    
class CombinedLoss(nn.Module):
    """0.5·CrossEntropy + 0.5·Dice"""
    def __init__(self, class_weights=[1.0, 1.0, 2.0]):
        # Emphasize cup class (2x weight)
```

**Evaluation Metrics:**
```python
def calculate_iou(pred, target, num_classes=3):
    """Per-class IoU + mean IoU"""
    
def calculate_dice(pred, target, num_classes=3):
    """Per-class Dice coefficient"""
```

---

#### 7. Data Augmentation (`transforms.py`)

```python
from albumentations import (
    HorizontalFlip, VerticalFlip,
    Rotate, RandomBrightnessContrast,
    Normalize, Compose
)

def get_training_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=20, p=0.5),
        RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
```

---

### Training Workflow Example

**ASPP-UNet Training:**
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Train ASPP-UNet (single-phase)
python src/training/train_aspp_unet.py \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --early_stopping_patience 15

# 3. Evaluate on test set
python src/training/evaluate.py \
    --model aspp_unet \
    --checkpoint checkpoints_aspp_unet/best_model.pth
```

**Enhancer Training:**
```bash
# Phase 1: Train enhancer only
python src/training/train_enhancer.py \
    --phase 1 \
    --enhancer_type atrous \
    --epochs 50

# Phase 2: Joint fine-tuning
python src/training/train_enhancer.py \
    --phase 2 \
    --enhancer_type atrous \
    --epochs 30 \
    --lr 1e-5
```

---

### Hardware & Environment

**Hardware:**
- GPU: NVIDIA GeForce RTX 2080 Ti (11 GB VRAM)
- CPU: Intel Core i7-8700K (3.70 GHz)
- RAM: 32 GB

**Software:**
- PyTorch (deep learning framework)
- Albumentations (data augmentation)
- Python 3.x in virtual environment

**Why Virtual Environment?**
```bash
# Our custom instruction for running Python commands
source .venv/bin/activate && python <command>
```
- Isolated dependencies
- Reproducible environment
- Avoids system-wide package conflicts

---

## 10. Discussion & Conclusions

### Main Finding: Efficiency Through Architecture

**ASPP-UNet Advantages:**
1. **Performance**: +0.86% mIoU (91.72% vs. 90.86%)
2. **Parameters**: 24% reduction (23.5M vs. 31.0M)
3. **Training Time**: 33% faster (67 vs. 100 epochs)
4. **Class-wise**: Largest cup improvement (+1.59%)

**Why It Works:**
- Multi-scale features learned **end-to-end** for segmentation
- ASPP bottleneck provides better inductive bias
- Task-specific optimization vs. generic enhancement

---

### Revisiting Preprocessing

#### CLAHE: Context Matters
- **Our result**: -0.07% decrease
- **Literature (RMHA-Net, Multi-GlaucNet)**: Positive effect

**Possible Explanations:**
1. **Resolution**: 256×256 may reduce CLAHE benefits
2. **Architecture**: UNet's learned normalization may conflict with CLAHE
3. **Context-dependent**: Benefits vary by dataset, model, pipeline

**Takeaway**: CLAHE is not universally beneficial for deep learning pipelines

---

#### Learned Enhancers: Limited Impact

**Standard Enhancer:**
- +0.26% improvement (marginal)
- 52K additional parameters
- Two-phase training complexity
- Performance variance across runs (sometimes negative)

**Atrous Enhancer:**
- -0.01% degradation (despite multi-scale design)
- 26K parameters
- Longer training (75 total epochs)
- Multi-scale preprocessing ≠ better segmentation

**Root Causes:**
1. **Task misalignment**: Enhancers optimize visual quality, not segmentation
2. **Training strategy**: Phase 1 frozen UNet prevents task-specific learning
3. **Modification patterns**: Enhancers change backgrounds/vessels, not disc/cup

---

### Comparison with State-of-the-Art

**Our Best Result (ASPP-UNet):**
- mIoU: 91.72%
- IoU Disc: 91.42%, IoU Cup: 85.59%

**SOTA Comparisons:**
- **RMHA-Net** (Zedan et al., 2025): Multi-scale + attention, uses CLAHE
- **EE-TransUNet** (Liu et al., 2025): Dice 0.967 (OD), 0.906 (OC) on REFUGE
- **Multi-GlaucNet** (Xiong et al., 2025): 96.7% accuracy, uses CLAHE

**Our Contribution:**
- Focus on **efficiency** (fewer parameters, faster training)
- Systematic **ablation study** (5 approaches)
- Challenge **preprocessing paradigm** (architecture > enhancement)
- Correlates with EE-TransUNet finding: architectural improvements > separate preprocessing

---

### Limitations

1. **Resolution Constraint**
   - Trained at 256×256 (hardware limitation)
   - Extreme large OD/OC cases struggle
   - Future: 512×512 or higher

2. **Dataset Quality**
   - Annotation ambiguity in challenging cases
   - Indistinct cup boundaries
   - Future: Incorporate uncertainty weighting (REFUGE inter-annotator agreement)

3. **Extreme CDR Cases**
   - Cup-to-disc ratio → 1.0 (severe glaucoma)
   - Both baseline and ASPP-UNet show reduced accuracy
   - Future: Specialized loss weighting or dedicated branches

4. **Generalization**
   - Findings specific to OD/OC segmentation
   - Need validation on other medical imaging tasks

---

### Key Insights & Contributions

#### 1. **Architecture > Preprocessing**
- Task-specific architectural integration outperforms learned enhancement
- End-to-end multi-scale learning more effective than separate enhancement step

#### 2. **Efficiency Matters**
- ASPP-UNet: Better performance with 24% fewer parameters
- Faster convergence (33% training time reduction)
- Practical for deployment on resource-constrained devices

#### 3. **CLAHE Context-Dependent**
- Contradicts some literature findings
- Benefits depend on architecture, dataset, resolution
- Not a universal preprocessing solution for deep learning

#### 4. **Multi-Scale Design Location**
- Multi-scale in preprocessing (atrous enhancer): No benefit
- Multi-scale in architecture (ASPP-UNet): Clear benefit
- **Where** you apply multi-scale matters more than **if** you apply it

---

### Future Work

1. **Higher Resolution Training**
   - 512×512 or adaptive resolution
   - Better handling of large OD/OC cases

2. **Uncertainty-Aware Training**
   - Leverage REFUGE annotation agreement
   - Weight loss by inter-annotator consensus

3. **Severe Glaucoma Focus**
   - Specialized handling for high CDR cases
   - Adaptive loss weighting

4. **Generalization Studies**
   - Test on other medical segmentation tasks
   - Validate architecture > preprocessing hypothesis

5. **Attention Mechanisms**
   - Combine ASPP with channel/spatial attention (like RMHA-Net)
   - Potential for further improvement

---

## Summary

### Research Question
> **Learned preprocessing vs. architectural improvements for OD/OC segmentation?**

### Answer
✅ **Architectural integration (ASPP-UNet) wins**
- +0.86% mIoU improvement
- 24% fewer parameters
- 33% faster training

❌ **Preprocessing approaches provide minimal benefit**
- CLAHE: -0.07%
- Standard enhancer: +0.26%
- Atrous enhancer: -0.01%

### Broader Impact
- Challenges preprocessing-first paradigm in medical imaging
- Demonstrates efficiency through architectural design
- Provides systematic comparison framework for future work

---

## Questions & Discussion

### Discussion Points
1. Why do you think CLAHE hurt our performance while helping SOTA models?
2. Should we explore attention mechanisms as next step?
3. Is 256×256 resolution sufficient, or should we prioritize higher resolution?
4. How can we better handle annotation uncertainty in training?

### Thank You!

**Code Available:** https://github.com/SebastianSiedler/CAP5410-roi-enhancer-odoc

**Contact:**
- Sebastian Siedler: ssiedler3117@floridapoly.edu
- Steffen Ullmann: sullmann3121@floridapoly.edu
