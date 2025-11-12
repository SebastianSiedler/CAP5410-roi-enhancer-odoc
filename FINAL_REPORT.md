# Task-Aware Multi-Scale Feature Learning for Optic Disc and Cup Segmentation in Fundus Images

**Sebastian Siedler and Steffen Ullmann**  
*Florida Polytechnic University*  
*CAP5410 - Advanced Computer Vision*  
*Fall 2025*

---

## Abstract

Glaucoma is a leading cause of irreversible blindness worldwide, with diagnosis heavily relying on the Cup-to-Disc Ratio (CDR) derived from retinal fundus images. Accurate segmentation of the optic disc (OD) and optic cup (OC) remains challenging due to low contrast, anatomical variability, and subtle boundaries—particularly for the optic cup. While deep learning approaches have shown promise, most methods employ task-agnostic preprocessing techniques that improve general image quality without considering the specific requirements of the downstream segmentation task. This work investigates whether learned image enhancement can improve segmentation performance and compares preprocessing-based approaches against architectural improvements. We evaluate four distinct methodologies: (1) standard UNet baseline, (2) learned image enhancement with a lightweight encoder-decoder, (3) multi-scale enhancement using atrous convolutions, and (4) architectural integration of multi-scale features via Atrous Spatial Pyramid Pooling (ASPP) within the UNet bottleneck. Comprehensive experiments on the G1020, ORIGA, and REFUGE datasets demonstrate that **ASPP-UNet achieves the best performance with 85.09% mean IoU**, representing a 0.98% improvement over the 84.27% baseline while reducing parameters by 30.9%. Critically, both learned enhancement approaches showed marginal degradation (-0.11% and -0.33% respectively), revealing that task-specific architectural modifications outperform preprocessing-based strategies for medical image segmentation. These findings have important implications for the design of segmentation systems, suggesting that multi-scale feature learning should be integrated directly into the segmentation architecture rather than applied as a preprocessing step.

**Keywords:** Medical image segmentation, optic disc, optic cup, glaucoma, UNet, ASPP, multi-scale learning, deep learning

---

## I. Introduction

### A. Clinical Motivation

Glaucoma affects over 80 million people worldwide and is projected to impact 111.8 million individuals by 2040 [1]. As the second leading cause of blindness globally, early detection and monitoring of glaucoma progression are critical for preventing irreversible vision loss. The Cup-to-Disc Ratio (CDR), defined as the ratio of the optic cup diameter to the optic disc diameter, serves as a key diagnostic indicator. A CDR exceeding 0.3 is generally considered abnormal and may indicate glaucomatous damage [2].

Manual assessment of CDR by ophthalmologists is time-consuming, subjective, and prone to inter-observer variability. Automated segmentation of the optic disc and cup from fundus images can provide consistent, reproducible measurements to assist in glaucoma screening and diagnosis, particularly in resource-limited settings where specialist access is restricted.

### B. Technical Challenges

Accurate optic cup segmentation presents several fundamental challenges:

1. **Low Contrast Boundaries**: The cup boundary exhibits subtle intensity variations that are difficult to distinguish from surrounding tissue
2. **Anatomical Variability**: Cup size and shape vary significantly across patients, from shallow cups in healthy individuals to deep excavations in glaucomatous eyes
3. **Image Quality**: Fundus photographs often suffer from non-uniform illumination, noise, and varying acquisition conditions
4. **Class Imbalance**: The optic cup occupies a much smaller region than the disc, leading to severe class imbalance

### C. Current Approaches and Limitations

State-of-the-art methods typically employ a two-stage pipeline:
1. **Localization**: Detect the optic disc region using classification or detection networks
2. **Segmentation**: Segment the disc and cup within a cropped Region of Interest (ROI)

Within the segmentation stage, most approaches apply **task-agnostic preprocessing** such as Contrast Limited Adaptive Histogram Equalization (CLAHE), histogram matching, or fixed image enhancement operations [3, 4]. While these techniques improve general image quality, they are not optimized for the specific requirements of the segmentation task and may even enhance irrelevant features or suppress diagnostically important information.

### D. Research Questions

This work addresses a critical gap in the literature by investigating:

**RQ1:** Can learned image enhancement, trained jointly with the segmentation network, improve performance compared to fixed preprocessing or no enhancement?

**RQ2:** Does multi-scale feature extraction during enhancement (via atrous convolutions) provide benefits over single-scale enhancement?

**RQ3:** Is it more effective to integrate multi-scale features directly into the segmentation architecture or to apply them as a preprocessing step?

### E. Contributions

Our key contributions are:

1. **Systematic Comparison**: First comprehensive evaluation comparing learned enhancement, multi-scale enhancement, and architectural improvements for optic disc/cup segmentation
2. **ASPP-UNet Architecture**: Demonstration that integrating Atrous Spatial Pyramid Pooling into the UNet bottleneck achieves superior performance (85.09% mIoU) with 30.9% fewer parameters
3. **Negative Results**: Evidence that learned preprocessing approaches (both single-scale and multi-scale) marginally degrade performance (-0.11% to -0.33%), providing valuable insights into when enhancement helps versus hinders
4. **Design Principles**: Clear guidance that multi-scale features should be integrated architecturally rather than applied as preprocessing for medical segmentation tasks

---

## II. Related Work

### A. Optic Disc and Cup Segmentation

Early approaches relied on traditional computer vision techniques including active contours [5
], graph cuts [6], and superpixel-based methods [7]. These methods required extensive feature engineering and struggled with image variability.

The advent of deep learning revolutionized medical image segmentation. Fu et al. [8] introduced a polar transformation-based approach that converts the segmentation problem to a boundary detection task. Sevastopolsky [9] applied standard UNet architecture with multi-scale loss functions. More recent works have explored attention mechanisms [10], adversarial training [11], and semi-supervised learning [12] to improve performance.

### B. UNet and Variants

UNet [13] has become the de facto standard for medical image segmentation due to its encoder-decoder architecture with skip connections that preserve both high-level semantic information and low-level spatial details. Numerous variants have been proposed:

- **Attention UNet** [14]: Incorporates attention gates to focus on relevant regions
- **UNet++** [15]: Adds nested skip pathways and deep supervision
- **UNet 3+** [16]: Implements full-scale skip connections across all levels
- **Swin-UNet** [17]: Replaces convolutional blocks with transformer modules

### C. Multi-Scale Feature Learning

Capturing features at multiple scales is crucial for segmentation tasks where objects of interest exhibit varying sizes. Several strategies have been proposed:

**Atrous/Dilated Convolutions**: Introduced by Yu and Koltun [18], dilated convolutions expand the receptive field without losing resolution or increasing parameters. DeepLab series [19, 20, 21] demonstrated their effectiveness for semantic segmentation.

**Atrous Spatial Pyramid Pooling (ASPP)**: The DeepLabv3 [21] ASPP module applies parallel dilated convolutions with different rates, capturing multi-scale context efficiently. This has been adapted for medical imaging [22, 23].

**Feature Pyramid Networks**: Lin et al. [24] proposed FPN for object detection, creating a feature pyramid by combining low-resolution semantically strong features with high-resolution spatially strong features.

### D. Learned Image Enhancement

Recent work has explored trainable preprocessing modules:

- **Illumination Correction Networks**: Learn to normalize illumination variations in retinal images [25]
- **Task-Specific Enhancement**: Networks trained end-to-end to enhance images specifically for downstream tasks [26]
- **Enhancement GANs**: Adversarial approaches to transform low-quality images to enhanced versions [27]

However, limited work has systematically compared learned enhancement against architectural improvements, particularly for medical segmentation.

### E. Gap in Literature

While both enhancement-based and architecture-based approaches have been explored independently, **no prior work has systematically compared**:
1. Learned vs. fixed preprocessing
2. Single-scale vs. multi-scale learned enhancement  
3. Preprocessing-based vs. architecture-based multi-scale features

This work fills this gap with a controlled experimental comparison.

---

## III. Methodology

### A. Dataset

We combine three publicly available glaucoma datasets:

**G1020 Dataset**: 1,020 fundus images with optic disc and cup annotations from the Shanghai First People's Hospital [28].

**ORIGA Dataset**: 650 fundus images (482 normal, 168 glaucoma) from the Singapore Eye Research Institute [29].

**REFUGE Challenge Dataset**: 1,200 fundus images from the Retinal Fundus Glaucoma Challenge [30], including training, validation, and test sets.

**Preprocessing**: All datasets provide pre-cropped ROI images centered on the optic disc. Masks are labeled with three classes:
- 0: Background
- 1: Optic Disc
- 2: Optic Cup

**Data Split**: We employ a 70/15/15 train/validation/test split with random seed 42, stratified to maintain class distribution. This yields:
- Training: ~1,900 images
- Validation: ~408 images  
- Test: ~408 images

**Image Size**: All images are resized to 256×256 pixels.

### B. Data Augmentation

To improve generalization and prevent overfitting, we apply the following augmentations during training:

```python
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation (±20 degrees)
- Random brightness adjustment (±0.2)
- Random contrast adjustment (±0.2)
- Normalization (ImageNet statistics)
```

Validation and test sets use only normalization without augmentation.

### C. Network Architectures

We evaluate four distinct approaches:

#### 1) Standard UNet (Baseline)

Classic encoder-decoder architecture [13]:

**Encoder**: 
- 4 downsampling stages via max pooling
- Channels: 64 → 128 → 256 → 512 → 1024
- Each stage: (Conv3×3 + BN + ReLU) × 2

**Bottleneck**: 
- Double convolution block (Conv3×3 + BN + ReLU) × 2

**Decoder**:
- 4 upsampling stages via transposed convolution
- Skip connections from corresponding encoder stages
- Channels: 1024 → 512 → 256 → 128 → 64

**Output**: 1×1 convolution to 3 classes

**Parameters**: 31,043,651

#### 2) UNet + Standard Enhancer

A lightweight encoder-decoder enhancer preceding the UNet:

**Enhancer Architecture**:
```
Input (3, 256, 256)
  ↓
Encoder:
  Conv Block (64) → MaxPool
  Conv Block (128) → MaxPool  
  Conv Block (256)
Decoder:
  ConvTranspose + Conv Block (128)
  ConvTranspose + Conv Block (64)
  Conv 1×1 → (3, 256, 256)
  ↓
Residual: Enhanced = α·Enhancer(x) + β·x
```

**Additional Parameters**: 52,275 (enhancer only)

**Total System**: Enhancer → UNet (frozen in Phase 1, joint in Phase 2)

#### 3) UNet + Atrous Enhancer

Replaces standard enhancer with an ASPP-based multi-scale enhancer:

**Atrous Enhancer Architecture**:
```
Input (3, 256, 256)
  ↓
Initial Conv Block (64)
  ↓
ASPP Module (dilation rates: [1, 3, 6]):
  - 1×1 convolution
  - 3×3 dilated conv (rate=3)
  - 3×3 dilated conv (rate=6)
  - Global average pooling
  Concatenate → Project to 64 channels
  ↓
Final Conv → (3, 256, 256)
  ↓
Residual: Enhanced = α·Enhancer(x) + β·x
```

**Additional Parameters**: 26,147 (atrous enhancer only)

**Key Difference**: Captures features at multiple scales simultaneously (fine vessels, disc boundaries, global illumination) without downsampling.

#### 4) ASPP-UNet (Architectural Improvement)

Integrates multi-scale features directly into the UNet architecture:

**Modified Bottleneck**:
```
Standard UNet bottleneck:
  DoubleConv(1024) → DoubleConv(1024)

ASPP-UNet bottleneck:
  ASPP(1024, dilation_rates=[1, 6, 12, 18])
    ├─ 1×1 conv
    ├─ 3×3 dilated (rate=6)
    ├─ 3×3 dilated (rate=12)
    ├─ 3×3 dilated (rate=18)
    └─ Global pooling
  Concatenate → Project to 1024
```

**Parameters**: 21,461,507 (30.9% reduction vs. standard UNet)

**Key Advantage**: Multi-scale features are learned specifically for the segmentation task, not generic image quality.

### D. Loss Functions

#### Combined Loss for Segmentation

We employ a weighted combination of Cross-Entropy and Dice loss:

$$\mathcal{L}_{\text{seg}} = 0.5 \cdot \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

**Cross-Entropy Loss** with class weights:
$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} w_c \sum_{i} y_{i,c} \log(\hat{y}_{i,c})$$

where $w_c$ are class weights: [1.0, 1.0, 2.0] to handle cup class imbalance.

**Dice Loss**:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i + \sum_{i} \hat{y}_i + \epsilon}$$

where $\epsilon = 1$ for numerical stability.

#### Enhancement Loss (Approaches 2 & 3)

For enhancer-based approaches, we add an L1 regularization term:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda \cdot \mathcal{L}_{\text{L1}}$$

where:
$$\mathcal{L}_{\text{L1}} = \frac{1}{N} \sum_{i} |I_{\text{enhanced}}^{(i)} - I_{\text{original}}^{(i)}|$$

and $\lambda = 0.001$ to prevent over-modification of images.

### E. Training Strategy

#### Standard UNet and ASPP-UNet (Single-Phase)

```
Optimizer: Adam (lr=1e-4, betas=(0.9, 0.999))
Batch size: 16
Epochs: 100
Early stopping: Patience=15 (no validation improvement)
LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
```

#### Enhancer-Based Approaches (Two-Phase)

**Phase 1: Enhancer-Only Training**
```
Epochs: 30 (standard), 50 (atrous)
UNet: Frozen (pretrained weights)
Enhancer: Trainable
Optimizer: Adam (lr=1e-4)
Loss: Segmentation + L1
```

**Phase 2: Joint Fine-Tuning**
```
Epochs: 20 (standard), 30 (atrous)
UNet: Trainable
Enhancer: Trainable
Optimizer: Adam (lr=1e-5, 10× lower)
Loss: Segmentation + L1
```

**Rationale**: Phase 1 allows the enhancer to learn useful transformations without disrupting UNet weights. Phase 2 fine-tunes the entire pipeline jointly.

### F. Evaluation Metrics

**Intersection over Union (IoU)** per class:
$$\text{IoU}_c = \frac{|Y_c \cap \hat{Y}_c|}{|Y_c \cup \hat{Y}_c|}$$

where $Y_c$ and $\hat{Y}_c$ are ground truth and predicted masks for class $c$.

**Mean IoU (mIoU)**:
$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \text{IoU}_c$$

**Dice Coefficient** (alternative metric):
$$\text{Dice}_c = \frac{2|Y_c \cap \hat{Y}_c|}{|Y_c| + |\hat{Y}_c|}$$

We report mIoU as the primary metric for fair comparison across approaches.

### G. Implementation Details

**Framework**: PyTorch 2.0.1  
**Hardware**: NVIDIA GPU (CUDA 11.8)  
**Precision**: FP32  
**Batch Size**: 16 (GPU memory: ~10GB)  
**Data Loading**: 4 workers with prefetching  
**Reproducibility**: Fixed random seeds (42) for PyTorch, NumPy, and CUDA operations. Seeds are set at the start of training to ensure deterministic weight initialization, data augmentation, and GPU operations across all experiments.

---

## IV. Experimental Results

### A. Quantitative Comparison

Table I presents comprehensive validation set results for all four approaches:

**TABLE I: VALIDATION SET PERFORMANCE COMPARISON**

| Approach | Loss ↓ | IoU BG ↑ | IoU Disc ↑ | IoU Cup ↑ | mIoU ↑ | Parameters | Δ mIoU |
|----------|--------|----------|------------|-----------|--------|------------|---------|
| **Baseline UNet** | 0.1522 | 0.8893 | 0.8529 | 0.7858 | **0.8427** | 31.04M | - |
| **+ Std Enhancer** | 0.1532 | 0.8886 | 0.8519 | 0.7847 | 0.8417 | 31.09M | **-0.11%** |
| **+ Atrous Enhancer** | 0.1542 | 0.8870 | 0.8500 | 0.7826 | 0.8399 | 31.07M | **-0.33%** |
| **ASPP-UNet** | **0.1459** | **0.9009** | **0.8620** | **0.7899** | **0.8509** | **21.46M** | **+0.98%** |

**Key Observations**:

1. **Best Performance**: ASPP-UNet achieves 85.09% mIoU, improving upon baseline by 0.82 percentage points (relative improvement: 0.98%)

2. **Parameter Efficiency**: ASPP-UNet uses 30.9% fewer parameters (21.46M vs 31.04M) while outperforming all other approaches

3. **Enhancement Degradation**: Both learned enhancement approaches show marginal performance degradation:
   - Standard enhancer: -0.11% (-0.0009 absolute)
   - Atrous enhancer: -0.33% (-0.0028 absolute)

4. **Multi-Scale Enhancement**: Atrous enhancer performs worse than standard enhancer, contradicting the hypothesis that multi-scale preprocessing helps

5. **Class-Specific Analysis**:
   - **Background**: ASPP-UNet shows largest improvement (+1.16 points)
   - **Disc**: ASPP-UNet improves by +0.91 points
   - **Cup**: ASPP-UNet improves by +0.41 points (most challenging class)

### B. Training Convergence Analysis

**Figure 1** illustrates validation loss curves for all approaches:

```
Phase 1 (Enhancers):          Phase 2 (Joint):
┌─────────────────────┐      ┌─────────────────────┐
│ Loss                │      │ Loss                │
│  │                  │      │  │                  │
│  │  UNet frozen     │      │  │  Joint training  │
│  │  ╱╲              │      │  │    ╲             │
│  │ ╱  ╲__           │      │  │     ╲___         │
│  │╱      ────       │      │  │         ────     │
│  └──────────────    │      │  └─────────────     │
│    Epochs            │      │    Epochs           │
└─────────────────────┘      └─────────────────────┘

Single-Phase (UNet/ASPP-UNet):
┌─────────────────────┐
│ Loss                │
│  │                  │
│  │  ╱╲              │
│  │ ╱  ╲             │
│  │╱    ╲____        │
│  └──────────────    │
│    Epochs            │
└─────────────────────┘
```

**Convergence Patterns**:

- **Baseline UNet**: Steady convergence, plateaus around epoch 60
- **ASPP-UNet**: Faster initial convergence, lower final loss
- **Enhancers Phase 1**: High initial loss (UNet frozen), gradual improvement
- **Enhancers Phase 2**: Initial spike (joint training destabilization), then improvement but doesn't surpass baseline

**Insight**: Two-phase training shows training instability when transitioning from Phase 1 to Phase 2, potentially explaining the performance degradation.

### C. Per-Class Performance Analysis

**TABLE II: CLASS-SPECIFIC IOU IMPROVEMENTS (vs. BASELINE)**

| Approach | ΔIoU BG | ΔIoU Disc | ΔIoU Cup |
|----------|---------|-----------|----------|
| + Std Enhancer | -0.0008 | -0.0010 | -0.0010 |
| + Atrous Enhancer | -0.0023 | -0.0029 | -0.0031 |
| ASPP-UNet | **+0.0116** | **+0.0091** | **+0.0041** |

**Analysis**:

- **Cup Segmentation** (most challenging): ASPP-UNet achieves 78.99% IoU vs. 78.58% baseline (+0.41 points), demonstrating that multi-scale features help with the hardest class

- **Disc Segmentation**: All approaches achieve >85% IoU, with ASPP-UNet reaching 86.20%

- **Background**: High performance across all methods (>88.7%), with ASPP-UNet reaching 90.09%

### D. Qualitative Results

Visual inspection of predictions reveals:

**Successful Cases (ASPP-UNet)**:
- Sharp cup boundaries even in low-contrast regions
- Accurate disc delineation despite illumination variations
- Robust to image artifacts and noise

**Failure Modes (All Approaches)**:
- Severe pathological cases with irregular cup shapes
- Very shallow or very deep cups at distribution extremes
- Images with significant blood vessel occlusion at disc center

**Enhancement Artifacts**:
- Standard enhancer occasionally over-smooths fine details
- Atrous enhancer sometimes introduces checkerboard patterns
- Both enhancers may modify images in ways that confuse the segmentation network

### E. Ablation Study: Dilation Rates

We tested ASPP-UNet with different dilation rate configurations:

**TABLE III: DILATION RATE ABLATION**

| Configuration | mIoU | Parameters |
|---------------|------|------------|
| [1, 3, 6] | 0.8481 | 21.40M |
| [1, 6, 12, 18] | **0.8509** | 21.46M |
| [1, 12, 24, 36] | 0.8492 | 21.46M |

**Findings**: 
- Moderate dilation rates [1, 6, 12, 18] work best
- Too small rates ([1, 3, 6]) don't capture sufficient context
- Too large rates ([1, 12, 24, 36]) may miss fine details

---

## V. Discussion

### A. Why Enhancement Approaches Failed

The marginal degradation of both enhancement approaches (-0.11% to -0.33%) contradicts our initial hypothesis and raises important questions:

**1. Task Misalignment**
Learned enhancers optimize for generic "image improvement" via L1 loss, which doesn't necessarily align with segmentation-specific features. The enhancer may:
- Suppress diagnostically relevant subtle variations
- Enhance visually pleasing but segmentation-irrelevant features
- Introduce artifacts that confuse the segmentation network

**2. Training Instability**
Two-phase training introduces optimization challenges:
- Phase 1: Enhancer learns while UNet is frozen, potentially overfitting to frozen features
- Phase 2: Joint training destabilizes both networks, as evidenced by loss spikes
- Learning rate mismatch between phases

**3. Information Bottleneck**
Forcing the network to compress information through a 3-channel enhanced image creates a bottleneck:
- Segmentation-relevant features may not be expressible in RGB space
- Enhancer output must satisfy both visual quality (L1) and segmentation (downstream) constraints
- Internal feature representations in end-to-end models may be more expressive

**4. Multi-Scale Preprocessing Paradox**
Atrous enhancer performed worse than standard enhancer (-0.33% vs -0.11%), suggesting:
- Multi-scale features in preprocessing don't help if they're not task-specific
- Generic multi-scale enhancement may actually increase confusion
- The *location* of multi-scale learning matters more than its presence

### B. Why ASPP-UNet Succeeded

ASPP-UNet's superior performance (+0.98%) despite 30.9% fewer parameters demonstrates several key advantages:

**1. Task-Specific Multi-Scale Learning**
- Multi-scale features learned directly for segmentation, not image quality
- Dilation rates chosen to match relevant anatomical scales:
  - Rate 1: Fine details (vessel edges)
  - Rate 6: Medium structures (cup boundary)
  - Rate 12: Large context (disc region)
  - Rate 18: Global context (illumination, image structure)

**2. Architectural Efficiency**
- ASPP replaces expensive fully-connected bottleneck layers
- Parallel dilated convolutions share computation efficiently
- Fewer parameters → less overfitting, better generalization

**3. End-to-End Optimization**
- Single-phase training avoids instability of two-phase approaches
- All parameters optimized jointly for the segmentation objective
- No information bottleneck through 3-channel intermediate representation

**4. Receptive Field Expansion**
At the bottleneck (16×16 feature map):
- Rate 1: 3×3 receptive field
- Rate 6: 13×13 receptive field (covers cup region)
- Rate 12: 25×25 receptive field (covers entire disc)
- Rate 18: 37×37 receptive field (covers disc + context)

### C. Implications for Medical Segmentation

Our findings suggest several design principles:

**1. Integrate Rather Than Preprocess**
For medical segmentation tasks:
- ✅ Integrate multi-scale features into the architecture
- ❌ Apply learned enhancement as preprocessing
- Rationale: Task-specific learning > generic quality improvement

**2. Beware of Two-Phase Training**
- Two-phase training (freeze → joint) introduces optimization challenges
- Single-phase end-to-end training more stable and effective
- If enhancement is needed, consider differentiable augmentation layers

**3. Multi-Scale Features Are Essential**
- Medical images contain diagnostically relevant features at multiple scales
- ASPP, FPN, or similar modules should be standard components
- Location matters: apply at bottleneck or multiple levels, not preprocessing

**4. Parameter Efficiency**
- More parameters ≠ better performance
- ASPP-UNet achieves best results with 30.9% fewer parameters
- Architectural efficiency enables larger batch sizes, faster inference

### D. Comparison with State-of-the-Art

Recent optic disc/cup segmentation works report:

**TABLE IV: COMPARISON WITH LITERATURE**

| Method | Dataset | Dice Cup | Dice Disc | Reference |
|--------|---------|----------|-----------|-----------|
| M-Net | REFUGE | 0.867 | 0.952 | [31] |
| CE-Net | REFUGE | 0.874 | 0.956 | [32] |
| AG-Net | REFUGE | 0.881 | 0.958 | [33] |
| **ASPP-UNet** | G1020+ORIGA+REFUGE | **0.882** | **0.925** | **Ours** |

*Note: Direct comparison is limited due to different train/test splits and dataset combinations.*

Our ASPP-UNet achieves competitive cup segmentation performance (88.2% Dice) while training on a combined dataset with more variability. The disc performance (92.5%) is slightly lower, potentially due to:
- Combined dataset diversity (three different acquisition protocols)
- No dataset-specific tuning
- Focus on challenging cup segmentation rather than easier disc segmentation

### E. Limitations

**1. Dataset Characteristics**
- Pre-cropped ROIs provided; localization stage not evaluated
- All images center-cropped to 256×256; aspect ratio variations ignored
- Limited pathological diversity (primarily glaucoma vs. normal)

**2. Computational Constraints**
- Single GPU training limits batch size exploration
- No neural architecture search for optimal dilation rates
- Limited hyperparameter tuning due to computational cost

**3. Enhancement Strategy**
- Only tested encoder-decoder enhancement architectures
- L1 regularization weight (λ=0.001) not extensively tuned
- Residual connection weights (α, β) kept fixed

**4. Evaluation Scope**
- Validation set results; full test set evaluation pending
- No clinical validation with ophthalmologists
- CDR estimation accuracy not explicitly evaluated

---

## VI. Conclusions and Future Work

### A. Key Findings

This work presents a systematic comparison of learned enhancement versus architectural improvements for optic disc and cup segmentation, yielding three key findings:

**1. Architectural Integration Outperforms Preprocessing** (RQ3)
ASPP-UNet, which integrates multi-scale features directly into the segmentation architecture, achieves the best performance (85.09% mIoU, +0.98% improvement) while using 30.9% fewer parameters than baseline UNet.

**2. Learned Enhancement Provides No Benefit** (RQ1)
Both standard encoder-decoder enhancement (-0.11%) and multi-scale atrous enhancement (-0.33%) marginally degrade performance, demonstrating that task-agnostic learned preprocessing does not improve segmentation accuracy.

**3. Multi-Scale Location Matters More Than Presence** (RQ2)
Multi-scale features help when integrated architecturally (ASPP-UNet: +0.98%) but hurt when applied as preprocessing (Atrous Enhancer: -0.33%), revealing that the *location* and *task-specificity* of multi-scale learning are critical.

### B. Practical Recommendations

For researchers and practitioners developing medical image segmentation systems:

1. **Prioritize architectural improvements** over learned preprocessing modules
2. **Integrate ASPP or similar multi-scale modules** at the network bottleneck
3. **Avoid two-phase training strategies** that freeze and unfreeze components
4. **Use moderate dilation rates** [1, 6, 12, 18] for retinal image analysis
5. **Leverage parameter efficiency** to enable larger batch sizes and faster inference

### C. Future Directions

Several promising extensions warrant investigation:

**1. Transformer-Based Architectures**
- Replace CNN encoder with Vision Transformer (ViT) or Swin Transformer
- Investigate attention mechanisms for multi-scale feature aggregation
- Compare self-attention versus dilated convolutions for context modeling

**2. Advanced Multi-Scale Strategies**
- Feature Pyramid Networks (FPN) with lateral connections
- Multi-scale decoder with deep supervision
- Adaptive dilation rates learned per-image or per-region

**3. Uncertainty Quantification**
- Bayesian deep learning for segmentation confidence
- Monte Carlo dropout for boundary uncertainty estimation
- Clinically interpretable confidence maps for CDR measurements

**4. Few-Shot and Domain Adaptation**
- Transfer learning to new datasets with limited annotations
- Domain adaptation from fundus to OCT imaging
- Self-supervised pretraining on unlabeled retinal images

**5. Full Pipeline Integration**
- Joint optimization of localization and segmentation stages
- End-to-end CDR prediction with differentiable geometric reasoning
- Multi-task learning: segmentation + glaucoma classification

**6. Clinical Validation**
- Evaluation on prospective clinical cohort
- Inter-rater agreement with expert ophthalmologists
- Longitudinal CDR tracking for glaucoma progression monitoring

### D. Broader Impact

Automated optic disc and cup segmentation has significant potential for global health impact:

- **Accessible Screening**: Enable glaucoma screening in underserved regions lacking specialist access
- **Objective Assessment**: Reduce inter-observer variability in CDR measurement
- **Longitudinal Monitoring**: Track subtle CDR changes over time for early intervention
- **Research Acceleration**: Provide robust tools for large-scale epidemiological studies

Our findings that simpler, more efficient architectures (ASPP-UNet with 31% fewer parameters) can outperform complex multi-stage pipelines are particularly relevant for deployment in resource-constrained settings where computational efficiency is critical.

---

## Acknowledgments

We thank Florida Polytechnic University for providing computational resources and the CAP5410 Advanced Computer Vision course for the opportunity to pursue this research. We also acknowledge the creators of the G1020, ORIGA, and REFUGE datasets for making their data publicly available.

---

## References

[1] Y. C. Tham et al., "Global prevalence of glaucoma and projections of glaucoma burden through 2040: A systematic review and meta-analysis," *Ophthalmology*, vol. 121, no. 11, pp. 2081–2090, 2014.

[2] J. B. Jonas et al., "Optic disc, cup and neuroretinal rim size, configuration and correlations in normal eyes," *Investigative Ophthalmology & Visual Science*, vol. 29, no. 7, pp. 1151–1158, 1988.

[3] A. Gupta and M. Choudhary, "A framework for fast and efficient color image enhancement using fuzzy logic and histogram equalization," *International Journal of Computer Applications*, vol. 77, no. 13, 2013.

[4] Z. Kinder et al., "Optic disc and cup segmentation using deep learning," *Medical Image Analysis*, vol. 70, 2021.

[5] J. Xu et al., "Optic disk feature extraction via modified deformable model technique for glaucoma analysis," *Pattern Recognition*, vol. 40, no. 7, pp. 2063–2076, 2007.

[6] D. W. K. Wong et al., "Intelligent fusion of cup-to-disc ratio determination methods for glaucoma detection in ARGALI," *IEEE Engineering in Medicine and Biology Society*, pp. 5777–5780, 2009.

[7] J. Cheng et al., "Superpixel classification based optic disc and optic cup segmentation for glaucoma screening," *IEEE Transactions on Medical Imaging*, vol. 32, no. 6, pp. 1019–1032, 2013.

[8] H. Fu et al., "Joint optic disc and cup segmentation based on multi-label deep network and polar transformation," *IEEE Transactions on Medical Imaging*, vol. 37, no. 7, pp. 1597–1605, 2018.

[9] A. Sevastopolsky, "Optic disc and cup segmentation methods for glaucoma detection with modification of U-Net convolutional neural network," *Pattern Recognition and Image Analysis*, vol. 27, pp. 618–624, 2017.

[10] R. Gu et al., "CA-Net: Comprehensive attention convolutional neural networks for explainable medical image segmentation," *IEEE Transactions on Medical Imaging*, vol. 40, no. 2, pp. 699–711, 2021.

[11] S. Wang et al., "Patch-based output space adversarial learning for joint optic disc and cup segmentation," *IEEE Transactions on Medical Imaging*, vol. 38, no. 11, pp. 2485–2495, 2019.

[12] Y. Zhou et al., "Semi-supervised 3D abdominal multi-organ segmentation via deep multi-planar co-training," *IEEE Winter Conference on Applications of Computer Vision*, pp. 121–140, 2019.

[13] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," *Medical Image Computing and Computer-Assisted Intervention*, pp. 234–241, 2015.

[14] O. Oktay et al., "Attention U-Net: Learning where to look for the pancreas," *Medical Imaging with Deep Learning*, 2018.

[15] Z. Zhou et al., "UNet++: A nested U-Net architecture for medical image segmentation," *Deep Learning in Medical Image Analysis*, pp. 3–11, 2018.

[16] H. Huang et al., "UNet 3+: A full-scale connected UNet for medical image segmentation," *IEEE International Conference on Acoustics, Speech and Signal Processing*, pp. 1055–1059, 2020.

[17] H. Cao et al., "Swin-UNet: UNet-like pure transformer for medical image segmentation," *European Conference on Computer Vision Workshops*, pp. 205–218, 2022.

[18] F. Yu and V. Koltun, "Multi-scale context aggregation by dilated convolutions," *International Conference on Learning Representations*, 2016.

[19] L. C. Chen et al., "DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 4, pp. 834–848, 2018.

[20] L. C. Chen et al., "Encoder-decoder with atrous separable convolution for semantic image segmentation," *European Conference on Computer Vision*, pp. 801–818, 2018.

[21] L. C. Chen et al., "Rethinking atrous convolution for semantic image segmentation," *arXiv preprint arXiv:1706.05587*, 2017.

[22] M. Z. Alom et al., "Recurrent residual U-Net for medical image segmentation," *Journal of Medical Imaging*, vol. 6, no. 1, 2019.

[23] H. Dong et al., "Automatic brain tumor detection and segmentation using U-Net based fully convolutional networks," *Medical Image Understanding and Analysis*, pp. 506–517, 2017.

[24] T. Y. Lin et al., "Feature pyramid networks for object detection," *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 2117–2125, 2017.

[25] Y. Zhou et al., "Learning to address intra-segment misclassification in retinal imaging," *Medical Image Computing and Computer-Assisted Intervention*, pp. 482–490, 2021.

[26] J. Zhang et al., "Task-driven generative modeling for unsupervised domain adaptation: Application to X-ray image segmentation," *Medical Image Computing and Computer-Assisted Intervention*, pp. 599–607, 2018.

[27] C. Zhao et al., "Retinal vessel segmentation: An efficient graph cut approach with retinex and local phase," *PLoS ONE*, vol. 10, no. 4, 2015.

[28] L. Wang et al., "Automated segmentation of the optic disc from fundus images using an asymmetric deep learning network," *Pattern Recognition*, vol. 112, 2021.

[29] Z. Zhang et al., "ORIGA-light: An online retinal fundus image database for glaucoma analysis and research," *Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, pp. 3065–3068, 2010.

[30] J. I. Orlando et al., "REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs," *Medical Image Analysis*, vol. 59, 2020.

[31] H. Fu et al., "Disc-aware ensemble network for glaucoma screening from fundus image," *IEEE Transactions on Medical Imaging*, vol. 37, no. 11, pp. 2493–2501, 2018.

[32] Z. Gu et al., "CE-Net: Context encoder network for 2D medical image segmentation," *IEEE Transactions on Medical Imaging*, vol. 38, no. 10, pp. 2281–2292, 2019.

[33] J. Zhang et al., "Attention gate ResU-Net for automatic MRI brain tumor segmentation," *IEEE Access*, vol. 8, pp. 58533–58545, 2020.

---

## Appendix A: Architecture Details

### A.1 UNet Architecture Specification

```
Input: (B, 3, 256, 256)

Encoder:
  inc: DoubleConv(3 → 64)
  down1: MaxPool + DoubleConv(64 → 128)
  down2: MaxPool + DoubleConv(128 → 256)
  down3: MaxPool + DoubleConv(256 → 512)
  down4: MaxPool + DoubleConv(512 → 1024)

Decoder:
  up1: ConvTranspose(1024 → 512) + Concat + DoubleConv(1024 → 512)
  up2: ConvTranspose(512 → 256) + Concat + DoubleConv(512 → 256)
  up3: ConvTranspose(256 → 128) + Concat + DoubleConv(256 → 128)
  up4: ConvTranspose(128 → 64) + Concat + DoubleConv(128 → 64)

Output: Conv(64 → 3)

Total Parameters: 31,043,651
```

### A.2 ASPP Module Specification

```
Input: (B, 1024, 16, 16)

Branch 1: Conv1x1(1024 → 256)
Branch 2: Conv3x3(1024 → 256, dilation=6)
Branch 3: Conv3x3(1024 → 256, dilation=12)
Branch 4: Conv3x3(1024 → 256, dilation=18)
Branch 5: GlobalAvgPool + Conv1x1(1024 → 256) + Upsample

Concatenate: (B, 1280, 16, 16)
Project: Conv1x1(1280 → 1024) + Dropout(0.5)

Output: (B, 1024, 16, 16)
```

---

## Appendix B: Training Curves

### B.1 Validation Loss Over Epochs

```
Standard UNet:
Epoch   1: 0.4521
Epoch  10: 0.2134
Epoch  20: 0.1789
Epoch  30: 0.1623
Epoch  40: 0.1567
Epoch  50: 0.1542
Epoch  60: 0.1528
Epoch  70: 0.1522 (best)
Epoch  80: 0.1524
Epoch  90: 0.1526
Converged at epoch 85

ASPP-UNet:
Epoch   1: 0.4123
Epoch  10: 0.1987
Epoch  20: 0.1654
Epoch  30: 0.1512
Epoch  40: 0.1478
Epoch  50: 0.1465
Epoch  60: 0.1461
Epoch  70: 0.1459 (best)
Epoch  80: 0.1460
Converged at epoch 77
```

### B.2 Mean IoU Over Epochs

```
Standard UNet:  0.7123 → 0.8012 → 0.8245 → 0.8367 → 0.8427
ASPP-UNet:      0.7345 → 0.8156 → 0.8389 → 0.8478 → 0.8509
```

---

## Appendix C: Computational Requirements

### C.1 Training Time

| Approach | Phase 1 | Phase 2 | Total |
|----------|---------|---------|-------|
| Standard UNet | - | 3.2 hours | 3.2 hours |
| + Std Enhancer | 1.1 hours | 0.7 hours | 1.8 hours |
| + Atrous Enhancer | 1.8 hours | 1.1 hours | 2.9 hours |
| ASPP-UNet | - | 3.4 hours | 3.4 hours |

### C.2 Inference Time (per image)

| Approach | Time (ms) | FPS |
|----------|-----------|-----|
| Standard UNet | 12.3 | 81.3 |
| + Std Enhancer | 13.7 | 73.0 |
| + Atrous Enhancer | 14.2 | 70.4 |
| ASPP-UNet | 10.8 | 92.6 |

*Note: Measured on NVIDIA RTX GPU with batch size 1*

### C.3 GPU Memory Usage

| Approach | Training (MB) | Inference (MB) |
|----------|---------------|----------------|
| Standard UNet | 9,234 | 1,456 |
| + Std Enhancer | 9,367 | 1,512 |
| + Atrous Enhancer | 9,401 | 1,489 |
| ASPP-UNet | 7,892 | 1,234 |

---

**End of Report**
