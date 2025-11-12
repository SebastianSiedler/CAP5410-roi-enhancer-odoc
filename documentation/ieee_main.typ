#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [ROI Enhancer for Optic Disc and Cup Segmentation in Fundus Images using Task-Aware Multi-Scale Feature Learning],
  abstract: [
    TODO:
  ],
  authors: (
    (
      name: "Sebastian Siedler",
      // department: [Co-Founder],
      // organization: [Typst GmbH],
      location: [Lakeland, FL],
      email: "ssiedler3117@floridapoly.edu",
    ),
    (
      name: "Steffen Ullmann",
      // department: [Co-Founder],
      // organization: [Typst GmbH],
      location: [Lakeland, FL],
      email: "sullmann3121@floridapoly.edu",
    ),
  ),
  index-terms: (
    "Medical image segmentation",
    "optic disc",
    "optic cup",
    "glaucoma",
    "UNet",
    "ASPP",
    "multi-scale learning",
    "deep learning",
  ),
  bibliography: bibliography("works.bib"),
  figure-supplement: [Fig.],
)

= Introduction
TODO:

optic disc (OD)
optic cup (OC)

= Related Work
== Glaucoma Image Segmentation
// Which models used for OD/OC Segmentation?
Most segmentation models used in related works are based on the UNet architecture due to its encoder-decoder architecture with skip connections that preserve high-level semantic information while also retaining low-level spatial details @ronneberger2015unetconvolutionalnetworksbiomedical. 
// --> We also use UNet as baseline

Xiong et al. @HaorenXiong.2025 proposed the multi-task deep learning model Multi-GlaucNet to simultaneously perform OD and blood vessel segmentation as well as glaucoma detection. The architecture is based on UNet with bottleneck layers in the encoder, pixel shuffle and a channel attention mechanism in the decoder. The final diagnosis is made by a ResNet50-based classification module. It achieves a high accuracy of 0.967 for glaucoma detection on the REFUGE dataset.

Liu et al. @Liu.2025 introduced EE-TransUNet for improved segmentation accuracy around the edges of the OD and OC. The model builds upon the TransUNet architecture including a Cascaded Convolutional Fusion (CCF) block to enhance feature abstraction, preserve original feature information, and improve the model's non-linear fitting ability. The model also incorporated a Channel Shuffling Multiple Expansion Fusion (CSMF) block to improve the network's capacity to perceive and characterize image features. EE-TransUNet demonstrated superior segmentation performance compared to other state-of-the-art models on multiple datasets.

Outstanding results were obtained by Zedan et al. @Zedan.2025 with the proposed RMHA-Net, a U-shaped encoder-decoder network for robust OD and OC segmentation. Through Residual-Atrous-Conv (RAC) modules and Atrous Spatial Pyramid Pooling (ASPP) blocks, the model effectively captures multi-scale features and contextual information, enhancing segmentation accuracy in fundus images. Furthermore, the model incorporates a Hybrid Attention Mechanism (Spatial and Channel Attention) to dynamically prioritize relevant features and suppress noise. The RMHA-Net demonstrated the best performance against benchmark models across multiple datasets.
// --> We also use ASPP blocks


== Fundus Image Enhancement Techniques
// What techniques used for fundus image enhancement?
Medical images are often affected by various artifacts such as low contrast, distortions, and noise, which can hinder accurate analysis and diagnosis @Zedan.2025. To address these issues, image preprocessing techniques are applied to enhance image quality by removing artifacts. Several ways of image enhancement are commonly used in the literature.

Traditional static enhancement methods such as Contrast Limited Adaptive Histogram Equalization (CLAHE) are the most widely used in fundus imaging. The previous mentioned works @HaorenXiong.2025 and @Zedan.2025 applied CLAHE as a preprocessing step before feeding the images into their segmentation models to achieve their remarkable results. CLAHE enhances the image contrast by applying histogram equalization in small regions of the image. This improves visibility of features of the blood vessels and the OD, leading to improved segmentation performance @HaorenXiong.2025.

In contrast to static preprocessing methods like CLAHE, only few works employ learned image enhancement models. Generative Adversarial Networks (GANs) are often used for image-to-image translation, including image enhancement tasks. Due to limited availability of paired training data in medical imaging, various adaptions have been proposed, such as CycleGAN @Zhu.2017 which employs an encoder–decoder-based generator architecture and enables unpaired image-to-image translation through cycle consistency.

You et al. @You.2019 further enhanced this method and proposed the Cycle-CBAM method for retinal image enhancement to translate poor-quality fundus images to high-quality images. Based on CycleGAN, it also integrates a Convolutional Block Attention Module (CBAM) into the Residual Blocks of the CycleGAN generators to preserve image textures and color details more effectively. The attention technique uses channel and spatial attention to adaptively emphasize important features and suppress irrelevant ones. The model was trained using unpaired low- and high-quality fundus images from the EyePACS and PD datasets. In comparison to static enhancement methods, Cycle-CBAM produced more natural and detailed images and also improved the accuracy of diabetic retinopathy classification when using the enhanced images.
// --> Enhancer trained independently and is not trained task-aware, not applied for Glaucoma segmentation
// --> We also train enhancer jointly with segmentation model to be task-aware

Other enhancement models are also based on the encoder-decoder architecture such as the UNet. It is designed for medical image segmentation tasks but is also widely used for image enhancement tasks @Lin.2025. However, Liu et al. @Liu.2025 demonstrated superior results using their previously mentioned EE-TransUNet without explicit image enhancement by focusing on architectural improvements to their segmentation model.
// --> correlates with our final result that architectural improvements are more effective than preprocessing

// Show gap:
While both static and learned enhancement methods have shown benefits in improving fundus image quality, most enhancement techniques are designed as independent preprocessing modules that are optimized separately from the actual analysis task. As a result, the enhancement focuses primarily on visual quality rather than task-specific feature optimization. This limits their effectiveness when they are integrated into diagnostic pipelines such as glaucoma detection and segmentation, where the final goal is accurate analysis rather than just improved image appearance.

To address this gap, we propose a task-aware learned enhancement approach to jointly train the enhancement model together with the segmentation model. By optimizing both modules end-to-end, the enhancer learns to emphasize features that are relevant for accurate OD and OC segmentation.


= Methodology
== Dataset
We combined three publicly available glaucoma datasets:
+ *G1020 Dataset*: 1,020 fundus images with optic disc and cup annotations from the Shanghai First People's Hospital @src_bajwa2020g1020benchmarkretinalfundus.

+ *ORIGA Dataset*: 650 fundus images (482 normal, 168 glaucoma) from the Singapore Eye Research Institute @src_Origa.

+ *REFUGE Challenge Dataset*: 1,200 fundus images from the Retinal Fundus Glaucoma Challenge @src_REFUGE_dataset, including training, validation, and test sets.

*Preprocessing:* \
We obtained these three datasets from the "Glaucoma Fundus Imaging Datasets" Kaggle repository#footnote("https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets/data"). They were already preprocessed to a certain extent, including pre-propped ROI images centered on the optic disc. The masks are labeled with three classes:

- 0: background
- 1: optic disc
- 2: optic cup

*Data Split:* \
We used a 70/15/15 split with random seed 42 for training, validation, and testing, respectively. After combining the datasets, we had a total of 2,870 images. We ensured that the splits were stratified to maintain the proportion of glaucoma and normal cases across all sets. Also some of the images had incomplete masks (234), e.g. missing optic cup or disc annotations. We filtered out these images to ensure the quality of our training data.

This resulted in the following distribution:
- Total images before filtering: 2870
- Images with incomplete masks: 234
- Total images after filtering: 2636
- Training set: 1845 images
- Validation set: 395 images
- Test set: 396 images

== Data Augmentation
To improve generatlization and robustness of our model, we applied the following augmenations durting training:
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation (±20 degrees)
- Random brightness adjustment (±0.2)
- Random contrast adjustment (±0.2)
- Normalization (ImageNet statistics)
The implementation was done using the Albumentations library @src_2018arXiv180906839B in the file `transforms.py`. Validation and test sets use only normalization without augmenation.

== Network Architectures
We evaluate five distinct approaches: // TODO: add clahe

+ Standard UNet (Baseline) \
  *Classic encoder-decoder architecture* @ronneberger2015unetconvolutionalnetworksbiomedical:

  *Encoder:*
  - 4 downsampling stages via max pooking
  - Channels: 64 → 128 → 256 → 512 → 1024
  - Each stage: (Conv3×3 + BN + ReLU) × 2

  *Bottleneck:*
  - Double convolution block (Conv3×3 + BN + ReLU) × 2

  *Decoder:*
  - 4 upsampling stages via transposed convolution
  - Skip connections from corresponding encoder stages
  - Channels: 1024 → 512 → 256 → 128 → 64

  *Output:*
  - 1x1 convolution to 3 classes

  *Parameters*:
  - 31,043,651


+ UNet + Standard Enhancer \
  A lightweight encoder-decoder enhancer preceding the UNet: \
  *Input* $(3, 256, 256)$ \
  *Encoder*:
  - Conv Block (64) -> MaxPool
  - Conv Block (128) -> MaxPool
  - Conv Block (256)
  - Decoder:
    - ConvTranspose + Conv Block (128)
    - ConvTranspose + Conv Block (64)
    - Conv 1x1 -> (3, 256, 256)
  - Residual: Enhanced = $alpha dot "Enhancer"(x) + beta dot x$ \


  *Additional Parameters*: 52,275 (enhancer only)

  *Total System*: Enhancer → UNet (frozen in Phase 1, joint in Phase 2)

+ UNet + Atrous Enhancer
  Replaces standard enhancer with an ASPP-based multi-scale enhancer:

  *Atrous Enhancer Architecture:*

  Input (3, 256, 256) \
  ↓ \
  Initial Conv Block (64) \
  ↓ \
  ASPP Module (dilation rates: [1, 3, 6]):
  - 1×1 convolution
  - 3×3 dilated conv (rate=3)
  - 3×3 dilated conv (rate=6)
  - Global average pooling
  Concatenate → Project to 64 channels \
  ↓ \
  Final Conv → (3, 256, 256) \
  ↓ \
  Residual: $"Enhanced" = alpha dot "Enhancer"(x) + beta dot x$
  *Additional Parameters*: 26,147 (atrous enhancer only) \
  *Key Difference*: Captures features at multiple scales simultaneously (fine vessels, disc boundaries, global illumination) without downsampling.

+ ASPP-UNet (Architectural Improvement) \
  Integrates multi-scale features directly into the UNet architecture:

  Modified Bottleneck: \

  Standard UNet bottleneck: \
  DoubleConv(1024) → DoubleConv(1024) \

  ASPP-UNet bottleneck: \
  ASPP(1024, dilation_rates=[1, 6, 12, 18]) \
  ├─ 1×1 conv \
  ├─ 3×3 dilated (rate=6) \
  ├─ 3×3 dilated (rate=12) \
  ├─ 3×3 dilated (rate=18) \
  └─ Global pooling \
  Concatenate → Project to 1024 \
  Parameters: 21,461,507 (30.9% reduction vs. standard UNet) \

  Key Advantage: Multi-scale features are learned specifically for the segmentation task, not generic image quality.



== Loss Functions



We employ a weighted combination of Cross-Entropy and Dice loss:

$ L_"seg" = 0.5 dot L_"CE" + 0.5 dot L_"Dice" $

*Cross-Entropy Loss* with class weights:

$ L_"CE" = -sum_(c=1)^C w_c sum_i y_"i,c" log(hat(y)_"i,c") $


where $w_c$ are class weights: $[1.0, 1.0, 2.0]$ to handle cup class imbalance.

*Dice Loss*:

$ L_"Dice" = 1 - (2 sum_i y_i hat(y)_i + epsilon)/(sum_i y_i + sum_i hat(y)_i + epsilon) $

where $epsilon = 1$ for numerical stability.





==== Enhancement Loss (Approaches 2 & 3)

For enhancer-based approaches, we add an L1 regularization term:

$ L_"total" = L_"seg" + lambda dot L_"L1" $

where:

$ L_"L1" = 1/N sum_i |I_"enhanced"^(i) - I_"original"^(i)| $

and $lambda = 0.001$ to prevent over-modification of images.



== Training Strategy
*Standard UNet and ASPP-Unet (Single-Phase)*:
- Optimizer: Adam (lr=1e-4, betas=(0.9, 0.999))
- Batch size: 16
- Epochs: 100
- Early stopping: Patience=15 (no validation improvement)
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)


*Enhancer-Based Approaches (Two-Phase)*
+ Enhancer-Only Training
  - Epochs: 30 (standard), 50 (atrous)
  - UNet: Frozen (pretrained weights)
  - Enhancer: Trainable
  - Optimizer: Adam (lr=1e-4)
  - Loss: Segmentation + L1

+ Joint Fine-Tuning
  - Epochs: 20 (standard), 30 (atrous)
  - UNet: Trainable
  - Enhancer: Trainable
  - Optimizer: Adam (lr=1e-5, 10× lower)
  - Loss: Segmentation + L1

Rationale: Phase 1 allows the enhancer to learn useful transformations without disrupting UNet weights. Phase 2 fine-tunes the entire pipeline jointly.


== Evaluation Metrics
*Intersection over Union (IoU)* per class:

$ "IoU"_c = (Y_c inter hat(Y)_c)/(Y_c union hat(Y)_c) $

where $Y_c$ is ground truth for class c, $hat(Y)_c$ is prediction.

*Mean IoU (mIoU)*:

$ "mIoU" = 1/C sum_(c=1)^C "IoU"_c $

*Dice Coefficient* (alternative metric):

$ "Dice"_c =(2 |Y_c inter hat(Y)_c|)/(|Y_c|+|hat(Y_c)|) $

We report mIoU as the primary metric for fair comparison across approaches.


== Implementation Details // Brauchen wir das?


= Experimental Results

== Quantitative Comparison

@fig_quantitative_comparison presents comprehensive validation set results for all five approaches.

// Load comparison data from JSON
#let comparison_data = json("../results/comparison_simple.json")

// Create table from the data
#let approaches = comparison_data.approaches
#let metrics = comparison_data.metrics
#let data = comparison_data.data

// Header mapping for display
#let header_map = (
  "loss": "Loss",
  "iou_bg": "IoU BG",
  "iou_disc": "IoU Disc",
  "iou_cup": "IoU Cup",
  "miou": "mIoU",
  "parameters": "Parameters",
  "delta_miou": "Δ mIoU",
)

// Approach display mapping
#let approach_display = (
  "baseline_unet": "Baseline UNet",
  "clahe": "CLAHE",
  "baseline_std_enhancer": "Baseline + Std Enhancer",
  "baseline_atrous_enhancer": "Baseline + Atrous Enhancer",
  "aspp_unet": "ASPP-UNet",
)

// Helper function to format numbers
#let format_number(value, metric) = {
  if value == none {
    "---"
  } else if type(value) == float {
    if metric == "delta_miou" {
      // Format as percentage with + sign for positive values
      let percent = calc.round(value * 100, digits: 2)
      if percent == 0 {
        "---"
      } else if percent > 0 {
        "+" + str(percent) + "%"
      } else {
        str(percent) + "%"
      }
    } else if value >= 1.0 {
      str(calc.round(value, digits: 0))
    } else if value >= 0.1 {
      str(calc.round(value, digits: 3))
    } else {
      str(calc.round(value, digits: 4))
    }
  } else {
    str(value)
  }
}

// TODO: Maybe spread this table across whole page width
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: center,
    table.header(
      [*Approach*],
      ..metrics.map(m => [*#(header_map.at(m))*]),
    ),
    ..for approach in approaches {
      let row = ([*#(approach_display.at(approach))*],)
      for metric in metrics {
        let value = data.at(approach).at(metric)
        row.push(format_number(value, metric))
      }
      row
    },
  ),
  caption: [Quantitative comparison of all five approaches on the validation set. ASPP-UNet achieves the best performance with 85.09% mIoU while using 30.9% fewer parameters than the baseline. Learned enhancement approaches show marginal degradation, suggesting that preprocessing-based methods are less effective than architectural improvements for this task.], // Caption TODO:
) <fig_quantitative_comparison>


Key Observations:
+ *Best Performance:* ASPP-UNet achieves #calc.round(comparison_data.data.aspp_unet.miou * 100, digits: 2)% mIoU, improving upon baseline by #calc.round(comparison_data.data.aspp_unet.delta_miou * 100, digits: 2) percentage points (relative improvement: #calc.round((comparison_data.data.aspp_unet.miou - comparison_data.data.baseline_unet.miou) / comparison_data.data.baseline_unet.miou * 100, digits: 2)%).

+ *Parameter Efficiency:* ASPP-UNet uses #calc.round((1 - comparison_data.data.aspp_unet.parameters / comparison_data.data.baseline_unet.parameters) * 100, digits: 2)% fewer parameters (#calc.round(comparison_data.data.aspp_unet.parameters / 1000000, digits: 2)M vs #calc.round(comparison_data.data.baseline_unet.parameters / 1000000, digits: 2)M) while outperforming all other approaches

+ *Enhancement Degradation:* Both standard and atrous enhancer approaches show slight performance drops compared to the baseline UNet
  - Standard enhancer: #calc.round(comparison_data.data.baseline_std_enhancer.delta_miou * 100, digits: 2)%
  - Atrous enhancer: #calc.round(comparison_data.data.baseline_atrous_enhancer.delta_miou * 100, digits: 2)%

// TODO: why is clahe worse?
+ *CLAHE Preprocessing*: Surprisingly, CLAHE preprocessing leads to a minor decrease in performance compared to the baseline UNet. (#calc.round(comparison_data.data.clahe.delta_miou * 100, digits: 2)% points)

// TODO: we should check this. This could also be just margin of error
+ *Multi-Scale Enhancement*: Altrous enhancer performs worse than standard enhancer, contradicting the hypothesis that multi-scale preprocessing helps

+ *Class-Specific Analysis*:
  - Background: ASPP-UNET shows largest improvement (+#calc.round((comparison_data.data.aspp_unet.iou_bg - comparison_data.data.baseline_unet.iou_bg) * 100, digits: 2)% points)
  - Disc: ASPP-Unet improves by +#calc.round((comparison_data.data.aspp_unet.iou_disc - comparison_data.data.baseline_unet.iou_disc) * 100, digits: 2)% points
  - ASPP-UNet improves by +#calc.round((comparison_data.data.aspp_unet.iou_cup - comparison_data.data.baseline_unet.iou_cup) * 100, digits: 2)% points (most challenging class)

== Training Convergence Analysis
// TODO:


== Per-Class Performance Analysis

#let baseline_iou_bg = comparison_data.data.baseline_unet.iou_bg
#let baseline_iou_disc = comparison_data.data.baseline_unet.iou_disc
#let baseline_iou_cup = comparison_data.data.baseline_unet.iou_cup

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: center,
    table.header([*Approach*], [*ΔIoU BG*], [*ΔIoU Disc*], [*ΔIoU Cup*]),
    [CLAHE],
    format_number(comparison_data.data.clahe.iou_bg - baseline_iou_bg, "Δ mIoU"),
    format_number(comparison_data.data.clahe.iou_disc - baseline_iou_disc, "Δ mIoU"),
    format_number(comparison_data.data.clahe.iou_cup - baseline_iou_cup, "Δ mIoU"),

    [#sym.plus Std Enhancer],
    format_number(comparison_data.data.baseline_std_enhancer.iou_bg - baseline_iou_bg, "Δ mIoU"),
    format_number(comparison_data.data.baseline_std_enhancer.iou_disc - baseline_iou_disc, "Δ mIoU"),
    format_number(comparison_data.data.baseline_std_enhancer.iou_cup - baseline_iou_cup, "Δ mIoU"),

    [#sym.plus Atrous Enhancer],
    format_number(comparison_data.data.baseline_atrous_enhancer.iou_bg - baseline_iou_bg, "Δ mIoU"),
    format_number(comparison_data.data.baseline_atrous_enhancer.iou_disc - baseline_iou_disc, "Δ mIoU"),
    format_number(comparison_data.data.baseline_atrous_enhancer.iou_cup - baseline_iou_cup, "Δ mIoU"),

    [ASPP-UNet],
    format_number(comparison_data.data.aspp_unet.iou_bg - baseline_iou_bg, "Δ mIoU"),
    format_number(comparison_data.data.aspp_unet.iou_disc - baseline_iou_disc, "Δ mIoU"),
    format_number(comparison_data.data.aspp_unet.iou_cup - baseline_iou_cup, "Δ mIoU"),
  ),
  caption: [Class-specific IoU improvements over baseline UNet. ASPP-UNet shows consistent improvements across all classes, with the largest gains in background segmentation.],
) <fig_class_specific>

*Analysis:*

- *Cup Segmentation (most challenging):* ASPP-UNet achieves #calc.round(comparison_data.data.aspp_unet.iou_cup * 100, digits: 2)% IoU vs. #calc.round(baseline_iou_cup * 100, digits: 2)% baseline (#format_number(comparison_data.data.aspp_unet.iou_cup - baseline_iou_cup, "Δ mIoU")), demonstrating that multi-scale features help with the hardest class

- *Disc Segmentation:* All approaches achieve >#calc.round(calc.min(..approaches.map(a => comparison_data.data.at(a).iou_disc)) * 100, digits: 1)%, with ASPP-UNet reaching #calc.round(comparison_data.data.aspp_unet.iou_disc * 100, digits: 2)%

- *Background:* High performance across all methods (>#calc.round(calc.min(..approaches.map(a => comparison_data.data.at(a).iou_bg)) * 100, digits: 1)%), with ASPP-UNet reaching #calc.round(comparison_data.data.aspp_unet.iou_bg * 100, digits: 2)%

== Qualitative Results
Visual inspection of predictions reveals:

// TODO: was genau machen wir hier rein?
// - wir können z.B. zeigen, dass die Edges bei asp-unet besser definiert sind. Nicht so franzig
//

= Discussion

= Conclusions and Future Work

// TODO: irgendwie gibt es so ein satz von machinelearning so nach dem Motto "Keep it simple"
