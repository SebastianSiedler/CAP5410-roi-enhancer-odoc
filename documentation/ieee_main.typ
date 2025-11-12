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

= Related Work

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
// TODO: hier noch mal kurz dazuschreiben, wie wir überhaupt darauf gekommen sind. Also grundidee der enhancer mit unet vergleich clahe. weil aber in dem aktuellen paper das mit dem aspp so krass sein soll, haben wir geschaut, wie sich der enhancer dann verhält, wenn wir das mit atrous machen.
We evaluate five distinct approaches: // TODO: add clahe

+ Standard UNet (Baseline) \
  *Classic encoder-decoder architecture* @src_ronneberger2015unetconvolutionalnetworksbiomedical:

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
// TODO: for whole chapter. check all parameters again in the end

*Single-Phase Training (UNet Variants)*

Standard UNet and CLAHE-UNet:
- Optimizer: Adam (lr=1e-4, betas=(0.9, 0.999))
- Batch size: 16
- Epochs: 100 (no early stopping)
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Note: Both models showed continued improvement through epoch 100

ASPP-UNet:
- Optimizer: Adam (lr=1e-4, betas=(0.9, 0.999))
- Batch size: 16
- Epochs: 100 (maximum)
- Early stopping: Patience=15
- Actual epochs: 67 (early stopping triggered)
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Note: Converged faster than baseline models, demonstrating superior training efficiency


*Enhancer-Based Approaches (Two-Phase)*

+ Phase 1: Enhancer-Only Training
  - Epochs: 30 (standard), 50 (atrous)
  - Early stopping: Patience=10 (standard), 15 (atrous)
  - Actual epochs: 26 (standard), 50 (atrous)
  - UNet: Frozen (pretrained weights)
  - Enhancer: Trainable
  - Optimizer: Adam (lr=1e-4)
  - Loss: Segmentation + L1

+ Phase 2: Joint Fine-Tuning
  - Epochs: 20 (standard), 30 (atrous)
  - Actual epochs: 20 (standard), 25 (atrous)
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
#let data = json("../results/test_comparison.json")
#let approaches = data.keys()
#let metrics = data.aspp_unet.keys() // just any approach to get metric names

// Header mapping for display
#let header_map = (
  "loss": "Loss",
  "iou_bg": "IoU BG",
  "iou_disc": "IoU Disc",
  "iou_cup": "IoU Cup",
  "miou": "mIoU",
  "parameters": "Parameters",
  "delta_miou": [#sym.Delta  mIoU],
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
// TODO: Missing values for CLAHE
#place(top + center, scope: "parent", float: true)[
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
]


Key Observations:
+ *Best Performance:* ASPP-UNet achieves #calc.round(data.aspp_unet.miou * 100, digits: 2)% mIoU, improving upon baseline by #calc.round(data.aspp_unet.delta_miou * 100, digits: 2) percentage points (relative improvement: #calc.round((data.aspp_unet.miou - data.baseline_unet.miou) / data.baseline_unet.miou * 100, digits: 2)%).

+ *Parameter Efficiency:* ASPP-UNet uses #calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% fewer parameters (#calc.round(data.aspp_unet.parameters / 1000000, digits: 2)M vs #calc.round(data.baseline_unet.parameters / 1000000, digits: 2)M) while outperforming all other approaches

+ *Training Efficiency:* ASPP-UNet demonstrated superior convergence, reaching optimal performance in 67 epochs compared to 100 epochs required by baseline models. This faster convergence, combined with better final performance, indicates that the ASPP module provides a more effective inductive bias for this segmentation task.

+ *Enhancement Degradation:* Both standard and atrous enhancer approaches show slight performance drops compared to the baseline UNet
  - Standard enhancer: #calc.round(data.baseline_std_enhancer.delta_miou * 100, digits: 2)%
  - Atrous enhancer: #calc.round(data.baseline_atrous_enhancer.delta_miou * 100, digits: 2)%

// TODO: why is clahe worse?
+ *CLAHE Preprocessing*: Surprisingly, CLAHE preprocessing leads to a minor decrease in performance compared to the baseline UNet. (#calc.round(data.clahe.delta_miou * 100, digits: 2)% points)

// TODO: we should check this. This could also be just margin of error
+ *Multi-Scale Enhancement*: Altrous enhancer performs worse than standard enhancer, contradicting the hypothesis that multi-scale preprocessing helps

+ *Class-Specific Analysis*:
  - Background: ASPP-UNET shows largest improvement (+#calc.round((data.aspp_unet.iou_bg - data.baseline_unet.iou_bg) * 100, digits: 2)% points)
  - Disc: ASPP-Unet improves by +#calc.round((data.aspp_unet.iou_disc - data.baseline_unet.iou_disc) * 100, digits: 2)% points
  - ASPP-UNet improves by +#calc.round((data.aspp_unet.iou_cup - data.baseline_unet.iou_cup) * 100, digits: 2)% points (most challenging class)

== Training Convergence Analysis

// Load convergence metrics (auto-generated from notebooks/training_curves_visualization.ipynb)
// Run the "Generate Convergence Metrics for Paper" section to regenerate if training data changes
#let conv = json("../results/convergence_metrics.json")

=== Single-Phase UNet Variants

@fig_single_phase_comparison illustrates the training dynamics of the three single-phase UNet architectures over #conv.unet.total_epochs epochs. The convergence patterns reveal significant differences in training efficiency and architectural effectiveness.

#figure(
  image("../results/single_phase_unets_comparison.png", width: 100%),
  caption: [Training and validation loss curves for single-phase UNet variants. ASPP-UNet (middle) demonstrates superior convergence, triggering early stopping at epoch #conv.aspp_unet.total_epochs, while Standard UNet (left) and CLAHE-UNet (right) required the full #conv.unet.total_epochs epochs.],
) <fig_single_phase_comparison>

*Standard UNet*: Required the full #conv.unet.total_epochs epochs to reach its best validation loss (#conv.unet.best_val_loss), with convergence (within 1% of optimal) achieved at epoch #conv.unet.convergence_epoch. The model exhibited stable improvement throughout training with minimal variance (#calc.round(conv.unet.last_15_variance, digits: 6)) in the final 15 epochs, suggesting it could potentially benefit from additional training. This baseline establishes the performance benchmark against which other approaches are evaluated.

*ASPP-UNet*: Demonstrated superior training efficiency, converging within 1% of optimal validation loss at epoch #conv.aspp_unet.convergence_epoch and triggering early stopping at epoch #conv.aspp_unet.total_epochs. The final validation loss of #conv.aspp_unet.final_val_loss represents a #conv.aspp_unet.overfitting_increase_percent% increase from the best (#conv.aspp_unet.best_val_loss at epoch #conv.aspp_unet.best_epoch), indicating slight overfitting that was appropriately halted by early stopping. The faster convergence (#conv.aspp_unet.convergence_epoch vs. #conv.unet.convergence_epoch epochs) demonstrates that the ASPP bottleneck provides a more effective inductive bias for this segmentation task.

*CLAHE-UNet*: Required near-complete training (#conv.clahe_unet.convergence_epoch/#conv.clahe_unet.total_epochs epochs to convergence), achieving best validation loss of #conv.clahe_unet.best_val_loss at epoch #conv.clahe_unet.best_epoch. Despite the preprocessing-based contrast enhancement, convergence was not accelerated compared to the baseline. The late convergence and slightly worse final performance suggest that CLAHE preprocessing does not provide meaningful benefits for this task when combined with deep learning.

=== Two-Phase Enhancer Approaches

The enhancer-based approaches employ a two-phase training strategy: Phase 1 trains only the enhancer with a frozen pretrained UNet, while Phase 2 performs joint fine-tuning of both components. @fig_std_enhancer_two_phase and @fig_atrous_enhancer_two_phase illustrate the complete training journey for each enhancer model, showing the transition from Phase 1 to Phase 2.

#figure(
  image("../results/standard_enhancer_two_phase.png", width: 80%),
  caption: [Two-phase training curve for Standard Enhancer. Phase 1 (blue/purple, UNet frozen) and Phase 2 (orange/red, joint training) are shown with a clear transition point at epoch #conv.std_enhancer.phase1_epochs. The model completed training in #conv.std_enhancer.total_epochs total epochs.],
) <fig_std_enhancer_two_phase>

#figure(
  image("../results/atrous_enhancer_two_phase.png", width: 80%),
  caption: [Two-phase training curve for Atrous Enhancer. Phase 1 (blue/purple, UNet frozen) and Phase 2 (orange/red, joint training) are shown with a clear transition point at epoch #conv.atrous_enhancer.phase1_epochs. The model required #conv.atrous_enhancer.total_epochs total epochs.],
) <fig_atrous_enhancer_two_phase>

*Standard Enhancer*: Phase 1 (#conv.std_enhancer.phase1_epochs epochs with early stopping) achieved modest loss reduction (#conv.std_enhancer.phase1_reduction_percent%), while Phase 2 joint fine-tuning (#conv.std_enhancer.phase2_epochs epochs) yielded #conv.std_enhancer.phase2_reduction_percent% additional reduction. The total training time of #conv.std_enhancer.total_epochs epochs is substantially lower than single-phase models. However, the final performance was marginally worse than baseline UNet, suggesting that the learned image enhancements do not provide task-specific improvements for segmentation.

*Atrous Enhancer*: Phase 1 required the full planned #conv.atrous_enhancer.phase1_epochs epochs and achieved substantial loss reduction (#conv.atrous_enhancer.phase1_reduction_percent%), demonstrating that the multi-scale ASPP-based enhancer requires more training to learn effective transformations across different receptive fields. Phase 2 (#conv.atrous_enhancer.phase2_epochs epochs) provided an additional #conv.atrous_enhancer.phase2_reduction_percent% reduction. Despite the extended training (#conv.atrous_enhancer.total_epochs total epochs), performance remained below baseline, indicating that even sophisticated multi-scale preprocessing cannot match task-specific architectural improvements.

*Key Convergence Insights:*

+ *Architectural efficiency matters*: ASPP-UNet's #(conv.aspp_unet.convergence_epoch)-epoch convergence vs. #(conv.unet.convergence_epoch)-#(conv.clahe_unet.total_epochs) epochs for baseline models demonstrates that integrating multi-scale features directly into the segmentation architecture is more effective than preprocessing-based approaches.

+ *Enhancement preprocessing shows limited benefit*: Both learned enhancement approaches required comparable or greater total training time than direct segmentation, while achieving lower final performance. This suggests that task-agnostic image enhancement is less effective than task-specific architectural improvements.

+ *Training strategy complexity*: The two-phase training adds complexity without performance gains. Phase 1's frozen UNet prevents the enhancer from learning task-specific transformations, while Phase 2's joint training cannot fully recover from the suboptimal Phase 1 initialization.


== Per-Class Performance Analysis

#let baseline_iou_bg = data.baseline_unet.iou_bg
#let baseline_iou_disc = data.baseline_unet.iou_disc
#let baseline_iou_cup = data.baseline_unet.iou_cup

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: center,
    table.header([*Approach*], [*#sym.Delta IoU BG*], [*#sym.Delta IoU Disc*], [* #sym.Delta IoU Cup*]),
    [CLAHE],
    format_number(data.clahe.iou_bg - baseline_iou_bg, "#sym.Delta  mIoU"),
    format_number(data.clahe.iou_disc - baseline_iou_disc, "#sym.Delta  mIoU"),
    format_number(data.clahe.iou_cup - baseline_iou_cup, "#sym.Delta  mIoU"),

    [#sym.plus Std Enhancer],
    format_number(data.baseline_std_enhancer.iou_bg - baseline_iou_bg, "#sym.Delta  mIoU"),
    format_number(data.baseline_std_enhancer.iou_disc - baseline_iou_disc, "#sym.Delta  mIoU"),
    format_number(data.baseline_std_enhancer.iou_cup - baseline_iou_cup, "#sym.Delta  mIoU"),

    [#sym.plus Atrous Enhancer],
    format_number(data.baseline_atrous_enhancer.iou_bg - baseline_iou_bg, "#sym.Delta  mIoU"),
    format_number(data.baseline_atrous_enhancer.iou_disc - baseline_iou_disc, "#sym.Delta  mIoU"),
    format_number(data.baseline_atrous_enhancer.iou_cup - baseline_iou_cup, "#sym.Delta  mIoU"),

    [ASPP-UNet],
    format_number(data.aspp_unet.iou_bg - baseline_iou_bg, "#sym.Delta  mIoU"),
    format_number(data.aspp_unet.iou_disc - baseline_iou_disc, "#sym.Delta  mIoU"),
    format_number(data.aspp_unet.iou_cup - baseline_iou_cup, "#sym.Delta  mIoU"),
  ),
  caption: [Class-specific IoU improvements over baseline UNet. ASPP-UNet shows consistent improvements across all classes, with the largest gains in background segmentation.],
) <fig_class_specific>

*Analysis:*

// evaluate in the end again. probably this could also be counted as error margin
- *Cup Segmentation (most challenging):* ASPP-UNet achieves #calc.round(data.aspp_unet.iou_cup * 100, digits: 2)% IoU vs. #calc.round(baseline_iou_cup * 100, digits: 2)% baseline (+#format_number(data.aspp_unet.iou_cup - baseline_iou_cup, "#sym.Delta  mIoU") points), demonstrating that multi-scale features help with the hardest class

- *Disc Segmentation:* All approaches achieve >#calc.round(calc.min(..approaches.map(a => data.at(a).iou_disc)) * 100, digits: 1)%, with ASPP-UNet reaching #calc.round(data.aspp_unet.iou_disc * 100, digits: 2)%

- *Background:* High performance across all methods (>#calc.round(calc.min(..approaches.map(a => data.at(a).iou_bg)) * 100, digits: 1)%), with ASPP-UNet reaching #calc.round(data.aspp_unet.iou_bg * 100, digits: 2)%

== Qualitative Results
Visual inspection of predictions reveals:

=== Frayed Edges <chapt_frayed_edges>
#figure(
  image("../results/sample_comparison_160.png"),
  caption: [
    Visualization of segmentation (OD in blue; OC in red) outputs from Baseline UNet (left) and ASPP-UNet (right) on a sample image next to the raw image and ground truth mask.
  ],
) <fig_sample_comparison_160>

In @fig_sample_comparison_160, we compare segmentation outputs from the baseline UNet and ASPP-UNet on representative test images. Similar to the findings from #cite(<src_zedan2025rmhanetrobustoptic>, form: "prose"), ASPP-UNet produces smoother and more anatomically plausible boundaries for both the optic disc and cup in comparison to the frayed edges produced by the baseline UNet. The multi-scale features learned by the ASPP module help capture fine vessel structures and disc edges that the baseline UNet often misses or segments poorly.


=== Challenging Samples
#figure(
  image("../results/sample_comparison_13.png"),
  caption: [
    Segmentation output for test sample 13, which has low contrast and poor cup visibility. .
  ],
) <fig_sample_comparison_13>

In @fig_sample_comparison_13, we observe that both models perform reasonably well, with ASPP-UNet capturing the cup boundary edges less frayed than the baseline like already shown in



=== Extreme Cases
In general both models perform very well with small to medium sized optic discs. However, in the case of extremely large optic discs and cups, both models sometimes struggle to accurately capture the full extent of the cup region.
#figure(
  image("../results/sample_comparison_52.png"),
  caption: [
    Segmentation output for test sample 52, which has an extremely large optic cup making accurate segmentation very challenging. Both models struggle to capture the full extent of the cup.
  ],
) <fig_sample_comparison_52>


#figure(
  image("../results/sample_comparison_56.png"),
  caption: [
    Segmentation output for test sample 56, which has a very large optic disc and cup. Both models perform reasonably well, but ASPP-UNet captures the cup boundary more accurately.
  ],
) <fig_sample_comparison_56>

=== Enhancement Visualization
To understand how the enhancers modify input images, we visualize enhanced outputs and difference heatmaps for representative test samples. The heatmaps highlight regions where the enhancer made significant changes compared to the original image. The heatmap is computed as the mean absolute difference across RGB channels. Dark areas indicate minimal changes, while bright areas show strong modifications. Because the images are normalized with ImageNet, but for the visualization denormalized back to [0, 1] range, a e.g. 0.15 difference would mean a 15% change in pixel intensity.


#figure(
  image("../results/std_enhancer_top5_sample_218.png"),
  caption: [
    Standard enhanced image (middle) next to the original (left) and the difference heatmap (right) for test sample 218.
  ],
) <fig_std_enhancer_top5_sample_218>

In @fig_std_enhancer_top5_sample_218, the standard enhancer focuses on lightening the background and blood vessels, whilst the optic disc and cup regions see less modification. The difference heatmap also shows a checkerboard pattern, indicating that the enhancer applies localized contrast adjustments to enhance vessel visibility.

#figure(
  image("../results/atrous_enhancer_top2_sample_391.png"),
  caption: [
    Atrous enhanced image (middle) next to the original (left) and the difference heatmap (right) for test sample 391.
  ],
) <fig_atrous_enhancer_top2_sample_391>

In @fig_atrous_enhancer_top2_sample_391, the atrous enhancer also applies more significant changes to the background and vessels, with less focus on the disc and cup areas. In comparision to the standard enhancer, the atrous version does not show the checkers pattern, indicating a different enhancement strategy.


// TODO: zeigen, dass auch verrauschte bilder gut funktionieren. Das liegt daran, dass wir gut mit Augmentation gearbeitet haben

// TODO: auch mal 1-2 failure cases zeigen.

// dieses differenz bild zeigen, wie die enhancer arbeiten

// 13 hat sehr undeutlich; trotzdem gute segmentierung

// 52 ultra schlecht. Da hab ich von der ground truth fast nichts erwischt.

// 56 aber z.B. auch große gt; aber da hab ich nicht so schlecht performed


= Discussion

= Conclusions and Future Work

// TODO: irgendwie gibt es so ein satz von machinelearning so nach dem Motto "Keep it simple"



// TODO: comparison with SOTA paper why our models are so much worse? Are they really worse? Or are they just calculating there metrics different. We are using IoU on cropped roi. Are they using Dice of Full image? -> roi smaller therefore hit rate way easier!

// TODO: noch mal die anderen beiden dokumente (gdoc und notes.typ) durchschauen, ob da noch was verwertbares dabei ist

// TODO: further research: ich glaube das trainings material an sich ist nicht perfekt. Vielleicht könnte man bei REFUGE unstimmigkeiten zwischen den verschiedenen leuten die labeln das mit in die Loss funktion mit rein packen.

// TODO: Ich glaube auch, dass unser model probleme hat, wenn das schon sehr fortgeschritten ist. Also OC:OD gegen 1:1. vielleicht das auch irgendwie mit in die Loss funktion packen, dass hohe ratio stärker gewichtet wird

// TODO: Auch das wir aufgrund der begrenzten hardware ressourcen das Ding nicht mit mehr auflösung trainieren konnte. Welche haben wir überhaupt jetzt benutzt? 256 oder 512?



// TODO: großes problem würde ich wirklich sagen, die trainingsdaten. Ich bin selbst kein augenarzt, aber das ist schon teilweise wirklich sehr sehr schwer zu erkennen.

// TODO: Link github repo:
// should we also upload the trained models somewhere?


// TODO: ganz am Ende schauen, ob wir noch irgendwo fest zahlen haben

// Oben wo clahe erklärt wird kann man bestimmt mal schön ein vergleichsbild rein machen.
