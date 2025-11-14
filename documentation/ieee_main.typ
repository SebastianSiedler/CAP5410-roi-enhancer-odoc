#import "@preview/charged-ieee:0.1.4": ieee


// Load comparison data from JSON
#let data = json("../results/test_comparison.json")
#let approaches = data.keys()
#let metrics = data.aspp_unet.keys() // just any approach to get metric names


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
Glaucoma is one of the leading causes of irreversible blindness worldwide @Wagner.2022. It is often progressing without noticeable symptoms until significant vision loss has occurred. Therefore, accurate and early detection is essential for effective treatment of the disease @Wagner.2022. Fundus imaging is a non-invasive technique that captures detailed images of the retina, showing the optic disc (OD) and optic cup (OC), which are critical for detecting glaucoma diagnosis. The Cup-to-Disc Ratio (CDR) is a crucial metric used to diagnose glaucoma by measuring the relative size of the OC to the OD.

Automated segmentation of the OD and OC in fundus images has become an important research area in medical image analysis. However, accurate segmentation remains challenging due to imperfections in the images, such as low contrast, uneven illumination, and blur @Lin.2025. To mitigate these issues, the images are often preprocessed using image enhancement techniques, improving the visibility of relevant anatomical features and therefore more accurate segmentation results @Lin.2025. Most existing enhancement methods are designed as independent preprocessing modules that are optimized separately from the actual segmentation task. As a result, the enhancement focuses primarily on visual quality rather than task-specific feature optimization.

In this work, we propose a task-aware learned multi-scale enhancement model that is jointly trained with the segmentation model, allowing the enhancer to learn features that are specifically relevant for accurate OD and OC segmentation#footnote[Code available at: #link("https://github.com/SebastianSiedler/CAP5410-roi-enhancer-odoc")]. // TODO: check at the end

The paper is structured as follows: Section II reviews related work on glaucoma image segmentation, showing the gap in existing enhancement methods. Section III describes the proposed methodology, including details about the dataset, network architectures, loss functions, training strategy, and evaluation metrics. Section IV presents experimental results comparing different approaches. Finally, Section V discusses the findings and concludes the paper with future work suggestions.


/*
This design allows us to isolate the contributions of preprocessing-based vs. architecture-based multi-scale feature learning.

This study investigates whether task-specific architectural improvements or learned preprocessing enhancements are more effective for optic disc and cup segmentation. Therefore, we compare five approaches:

+ Standard UNet baseline
+ CLAHE preprocessing + UNet
+ Learned standard enhancer + UNet
+ Learned atrous (ASPP-based) enhancer + UNet
+ ASPP-UNet with architectural multi-scale integration
*/

= Related Work
== Glaucoma Image Segmentation
// Which models used for OD/OC Segmentation?
Most segmentation models used in related works are based on the UNet architecture due to its encoder-decoder architecture with skip connections that preserve high-level semantic information while also retaining low-level spatial details @src_ronneberger2015unetconvolutionalnetworksbiomedical.
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

//TODO: highlight that RMHA-Net also uses ASPP

= Methodology
== Dataset
The dataset used in this study is a combination of three publicly available glaucoma fundus image datasets to increase the diversity and size of the training data:
+ G1020 Dataset: 1,020 fundus images with optic disc and cup annotations from private clinical practice in Kaiserslautern @src_bajwa2020g1020benchmarkretinalfundus.

+ ORIGA Dataset: 650 fundus images (482 normal, 168 glaucoma) from the Singapore Eye Research Institute @src_Origa.

+ REFUGE Challenge Dataset: 1,200 fundus images from the Retinal Fundus Glaucoma Challenge @src_REFUGE_dataset, including training, validation, and test sets.

//*Preprocessing:* \
We obtained these three datasets from the "Glaucoma Fundus Imaging Datasets" Kaggle repository#footnote("https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets/data"). The images were already preprocessed to a certain extent, including cropping around the Region of Interest (ROI) centered on the optic disc. The labeling of the masks was done with three classes: background, optic disc, and optic cup.
/*
- 0: background
- 1: optic disc
- 2: optic cup
*/

//*Data Split:* \
The dataset was divided into training, validation, and testing sets with a 70/15/15 split, using a random seed of 42 to ensure reproducibility. The combination of all datasets, led to a total number of 2,870 images. The splits were stratified to maintain the proportion of glaucoma and normal cases across all subsets. Also, a small number of images had incomplete masks, e.g. missing optic cup or disc annotations which we filtered out to ensure the quality of the training data.

This resulted in the following distribution:
- Total images before filtering: 2870
- Images with incomplete masks: 234
- Total images after filtering: 2636
- Training set: 1845 images
- Validation set: 395 images
- Test set: 396 images

== Data Augmentation
To improve the generalization and robustness of our model, we applied a series of data augmentations during training. These included flipping images randomly horizontally and vertically with a probability of 0.5, rotation within $plus.minus$20 degrees, as well as brightness and contrast adjustments within $plus.minus$0.2. All images were additionally normalized using ImageNet statistics.

The augmentations were implemented using the Albumentations library @src_2018arXiv180906839B in the file transforms.py. For the validation and test sets, only normalization was applied, with no additional augmentations, to ensure consistent evaluation.
// TODO What is transform.py and should we mention it?

== Network Architectures

This study investigates whether task-specific architectural improvements or learned preprocessing enhancements are more effective for optic disc and cup segmentation. Therefore, we compare five approaches:

+ Standard UNet baseline
+ CLAHE preprocessing + UNet
+ Learned standard enhancer + UNet
+ Learned atrous (ASPP-based) enhancer + UNet
+ ASPP-UNet with architectural multi-scale integration

/*
(1) Standard UNet baseline, (2) CLAHE preprocessing + UNet, (3) learned standard enhancer + UNet, (4) learned atrous (ASPP-based) enhancer + UNet, and (5) ASPP-UNet with architectural multi-scale integration.
*/
This design allows us to isolate the contributions of preprocessing-based vs. architecture-based multi-scale feature learning.

=== Baseline Architecture

Our baseline follows the standard UNet architecture @src_ronneberger2015unetconvolutionalnetworksbiomedical with four encoder-decoder levels. The encoder progressively downsamples the input through max pooling while increasing channel dimensions (64 → 128 → 256 → 512 → 1024). Each encoder stage consists of two 3×3 convolutions with batch normalization and ReLU activation. The decoder uses transposed convolutions for upsampling, concatenating skip connections from corresponding encoder stages. The final 1×1 convolution produces three-class segmentation (background, disc, cup). This baseline contains 31.0M parameters.


=== Preprocessing-Based Approaches

// *CLAHE-UNet:*
Three preprocessing approaches are evaluated by placing them before the baseline UNet.

The first preprocessing approach applies traditional CLAHE to enhance local contrast before feeding images to the baseline UNet. This traditional computer vision technique aims to improve visibility of disc and cup boundaries in low-contrast fundus images.

//*Standard Enhancer:*
As a second preprocessing technique a standard enhancer was tested. It consists out of a lightweight encoder-decoder network with 52K parameters that learns to enhance input images for improved segmentation. The enhancer has three encoder blocks (64, 128, 256 channels) with max pooling, followed by two decoder blocks with transposed convolutions. Enhanced images are combined with originals via learnable residual connection: $I_"enhanced" = alpha dot "Enhancer"(I) + beta dot I$. The system uses two-phase training: Phase 1 trains only the enhancer with frozen UNet; Phase 2 jointly fine-tunes both components.

//*Atrous Enhancer:*
Finally, the atrous enhancer was evaluated. The standard enhancer was replaced with an ASPP-based multi-scale enhancer with 26K parameters. Instead of spatial downsampling, it uses parallel atrous convolutions with dilation rates [1, 3, 6] to capture features at multiple scales simultaneously. The ASPP module concatenates outputs from: 1×1 convolution, 3×3 atrous convolutions at different rates, and global average pooling. This enables the enhancer to learn multi-scale image transformations without losing spatial resolution.

=== Architecture-Based Approach

Instead of using a separate enhancement module, ASPP-UNet integrates  multi-scale feature learning directly into the segmentation network by replacing the UNet bottleneck with an ASPP module. The ASPP operates on 1024-channel features with dilation rates [1, 6, 12, 18], capturing fine details, medium structures, and global context simultaneously. Crucially, these multi-scale features are learned end-to-end for the segmentation task, not for generic image enhancement. Despite the additional multi-scale processing, ASPP-UNet contains only #calc.round(data.aspp_unet.parameters / 1000000, digits: 2)M parameters (#calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% fewer than baseline), as the ASPP module is more parameter-efficient than the baseline's double convolution bottleneck.




== Loss Functions


The loss function for the segmentation model is based on a weighted combination of Cross-Entropy and Dice loss:

$ L_"seg" = 0.5 dot L_"CE" + 0.5 dot L_"Dice" $

The Cross-Entropy Loss is weighted by class weights:

$ L_"CE" = -sum_(c=1)^C w_c sum_i y_"i,c" log(hat(y)_"i,c") $


where $y_"i,c"$ represents the ground-truth label at pixel i for class c, and $hat(y)_"i,c"$ denotes the predicted probability for the same pixel and class, while $w_c$ are class weights $[1.0, 1.0, 2.0]$ to handle cup class imbalance.

The Dice Loss is defined as

$ L_"Dice" = 1 - (2 sum_i y_i hat(y)_i + epsilon)/(sum_i y_i + sum_i hat(y)_i + epsilon) $

with $epsilon = 1$ for numerical stability.


//==== Enhancement Loss (Approaches 2 & 3)

For the enhancer-based approaches, we add an L1 regularization term to the segmentation loss:

$ L_"total" = L_"seg" + lambda dot L_"L1" $

with the L1 term defined as

$ L_"L1" = 1/N sum_i |I_"enhanced"^(i) - I_"original"^(i)|. $

To encourage only minimal deviation from the original image, $lambda$ is set to $0.001$.



== Training Strategy
// TODO: for whole chapter. check all parameters again in the end

=== Single-Phase Training for Baseline UNet Variants

The training of the segmentation models was performed using a single-phase approach for the three UNet variants, namely the Standard UNet, CLAHE-UNet, and ASPP-UNet. In this setting, each model was trained in a single continuous phase without any separate pretraining or fine-tuning steps. All three networks were optimized using the Adam optimizer with a learning rate of $10^(-4)$ and $beta$ parameters of $(0.9, 0.999)$, with a batch size of 16. For stable convergence the ReduceLROnPlateau learning rate scheduler was applied with a factor of 0.5 and patience of 5 epochs.

The Standard UNet and CLAHE UNet were both trained for 100 epochs without early stopping, as both models showed continued improvement throughout the entire training duration.

The ASPP-UNet training used early stopping with a patience of 15 epochs to reduce overfitting. Although the maximum number of epochs was set to 100, the training stopped after 67 epochs due to early stopping. Despite this shorter training time, the ASPP-UNet converged faster than the baseline models, demonstrating superior training efficiency.


=== Two-Phase Training of Enhancer-Based Approaches

The enhancer-based models were trained in two phases.

The first phase focused on only training the enhancer module while the UNet segmentation backbone remained frozen with pretrained weights. This phase aimed to let the enhancer learn useful image transformations without affecting the segmentation model. The Standard enhancer was trained for 30 epochs, but stopped early after 26 epochs, with an early stopping patience of 10 epochs. The Atrous enhancer required a longer training period with 50 epochs with no early stopping triggered. The patience was set to 15 epochs for this model. For both models the Adam optimizer with a learning rate of $10^(-4)$ and the combined segmentation + L1 loss were used.

In the second phase, both the enhancement model and the segmentation model were trained jointly to fine-tune the entire pipeline. The Standard enhancer was trained for 20 epochs with no early stopping triggered, while the Atrous enhancer was trained for 30 epochs but stopped early after 25 epochs. The learning rate was reduced to $10^(-5)$ to allow gradual adaptation, while the same segmentation + L1 loss was applied.



== Evaluation Metrics
We evaluated segmentation performance using standard segmentation metrics, namely the Intersection over Union (IoU) and the Dice Coefficient.

//*Intersection over Union (IoU)* per class:
The IoU for each class $c$ is defined as

$ "IoU"_c = (Y_c inter hat(Y)_c)/(Y_c union hat(Y)_c) $

where $Y_c$ is the ground truth mask for class c, while $hat(Y)_c$ denotes the predicted mask for class c.

//*Mean IoU (mIoU)*:
The mean IoU (mIoU) is then computed as the average over all $C$ classes:

$ "mIoU" = 1/C sum_(c=1)^C "IoU"_c $

//*Dice Coefficient* (alternative metric):
As an additional metric, the Dice Coefficient was calculated for each class $c$ as an evaluation for segmentation overlap:

$ "Dice"_c =(2 |Y_c inter hat(Y)_c|)/(|Y_c|+|hat(Y_c)|) $

We used mIoU as the primary evaluation metric throughout this study to enable fair and consistent comparison across all approaches.


== Implementation Details // Brauchen wir das?

All experiments were conducted on a workstation equipped with an NVIDIA GeForce RTX 2080 Ti GPU (11 GB VRAM), an Intel Core i7-8700K CPU (3.70 GHz), and 32 GB of system memory. The models were implemented and trained using the PyTorch deep learning framework.





= Experimental Results

== Quantitative Comparison

@fig_quantitative_comparison presents comprehensive validation set results for all five approaches.

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
    caption: [Quantitative comparison of all five approaches on the test set. ],
  ) <fig_quantitative_comparison>
]


Key Observations:
+ *Best Performance:* ASPP-UNet achieves #calc.round(data.aspp_unet.miou * 100, digits: 2)% mIoU, improving upon baseline by #calc.round(data.aspp_unet.delta_miou * 100, digits: 2) percentage points (pp), corresponding to a relative improvement of #calc.round((data.aspp_unet.miou - data.baseline_unet.miou) / data.baseline_unet.miou * 100, digits: 2)%.

+ *Parameter Efficiency:* ASPP-UNet uses #calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% fewer parameters (#calc.round(data.aspp_unet.parameters / 1000000, digits: 2)M vs #calc.round(data.baseline_unet.parameters / 1000000, digits: 2)M) while outperforming all other approaches

+ *Training Efficiency:* ASPP-UNet demonstrated superior convergence, reaching optimal performance in 67 epochs compared to 100 epochs required by baseline models. This faster convergence, combined with better final performance, indicates that the ASPP module provides a more effective inductive bias for this segmentation task.

+ *Enhancement Performance:* The standard enhancer shows slight performance improvements (+#calc.round(data.baseline_std_enhancer.delta_miou * 100, digits: 2)%) compared to the baseline UNet, demonstrating that learned preprocessing can provide marginal benefits.

+ *CLAHE Preprocessing*: Surprisingly, CLAHE preprocessing leads to a minor decrease in performance compared to the baseline UNet (#calc.round(data.clahe.delta_miou * 100, digits: 2) percentage points), suggesting that traditional contrast enhancement techniques may not be optimal when combined with deep learning models that can learn their own feature representations.

+ *Multi-Scale Enhancement Limitation*: The atrous enhancer shows marginal degradation (#calc.round(data.baseline_atrous_enhancer.delta_miou * 100, digits: 2)%) compared to baseline, performing worse than the standard enhancer. This contradicts the hypothesis that multi-scale preprocessing helps segmentation performance. The result suggests that task-agnostic multi-scale image enhancement does not translate to improved segmentation accuracy, even when using sophisticated ASPP-based architectures

+ *Class-Specific Analysis*:
  - Background: ASPP-UNET shows largest improvement (+#calc.round((data.aspp_unet.iou_bg - data.baseline_unet.iou_bg) * 100, digits: 2)% points)
  - Disc: ASPP-Unet improves by +#calc.round((data.aspp_unet.iou_disc - data.baseline_unet.iou_disc) * 100, digits: 2)% points
  - ASPP-UNet improves by +#calc.round((data.aspp_unet.iou_cup - data.baseline_unet.iou_cup) * 100, digits: 2)% points (most challenging class)

== Training Convergence Analysis
The training dynamics of each approach provide insights into their learning efficiency and stability. We analyze convergence patterns using loss curves and key metrics such as best validation loss, convergence epoch, and overfitting behavior.
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


*Standard Enhancer*: Phase 1 (#conv.std_enhancer.phase1_epochs epochs with early stopping) achieved modest loss reduction (#conv.std_enhancer.phase1_reduction_percent%), while Phase 2 joint fine-tuning (#conv.std_enhancer.phase2_epochs epochs) yielded #conv.std_enhancer.phase2_reduction_percent% additional reduction. The total training time of #conv.std_enhancer.total_epochs epochs is substantially lower than single-phase models. The final performance shows marginal improvement over baseline UNet (+0.26 pp), suggesting that learned image enhancements provide only limited task-specific benefits for segmentation.

#figure(
  image("../results/standard_enhancer_two_phase.png", width: 80%),
  caption: [Two-phase training curve for Standard Enhancer. Phase 1 (blue/purple, UNet frozen) and Phase 2 (orange/red, joint training) are shown with a clear transition point at epoch #conv.std_enhancer.phase1_epochs. The model completed training in #conv.std_enhancer.total_epochs total epochs.],
) <fig_std_enhancer_two_phase>


*Atrous Enhancer*: Phase 1 required the full planned #conv.atrous_enhancer.phase1_epochs epochs and achieved substantial loss reduction (#conv.atrous_enhancer.phase1_reduction_percent%), demonstrating that the multi-scale ASPP-based enhancer requires more training to learn effective transformations across different receptive fields. Phase 2 (#conv.atrous_enhancer.phase2_epochs epochs) provided an additional #conv.atrous_enhancer.phase2_reduction_percent% reduction. Despite the extended training (#conv.atrous_enhancer.total_epochs total epochs), performance shows marginal degradation compared to baseline (-0.01 pp), indicating that even sophisticated multi-scale preprocessing cannot match task-specific architectural improvements. This result challenges the assumption that multi-scale preprocessing inherently benefits segmentation tasks.


#figure(
  image("../results/atrous_enhancer_two_phase.png", width: 80%),
  caption: [Two-phase training curve for Atrous Enhancer. Phase 1 (blue/purple, UNet frozen) and Phase 2 (orange/red, joint training) are shown with a clear transition point at epoch #conv.atrous_enhancer.phase1_epochs. The model required #conv.atrous_enhancer.total_epochs total epochs.],
) <fig_atrous_enhancer_two_phase>

*Key Convergence Insights:*

+ *Architectural efficiency matters*: ASPP-UNet's #(conv.aspp_unet.convergence_epoch)-epoch convergence vs. #(conv.unet.convergence_epoch)-#(conv.clahe_unet.total_epochs) epochs for baseline models demonstrates that integrating multi-scale features directly into the segmentation architecture is more effective than preprocessing-based approaches.

+ *Enhancement preprocessing shows limited benefit*: The standard enhancer achieved only marginal improvement (+0.26 pp), while the atrous enhancer showed slight degradation (-0.01 pp). Both approaches required comparable or greater total training time than direct segmentation. This demonstrates that task-agnostic image enhancement—even with sophisticated multi-scale architectures—provides minimal benefits compared to task-specific architectural improvements.

+ *Training strategy complexity*: The two-phase training adds complexity without substantial performance gains. Phase 1's frozen UNet prevents the enhancer from learning task-specific transformations, while Phase 2's joint training cannot fully recover from the suboptimal Phase 1 initialization. The multi-scale atrous enhancer paradoxically underperforms the simpler standard enhancer, suggesting that architectural complexity in the preprocessing stage does not translate to better segmentation performance.


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
In @fig_sample_comparison_160, we compare segmentation outputs from the baseline UNet and ASPP-UNet on representative test images. Similar to the findings from #cite(<Zedan.2025>, form: "prose"), ASPP-UNet produces smoother and more anatomically plausible boundaries for both the optic disc and cup in comparison to the frayed edges produced by the baseline UNet. The multi-scale features learned by the ASPP module help capture fine vessel structures and disc edges that the baseline UNet often misses or segments poorly.

#figure(
  image("../results/sample_comparison_160.png"),
  caption: [
    Visualization of segmentation (OD in blue; OC in red) outputs from Baseline UNet (left) and ASPP-UNet (right) on a sample image next to the raw image and ground truth mask.
  ],
) <fig_sample_comparison_160>




=== Challenging Samples

In @fig_sample_comparison_13, we observe that both models perform reasonably well, with ASPP-UNet capturing the cup boundary edges less frayed than the baseline like already shown in @chapt_frayed_edges. Despite the low contrast and poor cup visibility in this sample, both models manage to segment the optic disc and cup regions effectively, demonstrating robustness to challenging imaging conditions. This is likely due to the diverse training data and effective data augmentation strategies employed during training.

#figure(
  image("../results/sample_comparison_13.png"),
  caption: [
    Segmentation output for test sample 13, which has low contrast and poor cup visibility.
  ],
) <fig_sample_comparison_13>


=== Extreme Cases
In general, both models perform very well with small to medium sized optic discs. However, in the case of extremely large optic discs and cups, both models sometimes struggle to accurately capture the full extent of the cup region.
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
  image("../results/atrous_enhancer_top1_sample_391.png"),
  caption: [
    Atrous enhanced image (middle) next to the original (left) and the difference heatmap (right) for test sample 391.
  ],
) <fig_atrous_enhancer_top1_sample_391>
In @fig_atrous_enhancer_top1_sample_391, the atrous enhancer also applies more significant changes to the background and vessels, with less focus on the disc and cup areas. In comparision to the standard enhancer, the atrous version does not show the checkers pattern, indicating a different enhancement strategy.


= Discussion
== Main Finding: Efficiency Through Architectural Integration

Our results demonstrate that ASPP-UNet represents a more efficient approach than preprocessing-based methods for OD and OC segmentation. While the quantitative improvement is modest (+#calc.round(data.aspp_unet.delta_miou * 100, digits: 2)% mIoU), the efficiency gains are substantial and practically meaningful:

*Parameter Efficiency:* ASPP-UNet achieves #calc.round(data.aspp_unet.miou * 100, digits: 2)% mIoU with #calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% fewer parameters than the baseline (#calc.round(data.aspp_unet.parameters / 1000000, digits: 2)M vs. #calc.round(data.baseline_unet.parameters / 1000000, digits: 2)M). This dramatic reduction enables deployment on resource-constrained devices and allows for larger batch sizes during training.

*Training Efficiency:* ASPP-UNet converged in #conv.aspp_unet.total_epochs epochs compared to #conv.unet.total_epochs epochs for the baseline—a #calc.round((1 - conv.aspp_unet.total_epochs / conv.unet.total_epochs) * 100, digits: 0)% reduction in training time. This faster convergence translates to reduced computational costs and faster iteration cycles during model development.

*Comparison with Enhancement Approaches:* The preprocessing-based methods showed minimal or negative impact: CLAHE (#calc.round(data.clahe.delta_miou * 100, digits: 2)%), standard enhancer (+0.26 pp), and atrous enhancer (-0.01 pp). These marginal differences suggest that task-agnostic image enhancement provides limited benefits when combined with deep learning models that can learn their own feature representations.

The key insight is that architectural changes allowing the model to learn multi-scale features directly optimized for segmentation prove more efficient than separating enhancement and segmentation into distinct preprocessing and analysis stages. ASPP-UNet's multi-scale features are learned end-to-end specifically to maximize segmentation performance, while preprocessing focuses on generic image quality that may not align with task-specific needs.

== Revisiting the Role of Preprocessing

*CLAHE Preprocessing:* Our results show that CLAHE preprocessing led to a performance decrease (#calc.round(data.clahe.delta_miou * 100, digits: 2)%) compared to the baseline UNet. This finding contrasts with some findings in the literature, where CLAHE has been reported as beneficial for optic disc and cup segmentation. Notably, RMHA-Net @Zedan.2025 and Multi-GlaucNet @HaorenXiong.2025—both already cited in this work—successfully employed CLAHE as a preprocessing step and achieved state-of-the-art results. Additional studies have also demonstrated CLAHE's effectiveness for various fundus image analysis and retinal segmentation tasks.

This apparent contradiction may be explained by several factors. First, our 256×256 resolution may have reduced the need for explicit contrast enhancement, as important features are already visible at this scale. Second, the UNet baseline's convolutional layers may have learned sufficient contrast normalization internally, making explicit preprocessing redundant or even detrimental for our specific pipeline. Third, different datasets, architectures, and training procedures may respond differently to CLAHE preprocessing. Our finding suggests that CLAHE's benefits are not universal and depend on the specific modeling context.

*Learned Enhancers:* The learned enhancement approaches showed minimal impact: the standard enhancer achieved only +0.26 percentage points improvement, while the atrous enhancer showed -0.01 percentage points degradation. These marginal differences fall within typical performance variance and suggest that task-agnostic image enhancement—even when jointly trained with the segmentation model—provides limited benefits for this task. In separate experiments with different random seeds, the standard enhancer sometimes degraded performance compared to baseline, further demonstrating the instability of this approach.

== Why ASPP-UNet Succeeds: Efficiency and Task-Specific Features

ASPP-UNet's superior performance stems from two key advantages: efficiency and task-specific multi-scale learning.

*Parameter Efficiency:* ASPP-UNet achieves #calc.round(data.aspp_unet.miou * 100, digits: 2)% mIoU with only #calc.round(data.aspp_unet.parameters / 1000000, digits: 2)M parameters—#calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% fewer than the baseline's #calc.round(data.baseline_unet.parameters / 1000000, digits: 2)M parameters. This dramatic reduction in model size, while simultaneously improving performance, suggests that the ASPP bottleneck is a more efficient parameterization for capturing the relevant features for this task. The smaller parameter count likely aids generalization by reducing overfitting risk.

*Training Efficiency:* ASPP-UNet converged in only #conv.aspp_unet.total_epochs epochs compared to #conv.unet.total_epochs epochs required by the baseline UNet—a #calc.round((1 - conv.aspp_unet.total_epochs / conv.unet.total_epochs) * 100, digits: 0)% reduction in training time. This faster convergence demonstrates that the ASPP module provides a better inductive bias for this segmentation task, enabling the model to learn effective representations more quickly.

== Class-Specific Insights
The most challenging class for all models was the segmentation of the optic cup, due to its small size and low contrast boundaries (especially in glaucoma cases). Also the high anatomical variability of the cup shape makes accurate segmentation difficult.

== Qualitative Observations
Visual inspection revealed that ASPP-UNet produces smoother and more anatomically plausible boundaries, particularly for the optic cup. The multi-scale features help capture fine vessel structures and disc edges that the baseline UNet often misses or segments poorly. Both models struggled with extremely large optic discs and cups, indicating areas for future improvement. The extrem cases could also be due to the low 256x256 resolution we used for training.

== Comparison with State-of-the-Art

// TODO: @SteffEng-lab
// TODO: comparison with SOTA paper why our models are so much worse? Are they really worse? Or are they just calculating there metrics different. We are using IoU on cropped roi. Are they using Dice of Full image? -> roi smaller therefore hit rate way easier!


To compare our results with existing methods, we use performance and architectural complexity as key criteria. The key finding is that while some competing models achieve higher absolute metrics, they often rely on auxiliary complexity or external preprocessing.

This focus on architectural improvement is mirrored by models like EE-TransUNet @Liu.2025. This edge-focused model achieved high Dice scores (OD 0.967, OC 0.9056 on REFUGE) purely through internal feature enhancement modules (CCF and CSMF blocks) without relying on explicit image enhancement. This finding correlates with our conclusion that architectural improvements are more effective than decoupled preprocessing.

Similar trends are observed in multi-scale extensions of UNet. For example, ASPP-enhanced variants like RMHA-Net @Zedan.2025 often yield smoother and more anatomically plausible OD and OC boundaries compared to the baseline UNet, which tends to produce incomplete edges. While simpler architectures often miss fine vascular structures and subtle disk contours, the added multi-scale features of this complex architecture help to capture these details.

Our assessment of CLAHE-based preprocessing further highlights the limitations of relying on separate enhancement steps. Our results show a measurable reduction in segmentation quality when applying CLAHE before training. This outcome stands in contrast to several state-of-the-art models such as RMHA-Net @Zedan.2025 and Multi-GlaucNet @HaorenXiong.2025, which successfully integrated CLAHE and achieved notable performance gains. While prior work has demonstrated that CLAHE can be beneficial for certain fundus-analysis tasks, our results suggest that its effectiveness is highly model-dependent and can be outweighed by stronger architectural refinements.



= Conclusions and Future Work

This work systematically investigated whether learned image enhancement or direct architectural improvements better support optic disc and cup segmentation in fundus images. Our findings suggest that task-specific architectural integration is a more efficient approach than preprocessing-based methods—whether traditional or learned. While the quantitative performance differences are modest, the efficiency gains in terms of parameters and training time represent a meaningful practical advantage.

== The Limited Benefit of Enhancement-Based Approaches

The comprehensive evaluation reveals that all three enhancement strategies provided minimal or negative performance impact:

*Traditional Preprocessing (CLAHE):* Despite widespread use in medical imaging literature @HaorenXiong.2025 @Zedan.2025, CLAHE preprocessing yielded #calc.round(data.clahe.delta_miou * 100, digits: 2)% mIoU decrease compared to baseline (#calc.round(data.clahe.miou * 100, digits: 2)% vs. #calc.round(data.baseline_unet.miou * 100, digits: 2)%). This contrasts with findings from other works where CLAHE proved beneficial, and suggests that the utility of contrast enhancement may depend heavily on the specific architecture, dataset, and resolution used.

*Learned Standard Enhancer:* Even with end-to-end joint training, the standard enhancer achieved only +#calc.round(data.baseline_std_enhancer.delta_miou * 100, digits: 2)% improvement—a marginal gain that falls within typical performance variance. The #calc.round((data.baseline_std_enhancer.parameters - data.baseline_unet.parameters) / 1000, digits: 0)K additional parameters and two-phase training complexity provide negligible benefit.

*Learned Multi-Scale Enhancer:* The ASPP-based atrous enhancer—designed to learn multi-scale image transformations—showed a slight performance degradation of #calc.round(calc.abs(data.baseline_atrous_enhancer.delta_miou) * 100, digits: 2)% (#calc.round(data.baseline_atrous_enhancer.miou * 100, digits: 2)% mIoU). This demonstrates that sophisticated multi-scale preprocessing architectures do not necessarily translate to improved segmentation accuracy, even when trained jointly with the segmentation model.

The fundamental issue is that enhancers optimize for task-agnostic image quality rather than segmentation-relevant features. The two-phase training strategy may further limit effectiveness: Phase 1's frozen UNet prevents the enhancer from learning truly task-specific transformations from the outset, while Phase 2's joint training cannot fully compensate for this suboptimal initialization. Visualization of enhanced images (@fig_std_enhancer_top5_sample_218, @fig_atrous_enhancer_top1_sample_391) confirms that enhancers primarily modify backgrounds and vessels rather than critical disc/cup boundaries.

== Architectural Integration: A More Efficient Approach

ASPP-UNet achieved #calc.round(data.aspp_unet.delta_miou * 100, digits: 2)% mIoU improvement (#calc.round(data.aspp_unet.miou * 100, digits: 2)%) through direct architectural integration of multi-scale features. While this quantitative gain is modest, the true strength lies in the efficiency advantages: this performance was accomplished with #calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% fewer parameters (#calc.round(data.aspp_unet.parameters / 1000000, digits: 2)M vs. #calc.round(data.baseline_unet.parameters / 1000000, digits: 2)M) and #calc.round((1 - conv.aspp_unet.total_epochs / conv.unet.total_epochs) * 100, digits: 0)% faster convergence (#conv.aspp_unet.total_epochs vs. #conv.unet.total_epochs epochs).

These efficiency gains have practical implications for deployment and training at scale. A model with #calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% fewer parameters requires less memory, enables larger batch sizes, and can run on more resource-constrained devices. The #calc.round((1 - data.aspp_unet.parameters / data.baseline_unet.parameters) * 100, digits: 2)% reduction in training time translates to faster iteration cycles during model development. Combined with the improved or comparable segmentation accuracy, ASPP-UNet represents a more efficient solution than either the baseline or any of the preprocessing-enhanced approaches.

The ASPP bottleneck learns multi-scale representations directly optimized for segmentation through true end-to-end training, which appears more effective than separating enhancement and segmentation into distinct optimization problems. This finding challenges the prevalent practice of treating enhancement as a separate preprocessing step in medical image analysis pipelines. Our results suggest that architectural improvements targeting the specific analysis task should be prioritized over generic image enhancement techniques, particularly when efficiency and deployment constraints are considerations.

== Limitations and Future Work

*Resolution Constraints:* Hardware limitations restricted training to 256×256 resolution. Higher-resolution training may improve performance, particularly for extreme cases with very large optic discs and cups where both baseline and ASPP-UNet struggled.

*Dataset Quality:* Visual inspection revealed substantial annotation ambiguity in training data, particularly for challenging cases with indistinct cup boundaries. Incorporating annotation uncertainty into the loss function (e.g., weighting by inter-annotator agreement in REFUGE dataset) could improve model robustness.

*Extreme CDR Cases:* Both models showed reduced accuracy for cup-to-disc ratios approaching 1.0, suggesting that severe glaucoma cases may benefit from specialized loss weighting or dedicated model branches.

*Future Directions:* Beyond addressing these limitations, future work should explore whether our findings generalize to other medical image segmentation tasks. We hypothesize that task-specific architectural integration will consistently outperform preprocessing-based enhancement across medical imaging domains, but systematic validation is needed.





// TODO: irgendwie gibt es so ein satz von machinelearning so nach dem Motto "Keep it simple"






// TODO: noch mal die anderen beiden dokumente (gdoc und notes.typ) durchschauen, ob da noch was verwertbares dabei ist

// TODO: further research: ich glaube das trainings material an sich ist nicht perfekt. Vielleicht könnte man bei REFUGE unstimmigkeiten zwischen den verschiedenen leuten die labeln das mit in die Loss funktion mit rein packen.

// TODO: Ich glaube auch, dass unser model probleme hat, wenn das glaucom schon sehr fortgeschritten ist (sehr große OC). Also OC:OD gegen 1:1. vielleicht das auch irgendwie mit in die Loss funktion packen, dass hohe ratio stärker gewichtet wird

// TODO: Auch das wir aufgrund der begrenzten hardware ressourcen das Ding nicht mit mehr auflösung trainieren konnte. Welche haben wir überhaupt jetzt benutzt? 256 oder 512?





// TODO: großes problem würde ich wirklich sagen, die trainingsdaten. Ich bin selbst kein augenarzt, aber das ist schon teilweise wirklich sehr sehr schwer zu erkennen.



// TODO: ganz am Ende schauen, ob wir noch irgendwo fest zahlen haben

// TODO: Oben wo clahe erklärt wird kann man bestimmt mal schön ein vergleichsbild rein machen.
