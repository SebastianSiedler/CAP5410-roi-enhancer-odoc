# ROI Enhancer for Optic Disc and Cup Segmentation

Final project for **CAP5410 Advanced Computer Vision** investigating task-aware multi-scale feature learning for glaucoma diagnosis.

## Overview

This project systematically compares different approaches for optic disc (OD) and optic cup (OC) segmentation in fundus images, focusing on whether task-aware learned enhancement or direct architectural improvements better support medical image segmentation.

We evaluate five approaches:
- **Baseline UNet**: Standard encoder-decoder architecture
- **CLAHE + UNet**: Traditional preprocessing with histogram equalization
- **Learned Enhancer + UNet**: Task-aware enhancement module trained jointly
- **Atrous Enhancer + UNet**: Multi-scale enhancement using ASPP blocks
- **ASPP-UNet**: Integrated multi-scale features within the segmentation architecture

## Key Findings

Our experiments on 2,636 fundus images from three datasets (G1020, ORIGA, REFUGE) demonstrate that **architectural integration outperforms preprocessing-based methods**. ASPP-UNet achieves the highest mIoU while using fewer parameters than baseline, challenging the common practice of treating enhancement as a separate preprocessing step.

## Authors

**Sebastian Siedler** (ssiedler3117@floridapoly.edu)  
**Steffen Ullmann** (sullmann3121@floridapoly.edu)

Florida Polytechnic University, Fall 2025