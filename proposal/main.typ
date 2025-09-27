#set page(paper: "us-letter", margin: 1.5cm)

= ROI Enhancer for Optic Disc and Cup Segmentation in Fundus Images
09/27/2025

Proposal by Sebastian Siedler and Steffen Ullmann for the CAP5410 (Advanced Computer Vision) final project at Florida Polytechnic University, Fall 25.

== Introduction
Glaucoma is a leading cause of irreversible vision loss worldwide. Its diagnosis heavily relies on calculating the Cup-to-Disc Ratio (CDR), which requires accurate segmentation of the Optic Disc (OD) and Optic Cup (OC) from retinal fundus images. While deep learning models have achieved high overall performance, accurate delineation of the Optic Cup boundary remains a major challenge due to low image quality, noise, and poor contrast, especially in the diagnostic region.

== State-of-the-Art Review
The current State-of-the-Art typically employs a two-stage pipeline: 1) OD Localization followed by 2) Segmentation on a cropped Region of Interest (ROI). Within this ROI, standard methods often use simple, fixed preprocessing techniques like Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image quality before segmentation. @Gupta2021A @src_kinder2025optic

== Proposed Methodology
This project addresses a critical research gap: current enhancement techniques (including CLAHE) are task-agnostic, meaning they improve general image quality without being optimized to minimize the specific segmentation errors of the downstream neural network.

We propose a novel Task-Aware ROI Enhancement Framework. This approach integrates a trainable enhancement network as a dedicated module immediately before the segmentation network. Both modules are trained jointly under a combined loss function. This forces the Enhancer to selectively optimize the visual features (e.g., sharpening the OC boundary and suppressing irrelevant noise) that are most critical for the final segmentation accuracy, providing a significant performance advantage over static preprocessing methods.


#figure(
  image("architecture.png", width: 70%),
  caption: [
    "Proposed architecture: A trainable Enhancer module is placed before the Segmentation network. Both modules are trained jointly to optimize segmentation performance."
  ],
)


// === Q&A
// - warum zwei module und nicht nur ein Segmentierer?
//   - damit jedes model nur genau eine Aufgabe hat
// - warum nicht nur CLAHE?


#bibliography("works.bib", style: "ieee")
