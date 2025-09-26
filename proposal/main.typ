
= ROI Enhancer for Optic Disc and Cup Segmentation in Fundus Images

== Introduction
@src_kinder2025optic



== Technische Umsetzung


=== Architektur
Ihr Projekt bedeutet nicht, dass Sie zwei komplett unabhängige Netzwerke erstellen. Stattdessen bauen Sie eine End-to-End-Architektur auf, die aus zwei Modulen besteht:

- Modul 1 (Enhancer $E$): Ein kleines U-Net oder ein Residual Network (Ihr innovativer Teil).

- Modul 2 (Segmentierer $S$): Ein Standard-U-Net (das Sie von Grund auf bauen oder anpassen).

#figure(image("architecture.png", width: 80%))


=== Q&A
- warum zwei module und nicht nur ein Segmentierer?
  - damit jedes model nur genau eine Aufgabe hat
- warum nicht nur CLAHE?


#bibliography("works.bib", style: "ieee")
