# CAP5410-roi-enhancer-odoc

ROI-Enhancement für Optic Disc/Cup

### 1. ZIEL (GOAL)

Entwicklung einer **Task-Aware ROI-Enhancement Pipeline** zur Verbesserung der Segmentierungsgenauigkeit des **Optic Disc (OD)** und **Optic Cup (OC)** auf qualitativ minderwertigen (simulierten) Fundusbildern. Der primäre Output ist eine verbesserte **Cup-to-Disc Ratio (CDR)** Genauigkeit.

### 2. TECHNOLOGISCHER STACK (TECH_STACK)

| Parameter        | Wert                                             |
| :--------------- | :----------------------------------------------- |
| **Sprache**      | Python 3.x                                       |
| **Framework**    | PyTorch (bevorzugt)                              |
| **Bibliotheken** | NumPy, Pandas, OpenCV, Scikit-image, Torchvision |
| **Zielhardware** | GPU-beschleunigt                                 |

### 3. PIPELINE-ÜBERSICHT (PIPELINE_OVERVIEW)

Das Projekt besteht aus einer **zwei- oder dreistufigen Kette** (**Two/Three-Stage Pipeline**), abhängig davon, ob die OD-Lokalisierung notwendig ist:

1.  **[Optional] Lokalisierung (Localization):** Finde die Bounding Box der OD.
    - diese ist bereits im dataset enthalten (REFUGE Challenge) XXXX_cropped.jpeg
2.  **Enhancement:** Verbessere die Bildqualität des ausgeschnittenen (Cropped) ROI.
3.  **Segmentierung (Segmentation):** Verwende das verbesserte ROI für die pixelgenaue OD/OC-Segmentierung.

### 4. KOMPONENTEN UND ANFORDERUNGEN (COMPONENTS_AND_REQUIREMENTS)

#### A. Datenverarbeitung (DATA_PROCESSING)

| Modul                 | Klasse/Funktion           | Anforderung                                                                                                                                 |
| :-------------------- | :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------ |
| **Datenloader**       | `RetinaDataset(Dataset)`  | Lädt Bilder und Segmentierungsmasken. Gibt ein Tupel (Input_Image, GT_Mask) zurück.                                                         |
| **Defekt-Simulation** | `simulate_defects(image)` | Funktion zur Anwendung von **2-3 wählbaren Defekten** (z.B. Gaußsches Rauschen, Gaußsche Unschärfe, Kontrastreduktion) auf das Input_Image. |

#### B. Modell-Architekturen (MODEL_ARCHITECTURES)

| Modul         | Klasse                   | Anforderung                                                                                                                              |
| :------------ | :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Segmenter** | `UNet(nn.Module)`        | Standard U-Net-Implementierung (Encoder/Decoder mit Skip Connections). Input: Verbessertes ROI. Output: Segmentierungsmaske (OD und OC). |
| **Enhancer**  | `EnhancerNet(nn.Module)` | Leichtes CNN/Autoencoder-basiertes Netzwerk. Input: Defektes ROI. Output: Verbessertes ROI.                                              |

#### C. Training und Verlustfunktionen (LOSS_AND_TRAINING)

| Modul                 | Funktion                            | Anforderung                                                                                                                                                                       |
| :-------------------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task-Aware Loss**   | `CombinedLoss()`                    | Implementierung eines kombinierten Verlustes: $\mathcal{L}_{\text{Total}} = \lambda \cdot \mathcal{L}_{\text{Enhancement}} + (1-\lambda) \cdot \mathcal{L}_{\text{Segmentation}}$ |
| **Enhancement Loss**  | $\mathcal{L}_{\text{Enhancement}}$  | MSE- oder L1-Verlust, um die visuelle Verbesserung zu steuern.                                                                                                                    |
| **Segmentation Loss** | $\mathcal{L}_{\text{Segmentation}}$ | Dice-Loss oder Cross-Entropy-Loss, um die Segmentierungsgenauigkeit zu steuern (dies ist der 'Task-Aware' Teil).                                                                  |

#### D. Metriken (METRICS)

| Modul              | Funktion             | Anforderung                                                             |
| :----------------- | :------------------- | :---------------------------------------------------------------------- |
| **Dice-Score**     | `dice_coefficient()` | Hauptmetrik zur Bewertung der Segmentierungsgenauigkeit.                |
| **CDR-Berechnung** | `calculate_cdr()`    | Funktion zur Berechnung der Cup-to-Disc Ratio basierend auf den Masken. |

### 5. AUSFÜHRUNGSSEQUENZ (EXECUTION_SEQUENCE)

1.  **SETUP & BASELINE:** Implementiere `RetinaDataset` und die U-Net-Klasse. Trainiere und evaluiere die Baseline auf den Originalbildern (keine Defekte, kein Enhancement).
2.  **SIMULATION:** Implementiere `simulate_defects` und erzeuge den simulierten Trainingsdatensatz.
3.  **PIPELINE-TRAINING:** Implementiere `EnhancerNet` und `CombinedLoss`. Trainiere das gesamte System (Enhancer $\rightarrow$ Segmenter) auf den simulierten Bildern.
4.  **EVALUATION:** Führe eine vergleichende Analyse durch: **Baseline vs. Enhancer-Pipeline** auf dem simulierten Test-Set.
