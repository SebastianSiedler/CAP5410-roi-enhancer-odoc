# Task-Aware Enhancement Model für Glaukom-Erkennung

## Überblick

Dieses Projekt implementiert ein **Task-Aware Enhancement-Modell** (M_E), das speziell darauf trainiert wird, die Segmentierungsleistung eines nachgeschalteten Segmentierungsmodells (M_S) für Optic Disc (OD) und Optic Cup (OC) zu optimieren.

### Architektur: Hybrid U-Net mit Residual Blocks

Das Enhancement-Modell kombiniert:
- **U-Net Backbone**: Goldstandard für Image-to-Image-Rekonstruktion
- **Residual Blocks**: Bottleneck-Design aus CycleGAN-Generatoren für Effizienz
- **Leaky ReLU**: Erhält Informationen bei niedrigen Pixelwerten (wichtig für Low-Contrast Fundusbilder)
- **Instance Normalization**: Bessere Generalisierung

## Projektstruktur

```
src/
├── models/
│   ├── enhancement_unet.py          # Enhancement-Modell (M_E)
│   └── unet.py                       # Segmentierungsmodell (M_S)
├── training/
│   ├── task_aware_train.py          # Loss-Funktionen und Trainer
│   └── train_enhancement.py         # Haupttrainingsskript
├── data_loader/
│   ├── dataset.py                   # Dataset-Loader
│   └── transforms.py                # Augmentierungen
└── test_enhancement.py              # Test-Suite

configs/
└── enhancement_config.yaml          # Trainingskonfiguration
```

## Modellkomponenten

### 1. Enhancement U-Net (`EnhancementUNet`)

**Komponenten:**
- **Initial Conv**: 3 → 64 Kanäle
- **Encoder**: 4 Downsampling-Stufen (64 → 128 → 256 → 512 → 1024)
- **Bottleneck**: 6 Residual Blocks (reduzierte Komplexität)
- **Decoder**: 4 Upsampling-Stufen mit Skip Connections
- **Output**: Tanh-Aktivierung → [0, 1] Skalierung

**Parameter (base_channels=64, n_residual_blocks=6):** ~47M

### 2. Task-Aware Loss Funktion

Die kombinierte Loss-Funktion optimiert sowohl Segmentierungsleistung als auch Bildqualität:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{dice}} \cdot \mathcal{L}_{\text{dice}} + \lambda_{\text{perceptual}} \cdot \mathcal{L}_{\text{perceptual}}$$

Wo:
- $\mathcal{L}_{\text{dice}}$: Segmentierungs-Dice-Loss (Task-Aware-Komponente)
- $\mathcal{L}_{\text{perceptual}}$: L1-Distanz zwischen Enhanced und Original (Qualitätskomponente)
- $\lambda_{\text{dice}} = 1.0$: Gewicht für Task-Aware-Loss
- $\lambda_{\text{perceptual}} = 0.1$: Gewicht für Qualitätssicherung

### 3. End-to-End Pipeline (`EnhancementSegmentationPipeline`)

```
Input Image → Enhancement Model (M_E) → Enhanced Image → Segmentation Model (M_S) → Segmentation
     ↓                                        ↓                                      ↓
     └──────────── Perceptual Loss ─────────┘                                      │
                                                                                     │
                                                            Dice Loss (Task-Aware) ─┘
```

**Trainingsstrategie:**
- Segmentierungsmodell (M_S) wird **eingefroren** (freeze_segmentation=True)
- Nur Enhancement-Modell (M_E) wird trainiert
- M_E wird so optimiert, dass die Dice-Score von M_S maximiert wird

## Installation und Setup

### 1. Virtuelle Umgebung aktivieren

```bash
source .venv/bin/activate
```

### 2. Abhängigkeiten prüfen

```bash
pip list | grep -E "torch|torchvision|numpy|pillow|pyyaml|tensorboard|tqdm|matplotlib"
```

Falls fehlend:

```bash
source .venv/bin/activate && pip install torch torchvision torchaudio numpy pillow pyyaml tensorboard tqdm matplotlib
```

## Verwendung

### 1. Tests ausführen

Vor dem Training sollten Sie die Implementierung testen:

```bash
source .venv/bin/activate && python src/test_enhancement.py
```

**Erwartete Ausgabe:**
```
✓ Enhancement Model Test
✓ Pipeline Test
✓ Loss Function Test
✓ Training Step Test
✓ ALL TESTS PASSED
```

### 2. Training starten

#### Option A: Mit Konfigurationsdatei

```bash
source .venv/bin/activate && python src/training/train_enhancement.py --config configs/enhancement_config.yaml
```

#### Option B: Mit Kommandozeilenargumenten

```bash
source .venv/bin/activate && python src/training/train_enhancement.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0002 \
    --enhancement_channels 64 \
    --n_residual_blocks 6 \
    --lambda_dice 1.0 \
    --lambda_perceptual 0.1 \
    --exp_name task_aware_v1
```

### 3. Training mit vortrainiertem Segmentierungsmodell

Falls Sie bereits ein trainiertes Segmentierungsmodell haben:

```bash
source .venv/bin/activate && python src/training/train_enhancement.py \
    --config configs/enhancement_config.yaml \
    --segmentation_checkpoint experiments/segmentation/best_model.pth \
    --freeze_segmentation
```

### 4. Monitoring mit TensorBoard

```bash
tensorboard --logdir experiments/enhancement/task_aware_v1/logs
```

Dann öffnen Sie: http://localhost:6006

## Trainingskonfiguration anpassen

Bearbeiten Sie `configs/enhancement_config.yaml`:

```yaml
# Modellgröße reduzieren (für schnelleres Training/weniger VRAM)
enhancement_channels: 32
n_residual_blocks: 4

# Loss-Gewichte anpassen
lambda_dice: 1.0          # Höher → Fokus auf Segmentierung
lambda_perceptual: 0.05   # Niedriger → Weniger Constraint auf Bildähnlichkeit

# Training beschleunigen
batch_size: 16
num_workers: 8
```

## Evaluierung

Nach dem Training:

```python
import torch
from src.models.enhancement_unet import EnhancementUNet
from src.models.unet import UNet

# Modelle laden
enhancement_model = EnhancementUNet(n_channels=3, n_output_channels=3, base_channels=64)
checkpoint = torch.load('experiments/enhancement/task_aware_v1/checkpoints/best_model.pth')
enhancement_model.load_state_dict(checkpoint['enhancement_state_dict'])
enhancement_model.eval()

# Bild enhancen
with torch.no_grad():
    enhanced = enhancement_model(input_image)
```

## Vergleich mit Baseline (CLAHE)

Das Projekt ermöglicht den direkten Vergleich mit CLAHE-Preprocessing:

**Pfad A (Baseline):** Input → CLAHE → Segmentation → Dice Score  
**Pfad B (Task-Aware):** Input → Enhancement Model → Segmentation → Dice Score

Die Task-Aware-Optimierung sollte bessere Dice-Scores für OD/OC-Segmentierung erzielen.

## Wichtige Hyperparameter

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `enhancement_channels` | 64 | Basis-Kanalanzahl (höher = größeres Modell) |
| `n_residual_blocks` | 6 | Anzahl Residual Blocks im Bottleneck |
| `lambda_dice` | 1.0 | Gewicht für Task-Aware-Loss |
| `lambda_perceptual` | 0.1 | Gewicht für Bildqualitäts-Loss |
| `lr` | 0.0002 | Learning Rate (typisch für GANs) |
| `freeze_segmentation` | True | Segmentierungsmodell einfrieren |

## Troubleshooting

### CUDA Out of Memory

- Batch Size reduzieren: `--batch_size 4`
- Kleineres Modell: `--enhancement_channels 32 --n_residual_blocks 4`
- Kleinere Bildgröße in `transforms.py` anpassen

### Training instabil

- Learning Rate reduzieren: `--lr 0.0001`
- Perceptual Loss erhöhen: `--lambda_perceptual 0.2`
- Gradient Clipping hinzufügen (in `task_aware_train.py`)

### Schlechte Segmentierungsleistung

- Segmentierungsmodell zuerst trainieren
- Lambda_dice erhöhen: `--lambda_dice 2.0`
- Mehr Residual Blocks: `--n_residual_blocks 9`

## Ergebnisse visualisieren

Während des Trainings werden Visualisierungen gespeichert:

```
experiments/enhancement/task_aware_v1/visualizations/
├── epoch_001.png
├── epoch_002.png
└── ...
```

Jedes Bild zeigt:
1. Original Image
2. Enhanced Image
3. Ground Truth Segmentation
4. Predicted Segmentation

## Weitere Optimierungen

1. **Multi-Scale Loss**: Dice Loss auf mehreren Auflösungen
2. **Adversarial Loss**: Diskriminator hinzufügen für realistischere Enhancements
3. **Attention Mechanisms**: Self-Attention im Bottleneck
4. **Progressive Training**: Erst auf niedriger Auflösung, dann hochskalieren

## Zitation

Basierend auf:
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- CycleGAN: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
- Residual Networks: He et al., "Deep Residual Learning for Image Recognition"

## Kontakt und Support

Bei Fragen oder Problemen, siehe die Dokumentation oder erstellen Sie ein Issue.
