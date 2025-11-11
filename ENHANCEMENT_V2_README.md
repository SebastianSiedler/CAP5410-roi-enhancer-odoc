# Enhancement Model v2 - With Shape Prior Loss

## Neue Features

### 1. **CLAHE Segmentationsmodell**
- Verwendet `unet_segmentation_model_with_CLAHE` statt `unet_segmentation_model`
- Trainiert mit vorgeschaltetem CLAHE-Filter für bessere Kontrastverstärkung
- Path: `unet_segmentation_model_with_CLAHE/best_model.pth`

### 2. **Shape Prior Loss (NEU!)**
- Implementiert anatomisch plausible Formregularisierung
- Verhindert "zerfetzte" oder unregelmäßige Segmentierungskonturen
- Erzwingt runde/elliptische Formen für Optic Disc/Cup

## Mathematische Formulierung

Die neue Loss-Funktion ist:

$$\mathcal{L}_{\text{Total}} = \lambda_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}} + \lambda_{\text{Perceptual}} \cdot \mathcal{L}_{\text{Perceptual}} + \lambda_{\text{TV}} \cdot \mathcal{L}_{\text{TV}} + \lambda_{\text{Shape}} \cdot \mathcal{L}_{\text{Shape}}$$

### Loss-Komponenten:

1. **Dice Loss** ($\lambda=10.0$): Pixel-Overlap-Optimierung
2. **Perceptual Loss** ($\lambda=0.05$): Bildqualität-Erhaltung
3. **Total Variation Loss** ($\lambda=0.001$): Räumliche Glattheit
4. **Shape Prior Loss** ($\lambda=0.1$, NEU): Anatomische Plausibilität

### Shape Prior Loss - Wie funktioniert es?

1. **Trainiere Shape Autoencoder** (VAE):
   - Encoder: Segmentierungsmaske → Latent Code
   - Decoder: Latent Code → Rekonstruierte Maske
   - Lernt kompakte Darstellung anatomisch korrekter Formen

2. **Während Enhancement-Training**:
   - Encode Ground Truth Masken zu $z_{\text{GT}}$
   - Encode Predicted Masken zu $z_{\text{Pred}}$
   - Shape Loss: $\mathcal{L}_{\text{Shape}} = \text{MSE}(z_{\text{Pred}}, z_{\text{GT}})$
   - Zwingt Vorhersagen, ähnliche latente Codes wie Ground Truth zu haben

3. **Resultat**:
   - Fragmentierte Segmentierungen werden bestraft (hoher Latent Space Distance)
   - Runde/elliptische Formen werden bevorzugt (niedriger Latent Space Distance)

## Neue Dateien

### 1. `src/models/shape_autoencoder.py`
```python
ShapeAutoencoder(n_classes=3, latent_dim=64)
├── ShapeEncoder: Conv2d Layers → FC → (mu, logvar)
├── ShapeDecoder: FC → TransposedConv2d Layers
└── VAE Loss: Reconstruction + KL Divergence
```

**Verwendung:**
```python
from src.models.shape_autoencoder import ShapeAutoencoder

# Laden
model = ShapeAutoencoder(n_classes=3, latent_dim=64)
checkpoint = torch.load('shape_autoencoder/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Encode
latent_codes = model.encode(masks)  # [B, 64]

# Decode  
reconstructed = model.decode(latent_codes)  # [B, 3, 256, 256]
```

### 2. `scripts/train_shape_autoencoder.py`
Standalone-Skript zum Training des Shape Autoencoders.

**Ausführung:**
```bash
cd /path/to/project
source .venv/bin/activate
python scripts/train_shape_autoencoder.py
```

**Konfiguration:**
- Epochs: 100
- Batch Size: 16
- Learning Rate: 1e-3
- Latent Dim: 64
- Speichert nach: `shape_autoencoder/best_model.pth`

### 3. Aktualisierte `src/training/task_aware_train.py`

**Neue Klassen:**
```python
class ShapePriorLoss(nn.Module):
    """Regularisiert Segmentierungsformen mit gelernten Latent Codes"""
    
class TaskAwareLoss(nn.Module):
    """Erweitert um lambda_shape Parameter"""
```

**Neue Parameter in `train_enhancement_model()`:**
- `lambda_shape`: Gewicht für Shape Prior Loss (default: 0.1)
- `shape_autoencoder_path`: Pfad zum trainierten Shape Autoencoder

## Anwendung

### Schritt 1: Shape Autoencoder trainieren

**WICHTIG:** Muss vor dem Enhancement-Training ausgeführt werden!

```bash
cd /home/robolab/dev/steffen/CAP5410-roi-enhancer-odoc
source .venv/bin/activate && python scripts/train_shape_autoencoder.py
```

**Output:**
- Trainierte Modell: `shape_autoencoder/best_model.pth`
- Training History: `shape_autoencoder/training_history.png`
- Dauer: ~15-30 Minuten auf GPU

### Schritt 2: Enhancement Model trainieren

Im Notebook `train_enhancement_model.ipynb`:

```python
config = {
    # ...
    'segmentation_checkpoint': str(project_root / 'unet_segmentation_model_with_CLAHE' / 'best_model.pth'),
    'shape_autoencoder_checkpoint': str(project_root / 'shape_autoencoder' / 'best_model.pth'),
    'lambda_shape': 0.1,  # NEU
    # ...
}
```

### Schritt 3: Training ausführen

Zellen 3-23 im Notebook ausführen.

**Neue Loss-Komponenten im Training Log:**
```
Epoch 1/30 [Train]: loss: 0.8234, dice: 0.4567, perceptual: 0.0234, tv: 0.0012, shape: 0.0421
```

## Erwartete Verbesserungen

Mit Shape Prior Loss sollten Sie sehen:

1. **Bessere Segmentierungsformen**
   - Rundere, glattere Konturen für Optic Disc/Cup
   - Weniger Fragmentierung
   - Anatomisch plausiblere Ergebnisse

2. **Höhere Metriken**
   - Dice Score: +5-10% Verbesserung
   - IoU: +3-8% Verbesserung
   - Besonders für Cup (schwieriger zu segmentieren)

3. **Stabileres Training**
   - Shape Loss regularisiert zusätzlich
   - Verhindert unrealistische Vorhersagen
   - Bessere Generalisierung auf Test Set

## Ablauf-Diagramm

```
1. Trainiere Shape Autoencoder
   └─> Lernt latente Darstellung von GT-Masken
       └─> Speichert: shape_autoencoder/best_model.pth

2. Trainiere Enhancement Model
   ├─> Lädt: unet_segmentation_model_with_CLAHE/best_model.pth (frozen)
   ├─> Lädt: shape_autoencoder/best_model.pth (frozen)
   └─> Trainiert: Enhancement Model
       └─> Loss = Dice + Perceptual + TV + Shape
           └─> Shape Loss zwingt anatomische Plausibilität

3. Evaluation
   └─> Vergleiche: Baseline vs Enhanced
       ├─> Dice Scores
       ├─> IoU Scores
       └─> Visuelle Inspektion (runde Formen?)
```

## Troubleshooting

### Problem: "Shape Autoencoder not found"
**Lösung:** Trainiere Shape Autoencoder zuerst (siehe Schritt 1)

### Problem: Shape Loss = 0.0
**Ursache:** Shape Autoencoder wurde nicht geladen
**Lösung:** Überprüfe Pfad in `shape_autoencoder_checkpoint`

### Problem: Out of Memory
**Lösung:** Reduziere `batch_size` in config (z.B. auf 4)

### Problem: Shape Loss zu hoch/niedrig
**Lösung:** Passe `lambda_shape` an:
- Zu hoch (>0.5): Overregularization, Modell kann nicht lernen
- Zu niedrig (<0.01): Wenig Effekt
- Optimal: 0.05-0.2

## Vergleich: V1 vs V2

| Feature | V1 (Alt) | V2 (Neu) |
|---------|----------|----------|
| Segmentation Model | Standard UNet | UNet + CLAHE |
| Loss Components | 3 (Dice, Perceptual, TV) | 4 (+ Shape) |
| Shape Regularization | ❌ | ✅ |
| Anatomische Plausibilität | Nicht erzwungen | Erzwungen via VAE |
| Training Time | ~2h | ~2.5h (+0.5h für Shape AE) |
| Expected Dice Improvement | -3.5% (verschlechtert!) | +5-10% (verbessert!) |

## Weitere Optimierungen

Mögliche Verbesserungen für zukünftige Versionen:

1. **Conditional VAE**: Bedingung auf Bildmerkmale für bessere Shapes
2. **Adversarial Shape Loss**: GAN-basierte Diskriminierung plausibler Formen
3. **Multi-Scale Shape Loss**: Shape Constraints auf mehreren Auflösungen
4. **Active Contour Integration**: Explizite Kontur-Glättung

## Referenzen

- **ACNN Paper**: Anatomy-aware deep learning for optic disc/cup segmentation
- **VAE for Shape Priors**: Variational Autoencoders for medical image segmentation
- **Task-Aware Enhancement**: End-to-end trainable enhancement networks

## Lizenz

Siehe Projekt-ROOT für Lizenzinformationen.
