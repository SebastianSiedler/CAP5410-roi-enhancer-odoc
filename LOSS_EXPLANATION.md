# Loss Werte Erklärung (Loss Values Explanation)

## Problem: Total Loss > 1

### Warum liegt der Total Loss über 1?

Der **Total Loss** ist eine **gewichtete Summe** mehrerer Loss-Komponenten:

```
Total Loss = λ_dice * L_dice + λ_perceptual * L_perceptual + λ_tv * L_tv + λ_shape * L_shape
```

### Aktuelle Konfiguration:

| Loss-Komponente | Gewicht (λ) | Bereich (ungewichtet) | Beitrag zum Total Loss |
|-----------------|-------------|----------------------|------------------------|
| Dice Loss       | **10.0**    | 0.0 - 1.0           | 0.0 - 10.0            |
| Perceptual Loss | **0.05**    | 0.0 - 2.0           | 0.0 - 0.1             |
| TV Loss         | **0.001**   | 0.0 - 50.0          | 0.0 - 0.05            |
| Shape Loss      | **0.1**     | 0.0 - 1.0           | 0.0 - 0.1             |

### Beispiel-Berechnung:

Angenommen:
- Dice Loss (ungewichtet) = 0.3
- Perceptual Loss (ungewichtet) = 0.5  
- TV Loss (ungewichtet) = 10.0
- Shape Loss (ungewichtet) = 0.2

**Gewichtete Werte:**
```
Dice:       10.0  × 0.3  = 3.0
Perceptual:  0.05 × 0.5  = 0.025
TV:          0.001× 10.0 = 0.01
Shape:       0.1  × 0.2  = 0.02
---------------------------------
Total Loss:              = 3.055 ✓
```

### Was ist wichtig zu beachten?

✅ **Dice Score (Metric)**: 
- Bereich: 0.0 - 1.0 (höher ist besser)
- **Dies ist die primäre Metrik!**
- Misst die tatsächliche Segmentierungsqualität

⚠️ **Total Loss**:
- Kann > 1 sein (ist normal!)
- Sollte über Epochen **abnehmen**
- Wird für Gradientenberechnung verwendet

### Warum diese Gewichtung?

1. **λ_dice = 10.0** (sehr hoch):
   - Priorisiert Segmentierungsqualität
   - Macht den Dice Loss zur dominanten Komponente
   - Das Modell fokussiert sich auf bessere Segmentierung

2. **λ_perceptual = 0.05** (sehr niedrig):
   - Gibt dem Modell mehr Freiheit für Bildverbesserungen
   - Verhindert, dass das Bild zu sehr am Original "klebt"
   - Erlaubt starke Kontrast-Anpassungen

3. **λ_tv = 0.001** (niedrig):
   - Schwache Regularisierung für räumliche Glattheit
   - Verhindert extreme Artefakte
   - Stört die Hauptoptimierung nicht

4. **λ_shape = 0.1** (mittel):
   - Regularisiert die Segmentierungsform
   - Erzwingt anatomisch plausible Formen (rund/elliptisch)
   - Verhindert fragmentierte oder unregelmäßige Konturen

## Was sollte man überwachen?

### Training Logs:

```
[Train] total_loss: 3.0550, dice_loss: 0.3000, dice_metric: 0.7000
```

Interpretation:
- `total_loss = 3.055`: **Normal** (gewichtete Summe)
- `dice_loss = 0.300`: Ungewichteter Dice Loss (sollte sinken)
- `dice_metric = 0.700`: **WICHTIGSTE METRIK** (sollte steigen!)

### Erwartetes Verhalten über Epochen:

| Epoche | Total Loss | Dice Loss (ungew.) | Dice Score (Metrik) |
|--------|------------|-------------------|---------------------|
| 1      | 8.5        | 0.85              | 0.15                |
| 5      | 4.2        | 0.42              | 0.58                |
| 10     | 3.1        | 0.31              | 0.69                |
| 15     | 2.8        | 0.28              | 0.72                |
| 20     | 2.6        | 0.26              | 0.74                |

**Gesunder Trainings-Verlauf:**
- ✅ Total Loss fällt kontinuierlich
- ✅ Dice Score (Metrik) steigt kontinuierlich
- ✅ Verbesserung verlangsamt sich mit der Zeit (normal!)

**Problematischer Verlauf:**
- ❌ Total Loss stagniert oder steigt
- ❌ Dice Score stagniert oder fällt
- ❌ Große Schwankungen zwischen Epochen

## TensorBoard Metriken

Im TensorBoard werden folgende Metriken geloggt:

### Training:
- `train/loss_total`: Gesamtverlust (gewichtet)
- `train/loss_dice_unweighted`: Dice Loss vor Gewichtung
- `train/loss_perceptual_unweighted`: Perceptual Loss vor Gewichtung
- `train/loss_tv_unweighted`: TV Loss vor Gewichtung
- `train/loss_shape_unweighted`: Shape Loss vor Gewichtung
- `train/dice_mean`: **Dice Score Metrik** (WICHTIG!)

### Validation:
- `val/loss_total`: Gesamtverlust (gewichtet)
- `val/dice_mean`: **Dice Score Metrik** (WICHTIG!)

### Empfohlene TensorBoard-Ansicht:

Fügen Sie diese Diagramme hinzu:
1. **Dice Score**: `train/dice_mean` + `val/dice_mean` (primär!)
2. **Total Loss**: `train/loss_total` + `val/loss_total`
3. **Ungewichtete Losses**: Alle `_unweighted` Metriken zum Debuggen

## Zusammenfassung

🎯 **Hauptpunkt**: Der Total Loss > 1 ist **vollkommen normal** und **gewollt**!

📊 **Fokussiere dich auf**: Dice Score (Metrik) - das ist die eigentliche Segmentierungsqualität!

📉 **Erwartung**: Total Loss sollte fallen, Dice Score sollte steigen

⚙️ **Die Gewichtung** sorgt dafür, dass das Modell Segmentierungsqualität priorisiert

✨ **Shape Prior Loss** hilft dabei, anatomisch korrekte, runde Formen zu erzeugen
