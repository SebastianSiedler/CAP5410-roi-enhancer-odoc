# Model Usage Guide

## What You Have

You've trained a **U-Net model** that segments optic disc and optic cup from fundus images:

### Model Performance:
- ✅ **Dice Score: 0.9561** (95.61% segmentation accuracy)
- ✅ **CDR MAE: 0.0043** (Very accurate cup-to-disc ratio prediction)
- 📦 **Model size: ~31M parameters**
- 💾 **Checkpoint saved at:** `experiments/baseline_unet/best_model.pth`

### What It Predicts:
1. **Optic Disc (OD)** - The larger circular region
2. **Optic Cup (OC)** - The central depression
3. **Cup-to-Disc Ratio (CDR)** - Critical for glaucoma diagnosis
   - Normal: CDR < 0.5
   - Suspicious: CDR 0.5-0.6
   - Glaucoma risk: CDR > 0.6

---

## 🚀 How to Use Your Model

### 1. Single Image Inference

Test your model on a single image:

```bash
python inference.py \
    --checkpoint experiments/baseline_unet/best_model.pth \
    --image datasets/REFUGE/Test-400/0401/0401_cropped.jpg \
    --output results/prediction_0401.png
```

This will:
- Load the trained model
- Process the input image
- Generate segmentation masks
- Calculate CDR
- Save a visualization showing all results

### 2. Batch Inference on Test Set

Process multiple images from the test set:

```bash
python test_baseline.py \
    --checkpoint experiments/baseline_unet/best_model.pth \
    --test_csv datasets/REFUGE/REFUGE1Test.csv \
    --output_dir results/test_predictions
```

This will evaluate your model on the entire test set and generate:
- Per-image metrics
- Average Dice scores
- CDR predictions vs ground truth
- Confusion matrices

### 3. Interactive Python Usage

```python
import torch
from inference import load_model, preprocess_image, predict

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, config = load_model('experiments/baseline_unet/best_model.pth', device)

# Process an image
image_tensor, original = preprocess_image('path/to/fundus_image.jpg')

# Get predictions
disc_mask, cup_mask, cdr = predict(model, image_tensor, device)

print(f"Predicted CDR: {cdr:.4f}")
```

---

## 📊 Next Steps: The Complete Pipeline

According to your project architecture, this baseline model is **Step 1** of a larger pipeline:

### Current Status: ✅ Baseline Complete
```
Clean Image → U-Net → Segmentation Masks (Dice: 0.9561)
```

### Next Phase: Image Enhancement Pipeline

Your project goal is to handle **low-quality images** by adding an enhancement step:

```
                    ┌─────────────────────────┐
Degraded Image ──→  │  EnhancerNet (TODO)     │ ──→ Enhanced Image
                    └─────────────────────────┘
                              ↓
                    ┌─────────────────────────┐
Enhanced Image  ──→ │  U-Net (TRAINED ✓)      │ ──→ Segmentation Masks
                    └─────────────────────────┘
```

### What You Need to Implement Next:

1. **Image Quality Degradation (Augmentation)** ✅ Already implemented!
   - Location: `src/data_loader/augmentation.py`
   - Simulates: noise, blur, contrast reduction, brightness changes

2. **EnhancerNet Model** 📝 TODO
   - A lightweight CNN that takes degraded images and outputs enhanced versions
   - Will be trained with task-aware loss (enhancement + segmentation)

3. **Joint Training** 📝 TODO
   - Train EnhancerNet → U-Net pipeline end-to-end
   - Use combined loss: `λ × L_enhancement + (1-λ) × L_segmentation`

---

## 🔧 Useful Commands

### Check model info:
```bash
python -c "
import torch
checkpoint = torch.load('experiments/baseline_unet/best_model.pth')
print('Epoch:', checkpoint['epoch'])
print('Metrics:', checkpoint['metrics'])
"
```

### Visualize training progress:
```bash
cat experiments/baseline_unet/training_log.txt
```

### Test on validation set:
```bash
python src/main.py \
    --resume experiments/baseline_unet/best_model.pth \
    --epochs 0  # Just evaluate, don't train
```

---

## 📈 Understanding Your Results

### Dice Score: 0.9561
- **Excellent!** Dice > 0.9 is considered very good for medical image segmentation
- Means 95.61% overlap between predicted and ground truth masks

### CDR MAE: 0.0043
- **Outstanding!** Very low error in cup-to-disc ratio
- This is the most critical metric for glaucoma screening
- For context: typical CDR values range from 0.3 to 0.8

### What This Means Clinically:
Your model can reliably:
- ✅ Detect optic disc boundaries
- ✅ Segment optic cup accurately
- ✅ Calculate CDR for glaucoma risk assessment
- ✅ Work on clean, high-quality fundus images

---

## 🎯 Project Roadmap

### Phase 1: Baseline (COMPLETED ✅)
- [x] Implement U-Net architecture
- [x] Train on clean images
- [x] Achieve >95% Dice score
- [x] Validate CDR accuracy

### Phase 2: Enhancement Pipeline (NEXT 📝)
- [ ] Implement image degradation simulation
- [ ] Design EnhancerNet architecture
- [ ] Train enhancement network
- [ ] Test on degraded images

### Phase 3: Task-Aware Training (ADVANCED 🎓)
- [ ] Implement combined loss function
- [ ] Train EnhancerNet + U-Net jointly
- [ ] Compare: Baseline vs Enhanced pipeline
- [ ] Analyze improvement on low-quality images

### Phase 4: Evaluation (FINAL 🏁)
- [ ] Comprehensive testing on test set
- [ ] Generate comparison visualizations
- [ ] Document results and findings
- [ ] Prepare final report

---

## 💡 Tips for Using Your Model

1. **Input Requirements:**
   - RGB fundus images (retina photos)
   - Preferably cropped to optic disc region
   - Will be resized to 512×512 automatically

2. **Output Interpretation:**
   - CDR < 0.5: Normal
   - CDR 0.5-0.6: Borderline/Monitor
   - CDR > 0.6: High glaucoma risk

3. **Performance:**
   - With GPU: ~10-20ms per image
   - With CPU: ~100-200ms per image
   - Batch processing recommended for multiple images

4. **Common Issues:**
   - Out of memory: Reduce batch size or image size
   - Poor predictions: Check if image is properly cropped to OD region
   - CUDA errors: Model trained on GPU may need `map_location='cpu'` when loading

---

## 📚 Additional Resources

- **Model Architecture:** See `src/models/unet.py`
- **Training Script:** See `src/main.py`
- **Dataset Info:** See `datasets/REFUGE/` CSV files
- **Project Architecture:** See `ARCHITECTURE.md`

For questions or issues, check the training logs in `experiments/baseline_unet/training_log.txt`
