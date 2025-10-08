# 📓 How to Use Experiment Notebooks

## Quick Start

1. **Navigate to an experiment folder:**
   ```bash
   cd experiments/03_resnet34_unet_clahe
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook experiment.ipynb
   ```

3. **Run cells sequentially** (Shift+Enter or click ▶️)

## 📋 Notebook Structure

All experiment notebooks follow this standardized structure:

### 1️⃣ Environment Setup
- Import libraries
- Check GPU availability
- Set up paths

### 2️⃣ Configuration
- Define hyperparameters
- Set data paths
- Save config to JSON

### 3️⃣ Data Loading
- Create datasets
- Create data loaders
- Visualize samples (optional)

### 4️⃣ Model Definition
- Create model
- Count parameters
- Define loss & optimizer

### 5️⃣ Training
- Training loop
- Validation
- Save best model
- Plot training curves

### 6️⃣ Testing
- Load best model
- Test on test set
- Calculate metrics

### 7️⃣ Visualizations
- Generate sample predictions
- Save visualizations

### 8️⃣ Results Summary
- Key findings
- Comparison with previous experiments
- Next steps

## 🎯 Tips for Running Notebooks

### Running Full Experiment

To run a complete experiment from scratch:

1. Open `experiment.ipynb`
2. **Kernel → Restart & Run All**
3. Wait for training to complete (~5-10 hours depending on model)
4. Check `results/` folder for outputs

### Running Only Testing

If you already have a trained model:

1. Open `experiment.ipynb`
2. Run cells 1-4 (setup + model definition)
3. **Skip training section (section 5)**
4. Run sections 6-8 (testing + visualizations)

### Debugging

If something goes wrong:

1. **Kernel → Restart Kernel**
2. Check GPU memory: Run `nvidia-smi` in terminal
3. Reduce batch size if OOM (Out of Memory)
4. Check paths in configuration cell

## 📊 Output Files

After running a notebook, you'll find:

```
results/
├── best_model.pth              # Model checkpoint
├── config.json                 # Experiment config
├── training_history.json       # Loss/dice per epoch
├── training_curves.png         # Training plots
├── test_results.json          # Test metrics
└── visualizations/
    └── predictions.png         # Sample predictions
```

## 🔄 Creating New Experiments

1. **Copy template:**
   ```bash
   cp experiments/00_template/template.ipynb experiments/04_my_experiment/experiment.ipynb
   ```

2. **Customize sections:**
   - Update experiment overview (title, hypothesis, config)
   - Import your model
   - Modify training loop if needed
   - Update results summary

3. **Run and document:**
   - Run full experiment
   - Document findings in markdown cells
   - Create README.md for the experiment

4. **Update main overview:**
   - Add experiment to `experiments/README.md`
   - Update comparison table

## 💡 Best Practices

### Documentation
- ✅ Add markdown cells explaining key decisions
- ✅ Comment code for clarity
- ✅ Document unexpected results
- ✅ Compare with previous experiments

### Code Quality
- ✅ Keep cells focused (one task per cell)
- ✅ Use descriptive variable names
- ✅ Print progress and intermediate results
- ✅ Save checkpoints regularly

### Reproducibility
- ✅ Set random seeds (`np.random.seed(42)`)
- ✅ Save full configuration to JSON
- ✅ Document Python/library versions
- ✅ Keep data preprocessing consistent

## 🐛 Common Issues

### GPU Out of Memory
**Solution:** Reduce batch size
```python
CONFIG['batch_size'] = 4  # Instead of 8
```

### CUDA Error
**Solution:** Restart kernel and check GPU
```bash
nvidia-smi  # Check if GPU is available
```

### Import Error
**Solution:** Check virtual environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Slow Training
**Solution:** 
- Reduce `num_workers` if CPU bottleneck
- Use smaller model for debugging
- Enable `pin_memory=True` in DataLoader

## 📚 Additional Resources

- **Main Documentation:** `README.md` (project root)
- **Architecture Guide:** `ARCHITECTURE.md`
- **CLAHE Preprocessing:** `ENHANCEMENT_README.md`
- **Quick Start:** `QUICK_START.md`

## 🤝 Collaboration

When sharing notebooks with team:

1. **Clear outputs before committing:**
   - Kernel → Restart & Clear Output
   - Saves space in git

2. **Document changes:**
   - Add markdown cell explaining modifications
   - Update experiment README

3. **Version checkpoints:**
   - Save important checkpoints with descriptive names
   - E.g., `best_model_epoch47_dice0.8494.pth`

---

**Happy Experimenting! 🚀**

For questions, check `experiments/README.md` or the main project documentation.
