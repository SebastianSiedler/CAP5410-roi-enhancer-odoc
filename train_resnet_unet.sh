#!/bin/bash

# Training script for ResNet-UNet with CLAHE preprocessing
# This combines the benefits of:
# 1. Pretrained ResNet encoder (transfer learning)
# 2. CLAHE preprocessing (better contrast)
# 3. Two-stage training (freeze then unfreeze encoder)

echo "=========================================="
echo "Training ResNet34-UNet with CLAHE"
echo "=========================================="

# Configuration
EXPERIMENT_NAME="resnet34_unet_clahe"
BACKBONE="resnet34"
EPOCHS=50
BATCH_SIZE=8
LR=1e-4
UNFREEZE_EPOCH=10

# Create experiment directory
SAVE_DIR="experiments/${EXPERIMENT_NAME}"
mkdir -p "${SAVE_DIR}"

echo ""
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Backbone: ${BACKBONE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Unfreeze encoder at epoch: ${UNFREEZE_EPOCH}"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run training
python src/main.py \
    --model resnet_unet \
    --backbone ${BACKBONE} \
    --pretrained \
    --freeze_encoder \
    --unfreeze_epoch ${UNFREEZE_EPOCH} \
    --data_dir datasets/REFUGE \
    --train_csv datasets/REFUGE/REFUGETrain.csv \
    --val_csv datasets/REFUGE/REFUGE1Val.csv \
    --save_dir ${SAVE_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay 1e-4 \
    --target_size 512 \
    --use_clahe \
    --clahe_clip_limit 2.0 \
    --clahe_mode LAB \
    --lambda_dice 1.0 \
    --lambda_bce 1.0 \
    --save_freq 10

echo ""
echo "=========================================="
echo "Training completed!"
echo "Results saved to: ${SAVE_DIR}"
echo "=========================================="

# Print final model info
echo ""
echo "Model checkpoint: ${SAVE_DIR}/best_model.pth"
echo "Training log: ${SAVE_DIR}/training_log.txt"
echo "Config: ${SAVE_DIR}/config.json"

# Show best validation dice
if [ -f "${SAVE_DIR}/training_log.txt" ]; then
    echo ""
    echo "Best validation dice:"
    tail -n 1 "${SAVE_DIR}/training_log.txt"
fi
