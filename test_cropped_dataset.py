"""
Quick test to verify the dataset and model work with cropped masks
"""
import sys
sys.path.insert(0, 'src')

import torch
from data_loader.dataset import RetinaDataset
from models.unet import UNet
from torch.utils.data import DataLoader

print("=" * 60)
print("Testing Cropped Masks Dataset Integration")
print("=" * 60)

# 1. Test dataset loading
print("\n1. Testing dataset loading...")
dataset = RetinaDataset(
    root_dir='datasets/REFUGE',
    csv_file='datasets/REFUGE/REFUGETrain.csv',
    target_size=(512, 512),
    use_cropped=True,
    cropped_masks_dir='datasets/REFUGE_cropped_masks'
)
print(f"   ✓ Dataset created: {len(dataset)} samples")

# 2. Test dataloader
print("\n2. Testing dataloader...")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
batch = next(iter(dataloader))
images, masks = batch
print(f"   ✓ Batch shape: images={images.shape}, masks={masks.shape}")
print(f"   ✓ Disc coverage: {masks[:, 0].mean() * 100:.1f}%")
print(f"   ✓ Cup coverage: {masks[:, 1].mean() * 100:.1f}%")

# 3. Test model forward pass
print("\n3. Testing model forward pass...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

model = UNet(n_channels=3, n_classes=2, base_features=64)
model = model.to(device)
images = images.to(device)
masks = masks.to(device)

with torch.no_grad():
    outputs = model(images)

print(f"   ✓ Model output shape: {outputs.shape}")
print(f"   ✓ Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")

# 4. Test loss computation
print("\n4. Testing loss computation...")
from utils.loss_functions import CombinedSegmentationLoss

criterion = CombinedSegmentationLoss(lambda_dice=0.5, lambda_bce=0.5)
loss = criterion(outputs, masks)
print(f"   ✓ Loss computed: {loss.item():.4f}")

# 5. Test metrics
print("\n5. Testing metrics...")
from utils.metrics import batch_metrics

outputs_sigmoid = torch.sigmoid(outputs)
predictions = (outputs_sigmoid > 0.5).float()

metrics = batch_metrics(predictions, masks)
print(f"   ✓ Disc Dice: {metrics['dice_disc']:.4f}")
print(f"   ✓ Cup Dice: {metrics['dice_cup']:.4f}")
print(f"   ✓ Mean Dice: {metrics['dice_mean']:.4f}")
print(f"   ✓ CDR MAE: {metrics['cdr_mae']:.4f}")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nReady to train with:")
print("  python src/main.py --epochs 50 --batch_size 4 --lr 1e-4")
