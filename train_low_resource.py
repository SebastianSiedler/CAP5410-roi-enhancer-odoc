"""
Low-resource training script for EE-TransUNet
This script uses reduced settings for systems with limited hardware
"""

import subprocess
import sys

# Training configuration for low-resource systems
config = {
    'model_name': 'ViT-B_16',  # Smaller model (no ResNet50 backbone)
    'img_size': 224,            # Reduced from 512 (saves ~75% memory)
    'batch_size': 2,            # Very small batch size for low memory
    'epochs': 30,               # Reduced epochs for faster initial testing
    'lr': 0.005,               # Slightly lower learning rate
    'num_workers': 0,          # No multiprocessing overhead
}

# Build command
cmd = [
    sys.executable,
    'src/main.py',
    '--model_name', config['model_name'],
    '--img_size', str(config['img_size']),
    '--batch_size', str(config['batch_size']),
    '--epochs', str(config['epochs']),
]

print("="*60)
print("Starting Low-Resource EE-TransUNet Training")
print("="*60)
print(f"Model: {config['model_name']}")
print(f"Image Size: {config['img_size']}x{config['img_size']}")
print(f"Batch Size: {config['batch_size']}")
print(f"Epochs: {config['epochs']}")
print("="*60)
print("\n")

# Run training
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"\nTraining failed with error code: {e.returncode}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    sys.exit(0)
