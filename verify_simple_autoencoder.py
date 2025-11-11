#!/usr/bin/env python3
"""
Quick verification script to test loading the trained Simple Autoencoder.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.simple_shape_autoencoder import SimpleShapeAutoencoder

def verify_model():
    """Verify the trained Simple Autoencoder can be loaded and used."""
    
    print("="*80)
    print("Simple Autoencoder Verification")
    print("="*80)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load model
    print("\n1. Loading trained Simple Autoencoder...")
    checkpoint_path = project_root / 'shape_autoencoder' / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return False
    
    try:
        model = SimpleShapeAutoencoder(latent_dim=64)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully!")
        print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        train_loss = checkpoint.get('train_loss', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        if train_loss != 'unknown':
            print(f"   Train loss: {train_loss:.4f}")
        else:
            print(f"   Train loss: {train_loss}")
        if val_loss != 'unknown':
            print(f"   Val loss: {val_loss:.4f}")
        else:
            print(f"   Val loss: {val_loss}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return False
    
    # Test inference
    print("\n2. Testing inference...")
    try:
        # Create dummy input (batch_size=1, channels=3, height=256, width=256)
        dummy_input = torch.randn(1, 3, 256, 256, device=device)
        
        with torch.no_grad():
            # Test encoding
            latent = model.encode(dummy_input)
            print(f"✅ Encoding successful!")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Latent shape: {latent.shape}")
            print(f"   Latent dimension: {latent.shape[1]}")
            
            # Test full forward pass
            output = model(dummy_input)
            # Handle tuple output (reconstruction, loss)
            if isinstance(output, tuple):
                reconstruction = output[0]
            else:
                reconstruction = output
            print(f"✅ Reconstruction successful!")
            print(f"   Reconstruction shape: {reconstruction.shape}")
            
            # Check for NaN values
            if torch.isnan(latent).any():
                print("❌ ERROR: NaN values in latent representation!")
                return False
            if torch.isnan(reconstruction).any():
                print("❌ ERROR: NaN values in reconstruction!")
                return False
            
            print("✅ No NaN values detected!")
            
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        return False
    
    # Model info
    print("\n3. Model Information:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   Model size: ~113MB")
    
    print("\n" + "="*80)
    print("✅ ALL CHECKS PASSED - Model is ready for integration!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = verify_model()
    sys.exit(0 if success else 1)
