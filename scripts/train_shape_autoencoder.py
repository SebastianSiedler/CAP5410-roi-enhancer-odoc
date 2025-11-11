"""
Training script for Shape Autoencoder.

This script trains a Variational Autoencoder on ground truth segmentation masks
to learn a latent representation of anatomically plausible optic disc/cup shapes.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.shape_autoencoder import ShapeAutoencoder, vae_loss
from src.data_loader.dataset import GlaucomaDataset
from src.data_loader.transforms import get_validation_transforms


def train_shape_autoencoder(
    root_dir,
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    image_size=256,
    latent_dim=64,
    num_workers=4,
    device=None,
    save_dir='shape_autoencoder',
    filter_incomplete=True
):
    """
    Train Shape Autoencoder on ground truth masks.
    
    Args:
        root_dir: Project root directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        image_size: Image size
        latent_dim: Latent dimension
        num_workers: Data loading workers
        device: Training device
        save_dir: Save directory
        filter_incomplete: Filter incomplete masks
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load datasets (we only need masks)
    print("\nLoading datasets...")
    transform = get_validation_transforms(image_size=image_size)
    
    train_dataset = GlaucomaDataset(
        root_dir, 
        split='train', 
        transform=transform,
        filter_incomplete=filter_incomplete
    )
    val_dataset = GlaucomaDataset(
        root_dir, 
        split='val', 
        transform=transform,
        filter_incomplete=filter_incomplete
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # Create model
    print(f"\nInitializing Shape Autoencoder (latent_dim={latent_dim})...")
    model = ShapeAutoencoder(n_classes=3, latent_dim=latent_dim)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer with gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kld': [],
        'val_loss': [],
        'val_recon': [],
        'val_kld': []
    }
    
    best_val_loss = float('inf')
    
    # KLD warmup schedule (gradually increase KLD weight)
    warmup_epochs = 20  # Warm up KLD over first 20 epochs
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"KLD Warmup: First {warmup_epochs} epochs")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        # KLD warmup factor (0.0 -> 1.0 over warmup_epochs)
        warmup_factor = min(1.0, epoch / warmup_epochs)
        
        # Training
        model.train()
        train_losses = []
        train_recons = []
        train_klds = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for images, masks in pbar:
            masks = masks.to(device)
            
            # Convert masks to one-hot
            masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=3)
            masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()
            
            # Forward pass
            recon, mu, logvar = model(masks_one_hot)
            loss, recon_loss, kld_loss = vae_loss(
                recon, masks_one_hot, mu, logvar, 
                kld_weight=0.00001,
                warmup_factor=warmup_factor
            )
            
            # Check for NaN before backward
            if torch.isnan(loss):
                print(f"\nWARNING: NaN loss detected at batch {len(train_losses)}, skipping...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            train_recons.append(recon_loss.item())
            train_klds.append(kld_loss.item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kld': f"{kld_loss.item():.4f}",
                'warmup': f"{warmup_factor:.2f}"
            })
        
        # Validation
        model.eval()
        val_losses = []
        val_recons = []
        val_klds = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                masks = masks.to(device)
                
                # Convert masks to one-hot
                masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=3)
                masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()
                
                # Forward pass
                recon, mu, logvar = model(masks_one_hot)
                loss, recon_loss, kld_loss = vae_loss(
                    recon, masks_one_hot, mu, logvar,
                    kld_weight=0.00001,
                    warmup_factor=warmup_factor
                )
                
                # Skip NaN losses in validation too
                if not torch.isnan(loss):
                    val_losses.append(loss.item())
                    val_recons.append(recon_loss.item())
                    val_klds.append(kld_loss.item())
        
        # Record epoch metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_train_recon = np.mean(train_recons)
        epoch_train_kld = np.mean(train_klds)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_recon = np.mean(val_recons)
        epoch_val_kld = np.mean(val_klds)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_recon'].append(epoch_train_recon)
        history['train_kld'].append(epoch_train_kld)
        history['val_loss'].append(epoch_val_loss)
        history['val_recon'].append(epoch_val_recon)
        history['val_kld'].append(epoch_val_kld)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} (Recon: {epoch_train_recon:.4f}, KLD: {epoch_train_kld:.4f})")
        print(f"  Val Loss:   {epoch_val_loss:.4f} (Recon: {epoch_val_recon:.4f}, KLD: {epoch_val_kld:.4f})")
        print(f"  KLD Warmup: {warmup_factor:.2f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 80)
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'latent_dim': latent_dim
            }
            save_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"  ✓ Best model saved (Val Loss: {best_val_loss:.4f})")
        
        print("-" * 80)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history['train_recon'], label='Train')
    axes[1].plot(history['val_recon'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(history['train_kld'], label='Train')
    axes[2].plot(history['val_kld'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"Training history saved to: {save_dir / 'training_history.png'}")
    
    return model, history


if __name__ == '__main__':
    # Training configuration
    config = {
        'root_dir': str(project_root),
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'image_size': 256,
        'latent_dim': 64,
        'num_workers': 4,
        'save_dir': str(project_root / 'shape_autoencoder'),
        'filter_incomplete': True
    }
    
    print("Shape Autoencoder Training Configuration:")
    print("=" * 80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Train
    model, history = train_shape_autoencoder(**config)
