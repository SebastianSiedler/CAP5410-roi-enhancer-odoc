"""
Training script for Simple Shape Autoencoder (non-variational).

Much more stable than VAE - just learns to reconstruct shapes.
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

from src.models.simple_shape_autoencoder import SimpleShapeAutoencoder, simple_autoencoder_loss
from src.data_loader.dataset import GlaucomaDataset
from src.data_loader.transforms import get_validation_transforms


def train_simple_autoencoder(
    root_dir,
    num_epochs=50,
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
    Train Simple Shape Autoencoder.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load datasets
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
    print(f"\nInitializing Simple Shape Autoencoder (latent_dim={latent_dim})...")
    model = SimpleShapeAutoencoder(n_classes=3, latent_dim=latent_dim)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_bce': [],
        'train_dice': [],
        'val_loss': [],
        'val_bce': [],
        'val_dice': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_losses = []
        train_bces = []
        train_dices = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for images, masks in pbar:
            masks = masks.to(device)
            
            # Convert masks to one-hot
            masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=3)
            masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()
            
            # Forward pass
            recon, latent = model(masks_one_hot)
            loss, bce_loss, dice_loss = simple_autoencoder_loss(recon, masks_one_hot)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            train_bces.append(bce_loss.item())
            train_dices.append(dice_loss.item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bce': f"{bce_loss.item():.4f}",
                'dice': f"{dice_loss.item():.4f}"
            })
        
        # Validation
        model.eval()
        val_losses = []
        val_bces = []
        val_dices = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                masks = masks.to(device)
                
                # Convert masks to one-hot
                masks_one_hot = torch.nn.functional.one_hot(masks, num_classes=3)
                masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()
                
                # Forward pass
                recon, latent = model(masks_one_hot)
                loss, bce_loss, dice_loss = simple_autoencoder_loss(recon, masks_one_hot)
                
                val_losses.append(loss.item())
                val_bces.append(bce_loss.item())
                val_dices.append(dice_loss.item())
        
        # Record epoch metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_train_bce = np.mean(train_bces)
        epoch_train_dice = np.mean(train_dices)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_bce = np.mean(val_bces)
        epoch_val_dice = np.mean(val_dices)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_bce'].append(epoch_train_bce)
        history['train_dice'].append(epoch_train_dice)
        history['val_loss'].append(epoch_val_loss)
        history['val_bce'].append(epoch_val_bce)
        history['val_dice'].append(epoch_val_dice)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f} (BCE: {epoch_train_bce:.4f}, Dice: {epoch_train_dice:.4f})")
        print(f"  Val Loss:   {epoch_val_loss:.4f} (BCE: {epoch_val_bce:.4f}, Dice: {epoch_val_dice:.4f})")
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
                'history': history
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (Val Loss: {best_val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")
    
    return model, history


if __name__ == '__main__':
    config = {
        'root_dir': '/home/robolab/dev/steffen/CAP5410-roi-enhancer-odoc',
        'num_epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.001,
        'image_size': 256,
        'latent_dim': 64,
        'num_workers': 4,
        'save_dir': '/home/robolab/dev/steffen/CAP5410-roi-enhancer-odoc/shape_autoencoder',
        'filter_incomplete': True
    }
    
    print("Simple Shape Autoencoder Training Configuration:")
    print("=" * 80)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    model, history = train_simple_autoencoder(**config)
