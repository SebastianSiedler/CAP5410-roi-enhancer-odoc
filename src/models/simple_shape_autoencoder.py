"""
Simple Shape Autoencoder (NOT Variational) for learning segmentation shape priors.

This is a standard autoencoder (not VAE) which is much more stable for training
on segmentation masks. It learns to reconstruct shapes and can be used for
shape regularization during enhancement training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleShapeAutoencoder(nn.Module):
    """
    Simple Autoencoder for learning shape priors from segmentation masks.
    
    Much more stable than VAE - no KLD term, just reconstruction.
    """
    
    def __init__(self, n_classes=3, latent_dim=64):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_classes, 32, 4, 2, 1),  # 256 -> 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Bottleneck
        self.fc_encode = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, n_classes, 4, 2, 1),  # 128 -> 256
        )
        
    def encode(self, x):
        """Encode masks to latent space"""
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)  # Use reshape instead of view
        z = self.fc_encode(h)
        return z
    
    def decode(self, z):
        """Decode latent codes to masks"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 16, 16)
        recon = self.decoder(h)
        return recon
    
    def forward(self, x):
        """Autoencoder forward pass"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def simple_autoencoder_loss(recon, target):
    """
    Simple reconstruction loss (Dice + BCE).
    
    Args:
        recon: Reconstructed masks [B, C, H, W] (logits)
        target: Ground truth masks [B, C, H, W] (one-hot)
    
    Returns:
        loss: Reconstruction loss
    """
    # BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(recon, target, reduction='mean')
    
    # Dice loss for better shape matching
    recon_sigmoid = torch.sigmoid(recon)
    eps = 1e-7
    
    dice_losses = []
    for c in range(recon.shape[1]):
        pred_c = recon_sigmoid[:, c]
        target_c = target[:, c]
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_losses.append(1.0 - dice)
    
    dice_loss = torch.stack(dice_losses, dim=1).mean()
    
    # Combined loss
    loss = bce_loss + dice_loss
    
    return loss, bce_loss, dice_loss
