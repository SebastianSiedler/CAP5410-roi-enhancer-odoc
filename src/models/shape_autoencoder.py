"""
Shape Autoencoder for learning anatomically plausible segmentation shapes.

This module implements a convolutional autoencoder that learns a latent representation
of anatomically correct optic disc/cup shapes. Used for shape prior regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeEncoder(nn.Module):
    """
    Encoder network that compresses segmentation masks into latent codes.
    """
    
    def __init__(self, n_classes=3, latent_dim=64):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        
        # Convolutional encoder
        self.conv1 = nn.Conv2d(n_classes, 32, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers to latent space
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: Segmentation masks [B, n_classes, H, W]
        
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        # Encoder
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        
        # Flatten - use reshape instead of view for non-contiguous tensors
        h = h.reshape(h.size(0), -1)
        
        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class ShapeDecoder(nn.Module):
    """
    Decoder network that reconstructs segmentation masks from latent codes.
    """
    
    def __init__(self, n_classes=3, latent_dim=64):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        
        # Fully connected to spatial features
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        
        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.bn1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.bn2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.bn3 = nn.BatchNorm2d(32)
        
        self.deconv4 = nn.ConvTranspose2d(32, n_classes, kernel_size=4, stride=2, padding=1)  # 128 -> 256
        
    def forward(self, z):
        """
        Args:
            z: Latent codes [B, latent_dim]
        
        Returns:
            recon: Reconstructed segmentation masks [B, n_classes, H, W]
        """
        # Fully connected
        h = self.fc(z)
        h = h.view(h.size(0), 256, 16, 16)
        
        # Decoder
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        h = F.relu(self.bn3(self.deconv3(h)))
        recon = self.deconv4(h)
        
        return recon


class ShapeAutoencoder(nn.Module):
    """
    Variational Autoencoder for learning shape priors.
    
    This network learns a latent representation of anatomically plausible
    optic disc/cup shapes. The latent space can be used to regularize
    segmentation outputs to have realistic shapes.
    
    Args:
        n_classes: Number of segmentation classes (default: 3 for BG, OD, OC)
        latent_dim: Dimensionality of latent space (default: 64)
    """
    
    def __init__(self, n_classes=3, latent_dim=64):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        
        self.encoder = ShapeEncoder(n_classes, latent_dim)
        self.decoder = ShapeDecoder(n_classes, latent_dim)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        
        Returns:
            z: Sampled latent code [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Args:
            x: Segmentation masks [B, n_classes, H, W]
        
        Returns:
            recon: Reconstructed masks [B, n_classes, H, W]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder(z)
        
        return recon, mu, logvar
    
    def encode(self, x):
        """
        Encode masks to latent space (deterministic).
        
        Args:
            x: Segmentation masks [B, n_classes, H, W]
        
        Returns:
            mu: Latent codes [B, latent_dim]
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        """
        Decode latent codes to masks.
        
        Args:
            z: Latent codes [B, latent_dim]
        
        Returns:
            recon: Reconstructed masks [B, n_classes, H, W]
        """
        return self.decoder(z)


def vae_loss(recon, target, mu, logvar, kld_weight=0.00001, warmup_factor=1.0):
    """
    Stable VAE loss with optional KLD warmup.
    
    Args:
        recon: Reconstructed output [B, C, H, W] (logits)
        target: Ground truth [B, C, H, W] (one-hot, values 0 or 1)
        mu: Latent mean [B, latent_dim]
        logvar: Latent log variance [B, latent_dim]
        kld_weight: Weight for KLD loss (default: 0.00001 for extreme stability)
        warmup_factor: KLD warmup factor (0.0 to 1.0, gradually increase during training)
    
    Returns:
        loss: Total VAE loss
        recon_loss: Reconstruction loss
        kld_loss: KL divergence loss (unweighted for logging)
    """
    # Reconstruction loss - Use Dice-like loss for better stability with masks
    eps = 1e-7
    recon_sigmoid = torch.sigmoid(recon)
    
    # Per-class Dice loss
    dice_losses = []
    for c in range(recon.shape[1]):
        pred_c = recon_sigmoid[:, c]
        target_c = target[:, c]
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_losses.append(1.0 - dice)
    
    recon_loss = torch.stack(dice_losses, dim=1).mean()
    
    # KL divergence with aggressive clamping
    logvar_clamped = torch.clamp(logvar, min=-5, max=2)
    mu_clamped = torch.clamp(mu, min=-5, max=5)
    
    # Standard KLD formula with mean reduction
    kld_loss = -0.5 * torch.mean(1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp())
    kld_loss = torch.clamp(kld_loss, min=0.0, max=100.0)  # Prevent negative or explosion
    
    # Total loss with warmup and very small weight
    effective_kld_weight = kld_weight * warmup_factor
    loss = recon_loss + effective_kld_weight * kld_loss
    
    # Safety check for NaN
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: NaN/Inf detected! recon={recon_loss.item():.4f}, kld={kld_loss.item():.4f}")
        loss = recon_loss  # Fall back to reconstruction only
    
    return loss, recon_loss, kld_loss
