"""
Experiment utility functions for training, testing, and visualization.
Reduces code duplication across experiment notebooks.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from datetime import datetime

from utils.metrics import batch_metrics


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        tuple: (average_loss, average_dice)
    """
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = batch_metrics(outputs, masks)
            epoch_loss += loss.item()
            epoch_dice += metrics['dice_mean']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice_mean']:.4f}"
        })
    
    return epoch_loss / len(loader), epoch_dice / len(loader)


def validate_epoch(model, loader, criterion, device):
    """
    Validate model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        dict: Validation metrics (loss, dice_mean, dice_disc, dice_cup)
    """
    model.eval()
    epoch_loss = 0
    epoch_dice = 0
    epoch_dice_disc = 0
    epoch_dice_cup = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            metrics = batch_metrics(outputs, masks)
            epoch_loss += loss.item()
            epoch_dice += metrics['dice_mean']
            epoch_dice_disc += metrics['dice_disc']
            epoch_dice_cup += metrics['dice_cup']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice_mean']:.4f}"
            })
    
    return {
        'loss': epoch_loss / len(loader),
        'dice_mean': epoch_dice / len(loader),
        'dice_disc': epoch_dice_disc / len(loader),
        'dice_cup': epoch_dice_cup / len(loader),
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                config, device, output_dir):
    """
    Train model for multiple epochs with validation.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        config: Configuration dictionary
        device: Device to train on
        output_dir: Directory to save checkpoints
        
    Returns:
        dict: Training history
    """
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_dice_disc': [],
        'val_dice_cup': [],
    }
    
    best_dice = 0.0
    start_time = datetime.now()
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice_mean'])
        history['val_dice_disc'].append(val_metrics['dice_disc'])
        history['val_dice_cup'].append(val_metrics['dice_cup'])
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Dice:   {val_metrics['dice_mean']:.4f}")
        print(f"  Val Disc:   {val_metrics['dice_disc']:.4f} | Val Cup:    {val_metrics['dice_cup']:.4f}")
        
        # Save best model
        if val_metrics['dice_mean'] > best_dice:
            best_dice = val_metrics['dice_mean']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': config,
            }, f"{output_dir}/best_model.pth")
            print(f"  ✅ Best model saved (Dice: {best_dice:.4f})")
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 3600
    
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time:.2f} hours")
    print(f"Best validation Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print(f"{'='*60}")
    
    return history


def test_model(model, test_loader, device):
    """
    Test model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to test on
        
    Returns:
        tuple: (test_results dict, predictions list, masks list)
    """
    model.eval()
    
    test_metrics = {
        'dice_mean': [],
        'dice_disc': [],
        'dice_cup': [],
        'cdr_mae': [],
    }
    
    all_predictions = []
    all_masks = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate metrics
            metrics = batch_metrics(outputs, masks)
            test_metrics['dice_mean'].append(metrics['dice_mean'])
            test_metrics['dice_disc'].append(metrics['dice_disc'])
            test_metrics['dice_cup'].append(metrics['dice_cup'])
            test_metrics['cdr_mae'].append(metrics['cdr_mae'])
            
            # Store predictions
            preds = torch.sigmoid(outputs).cpu()
            all_predictions.append(preds)
            all_masks.append(masks.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'dice': f"{metrics['dice_mean']:.4f}",
                'cdr_mae': f"{metrics['cdr_mae']:.4f}"
            })
    
    # Calculate average metrics
    test_results = {
        'dice_mean': np.mean(test_metrics['dice_mean']),
        'dice_disc': np.mean(test_metrics['dice_disc']),
        'dice_cup': np.mean(test_metrics['dice_cup']),
        'cdr_mae': np.mean(test_metrics['cdr_mae']),
    }
    
    # Concatenate predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    return test_results, all_predictions, all_masks


def plot_training_curves(history, output_path, freeze_epoch=None):
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        output_path: Path to save plot
        freeze_epoch: Epoch when encoder was unfrozen (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    if freeze_epoch is not None:
        axes[0].axvline(x=freeze_epoch, color='red', linestyle='--', 
                       label=f'Unfreeze Encoder (Epoch {freeze_epoch})', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice curves
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Val Dice (Overall)', linewidth=2)
    axes[1].plot(history['val_dice_disc'], label='Val Dice (Disc)', linewidth=2, linestyle='--')
    axes[1].plot(history['val_dice_cup'], label='Val Dice (Cup)', linewidth=2, linestyle='--')
    if freeze_epoch is not None:
        axes[1].axvline(x=freeze_epoch, color='red', linestyle='--', 
                       label=f'Unfreeze Encoder', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_predictions(test_dataset, predictions, masks, output_path, 
                          num_samples=4, seed=42):
    """
    Create visualization of predictions.
    
    Args:
        test_dataset: Test dataset
        predictions: Model predictions
        masks: Ground truth masks
        output_path: Path to save visualization
        num_samples: Number of samples to visualize
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Get data
        image, mask_gt = test_dataset[idx]
        pred = predictions[idx]
        
        # Denormalize image
        img_np = image.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # Ground truth masks
        disc_gt = mask_gt[0].numpy()
        cup_gt = mask_gt[1].numpy()
        
        # Predicted masks (binary)
        disc_pred = (pred[0].numpy() > 0.5).astype(float)
        cup_pred = (pred[1].numpy() > 0.5).astype(float)
        
        # Calculate individual metrics
        disc_dice = 2 * (disc_pred * disc_gt).sum() / (disc_pred.sum() + disc_gt.sum() + 1e-8)
        cup_dice = 2 * (cup_pred * cup_gt).sum() / (cup_pred.sum() + cup_gt.sum() + 1e-8)
        overall_dice = (disc_dice + cup_dice) / 2
        
        # CDR
        cdr_gt = cup_gt.sum() / (disc_gt.sum() + 1e-8)
        cdr_pred = cup_pred.sum() / (disc_pred.sum() + 1e-8)
        
        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Sample {idx}\nOverall Dice: {overall_dice:.3f}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(disc_gt, cmap='gray')
        axes[i, 1].set_title(f'GT Disc\n(CDR: {cdr_gt:.3f})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(cup_gt, cmap='gray')
        axes[i, 2].set_title('GT Cup')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(disc_pred, cmap='gray')
        axes[i, 3].set_title(f'Pred Disc\nDice: {disc_dice:.3f}')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(cup_pred, cmap='gray')
        axes[i, 4].set_title(f'Pred Cup\nDice: {cup_dice:.3f}\nCDR: {cdr_pred:.3f}')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_data_samples(dataset, output_path, num_samples=2):
    """
    Visualize sample images from dataset.
    
    Args:
        dataset: Dataset to visualize
        output_path: Path to save visualization
        num_samples: Number of samples to show
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Calculate safe indices to avoid out of bounds
    dataset_len = len(dataset)
    step = max(1, dataset_len // (num_samples * 2))  # Spread samples across dataset
    
    for i in range(num_samples):
        # Get sample - use safe index
        idx = min(i * step, dataset_len - 1)
        image, mask = dataset[idx]
        
        # Denormalize image
        img_np = image.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # Show image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Sample {idx+1}: Image')
        axes[i, 0].axis('off')
        
        # Show disc mask
        axes[i, 1].imshow(mask[0], cmap='gray')
        axes[i, 1].set_title('Optic Disc Mask')
        axes[i, 1].axis('off')
        
        # Show cup mask
        axes[i, 2].imshow(mask[1], cmap='gray')
        axes[i, 2].set_title('Optic Cup Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_config(config, output_path):
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Configuration saved to {output_path}")


def save_history(history, output_path):
    """Save training history to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"✅ Training history saved to {output_path}")


def save_test_results(test_results, output_path):
    """Save test results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"✅ Test results saved to {output_path}")


def print_test_results(test_results, title="TEST RESULTS"):
    """Print test results in a formatted way."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(f"Overall Dice Score:     {test_results['dice_mean']:.4f} ({test_results['dice_mean']*100:.2f}%)")
    print(f"Disc Dice Score:        {test_results['dice_disc']:.4f} ({test_results['dice_disc']*100:.2f}%)")
    print(f"Cup Dice Score:         {test_results['dice_cup']:.4f} ({test_results['dice_cup']*100:.2f}%)")
    print(f"CDR Mean Absolute Error: {test_results['cdr_mae']:.4f}")
    print("="*60)


def print_model_info(model, title="Model Information"):
    """Print model parameter information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{title}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    if total_params != trainable_params:
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
