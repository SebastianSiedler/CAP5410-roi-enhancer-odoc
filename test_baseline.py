"""
Quick test script to verify baseline implementation
Tests dataset loading, model forward pass, and basic training components
"""
from utils.metrics import batch_metrics
from utils.loss_functions import CombinedSegmentationLoss
from models.unet import UNet
from data_loader.dataset import RetinaDataset
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_dataset():
    """Test dataset loading"""
    print("\n" + "="*50)
    print("Testing Dataset...")
    print("="*50)

    try:
        dataset = RetinaDataset(
            root_dir='datasets/REFUGE',
            csv_file='datasets/REFUGE/REFUGETrain.csv',
            target_size=(256, 256),
            use_cropped=True
        )

        print(f"✓ Dataset created successfully")
        print(f"  Number of samples: {len(dataset)}")

        # Test loading one sample
        image, mask = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")

        assert image.shape == (3, 256, 256), "Image shape mismatch!"
        assert mask.shape == (2, 256, 256), "Mask shape mismatch!"

        print("✓ Dataset test passed!")
        return True

    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_model():
    """Test U-Net model"""
    print("\n" + "="*50)
    print("Testing U-Net Model...")
    print("="*50)

    try:
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        print(f"✓ Model created successfully")

        # Test forward pass
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        assert output.shape == (2, 2, 256, 256), "Output shape mismatch!"

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")

        print("✓ Model test passed!")
        return True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_loss_and_metrics():
    """Test loss functions and metrics"""
    print("\n" + "="*50)
    print("Testing Loss and Metrics...")
    print("="*50)

    try:
        # Create dummy data
        pred = torch.randn(2, 2, 256, 256)
        target = torch.randint(0, 2, (2, 2, 256, 256)).float()

        # Test loss
        criterion = CombinedSegmentationLoss(lambda_dice=0.5, lambda_bce=0.5)
        loss = criterion(pred, target)

        print(f"✓ Loss calculation successful")
        print(f"  Loss value: {loss.item():.4f}")

        # Test metrics
        metrics = batch_metrics(pred, target)

        print(f"✓ Metrics calculation successful")
        print(f"  Dice (disc): {metrics['dice_disc']:.4f}")
        print(f"  Dice (cup): {metrics['dice_cup']:.4f}")
        print(f"  Mean Dice: {metrics['dice_mean']:.4f}")
        print(f"  CDR MAE: {metrics['cdr_mae']:.4f}")

        print("✓ Loss and metrics test passed!")
        return True

    except Exception as e:
        print(f"✗ Loss and metrics test failed: {e}")
        return False


def test_training_step():
    """Test a single training step"""
    print("\n" + "="*50)
    print("Testing Training Step...")
    print("="*50)

    try:
        # Create model, loss, optimizer
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        criterion = CombinedSegmentationLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create dummy batch
        images = torch.randn(2, 3, 256, 256)
        masks = torch.randint(0, 2, (2, 2, 256, 256)).float()

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")

        # Validation step
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            val_loss = criterion(outputs, masks)
            metrics = batch_metrics(outputs, masks)

        print(f"✓ Validation step successful")
        print(f"  Val Loss: {val_loss.item():.4f}")
        print(f"  Val Dice: {metrics['dice_mean']:.4f}")

        print("✓ Training step test passed!")
        return True

    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" BASELINE U-NET IMPLEMENTATION TEST SUITE")
    print("="*70)

    results = {
        'Dataset': test_dataset(),
        'Model': test_model(),
        'Loss & Metrics': test_loss_and_metrics(),
        'Training Step': test_training_step()
    }

    print("\n" + "="*70)
    print(" TEST RESULTS SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:.<30} {status}")

    all_passed = all(results.values())

    print("="*70)
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("\nYou can now run training with:")
        print("  cd src")
        print("  python main.py --epochs 100 --batch_size 8")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease fix the errors before running training.")
    print("")


if __name__ == '__main__':
    main()
