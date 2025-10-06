"""
Test script for Enhancement Pipeline
Tests defect simulation, EnhancerNet, and enhanced dataset
"""
from models.enhancer_net import EnhancerNet, LightweightEnhancerNet
from data_loader.augmentation import DefectSimulator, simulate_defects
from data_loader.dataset import EnhancedRetinaDataset
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_defect_simulation():
    """Test defect simulation functions"""
    print("\n" + "="*70)
    print("Testing Defect Simulation")
    print("="*70)

    try:
        # Create dummy image
        clean_image = torch.rand(3, 256, 256)

        # Test simple defect simulation
        degraded = simulate_defects(
            clean_image,
            defect_types=['noise', 'blur', 'contrast'],
            intensity='medium'
        )

        print(f"✓ Simple defect simulation successful")
        print(
            f"  Input range:  [{clean_image.min():.3f}, {clean_image.max():.3f}]")
        print(f"  Output range: [{degraded.min():.3f}, {degraded.max():.3f}]")

        # Test DefectSimulator
        simulator = DefectSimulator(
            noise_prob=0.8,
            blur_prob=0.8,
            contrast_prob=0.8,
            num_defects=(1, 3)
        )

        degraded2 = simulator(clean_image)
        print(f"✓ DefectSimulator successful")
        print(
            f"  Output range: [{degraded2.min():.3f}, {degraded2.max():.3f}]")

        # Test batch processing
        batch = torch.rand(4, 3, 256, 256)
        degraded_batch = simulator(batch)
        assert degraded_batch.shape == batch.shape, "Batch shape mismatch!"

        print(f"✓ Batch defect simulation successful")
        print(f"  Batch shape: {degraded_batch.shape}")

        print("✓ Defect simulation test passed!")
        return True

    except Exception as e:
        print(f"✗ Defect simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhancer_models():
    """Test EnhancerNet architectures"""
    print("\n" + "="*70)
    print("Testing EnhancerNet Models")
    print("="*70)

    try:
        x = torch.randn(2, 3, 256, 256)

        # Test standard EnhancerNet
        model = EnhancerNet(base_features=32, num_residual_blocks=4)
        output = model(x)
        num_params = sum(p.numel() for p in model.parameters())

        print(f"\n1. Standard EnhancerNet:")
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters:   {num_params:,}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

        assert output.shape == x.shape, "Output shape mismatch!"
        assert output.min() >= 0 and output.max(
        ) <= 1, "Output not in [0,1] range!"

        # Test lightweight version
        model_light = LightweightEnhancerNet(base_features=16)
        output_light = model_light(x)
        num_params_light = sum(p.numel() for p in model_light.parameters())

        print(f"\n2. Lightweight EnhancerNet:")
        print(f"   Input shape:  {x.shape}")
        print(f"   Output shape: {output_light.shape}")
        print(f"   Parameters:   {num_params_light:,}")
        print(
            f"   Parameter reduction: {(1 - num_params_light/num_params)*100:.1f}%")

        assert output_light.shape == x.shape, "Output shape mismatch!"

        print("\n✓ EnhancerNet tests passed!")
        return True

    except Exception as e:
        print(f"✗ EnhancerNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_dataset():
    """Test EnhancedRetinaDataset"""
    print("\n" + "="*70)
    print("Testing Enhanced Dataset")
    print("="*70)

    try:
        # Create defect simulator
        simulator = DefectSimulator(
            noise_prob=0.8,
            blur_prob=0.8,
            contrast_prob=0.8,
            num_defects=(1, 2)
        )

        # Create dataset
        dataset = EnhancedRetinaDataset(
            root_dir='datasets/REFUGE',
            csv_file='datasets/REFUGE/REFUGETrain.csv',
            defect_simulator=simulator,
            target_size=(256, 256),
            use_cropped=True
        )

        print(f"✓ Enhanced dataset created successfully")
        print(f"  Number of samples: {len(dataset)}")

        # Load one sample
        degraded, clean, mask = dataset[0]

        print(f"✓ Sample loaded successfully")
        print(f"  Degraded image shape: {degraded.shape}")
        print(f"  Clean image shape:    {clean.shape}")
        print(f"  Mask shape:           {mask.shape}")

        assert degraded.shape == (
            3, 256, 256), "Degraded image shape mismatch!"
        assert clean.shape == (3, 256, 256), "Clean image shape mismatch!"
        assert mask.shape == (2, 256, 256), "Mask shape mismatch!"

        print("✓ Enhanced dataset test passed!")
        return True

    except Exception as e:
        print(f"✗ Enhanced dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhancement_pipeline():
    """Test full enhancement pipeline: degradation → enhancement → segmentation"""
    print("\n" + "="*70)
    print("Testing Full Enhancement Pipeline")
    print("="*70)

    try:
        # Create components
        simulator = DefectSimulator(num_defects=(1, 2))
        enhancer = EnhancerNet(base_features=32, num_residual_blocks=4)

        # Create dummy input
        clean_image = torch.rand(2, 3, 256, 256)

        # Apply defects
        degraded = simulator(clean_image)
        print(f"✓ Applied defects to images")

        # Apply enhancement
        enhancer.eval()
        with torch.no_grad():
            enhanced = enhancer(degraded)

        print(f"✓ Enhanced degraded images")
        print(
            f"  Clean range:    [{clean_image.min():.3f}, {clean_image.max():.3f}]")
        print(
            f"  Degraded range: [{degraded.min():.3f}, {degraded.max():.3f}]")
        print(
            f"  Enhanced range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")

        # Calculate simple quality metrics
        mse_degraded = torch.mean((clean_image - degraded) ** 2).item()
        mse_enhanced = torch.mean((clean_image - enhanced) ** 2).item()

        print(f"\n  MSE (degraded vs clean): {mse_degraded:.4f}")
        print(f"  MSE (enhanced vs clean): {mse_enhanced:.4f}")

        # Note: Untrained enhancer won't necessarily improve quality
        print(
            f"\n  Note: Enhancer is untrained, so enhancement may not improve quality yet")

        print("\n✓ Full pipeline test passed!")
        return True

    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_enhancement(save_path='enhancement_visualization.png'):
    """Create visualization of enhancement pipeline (optional)"""
    print("\n" + "="*70)
    print("Creating Enhancement Visualization")
    print("="*70)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # Create components
        simulator = DefectSimulator(num_defects=2)
        enhancer = LightweightEnhancerNet(base_features=16)

        # Load a real image from dataset
        dataset = EnhancedRetinaDataset(
            root_dir='datasets/REFUGE',
            csv_file='datasets/REFUGE/REFUGETrain.csv',
            defect_simulator=simulator,
            target_size=(256, 256),
            use_cropped=True
        )

        degraded, clean, mask = dataset[0]

        # Enhance (untrained, just for visualization)
        enhancer.eval()
        with torch.no_grad():
            enhanced = enhancer(degraded.unsqueeze(0)).squeeze(0)

        # Denormalize for visualization
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return torch.clamp(tensor * std + mean, 0, 1)

        clean_vis = denormalize(clean)
        degraded_vis = denormalize(degraded)
        enhanced_vis = denormalize(enhanced)

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(clean_vis.permute(1, 2, 0).cpu())
        axes[0].set_title('Original Clean Image')
        axes[0].axis('off')

        axes[1].imshow(degraded_vis.permute(1, 2, 0).cpu())
        axes[1].set_title('Degraded Image')
        axes[1].axis('off')

        axes[2].imshow(enhanced_vis.permute(1, 2, 0).cpu())
        axes[2].set_title('Enhanced Image (Untrained)')
        axes[2].axis('off')

        # Show mask
        mask_vis = mask[0].cpu()  # Disc mask
        axes[3].imshow(mask_vis, cmap='gray')
        axes[3].set_title('Optic Disc Mask')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")

        return True

    except Exception as e:
        print(f"⚠ Visualization skipped: {e}")
        return False


def main():
    """Run all enhancement tests"""
    print("\n" + "="*70)
    print(" ENHANCEMENT PIPELINE TEST SUITE")
    print("="*70)

    results = {
        'Defect Simulation': test_defect_simulation(),
        'EnhancerNet Models': test_enhancer_models(),
        'Enhanced Dataset': test_enhanced_dataset(),
        'Full Pipeline': test_enhancement_pipeline()
    }

    print("\n" + "="*70)
    print(" TEST RESULTS SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:.<35} {status}")

    all_passed = all(results.values())

    print("="*70)

    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("\nNext steps:")
        print("  1. Train baseline U-Net on clean images")
        print("  2. Train enhancer with defect simulation")
        print("  3. Evaluate enhancement quality")
        print("  4. Train full pipeline with task-aware loss")
        print("\nTo create visualization:")
        print("  python -c 'from test_enhancement import visualize_enhancement; visualize_enhancement()'")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease fix the errors before proceeding.")

    print("")

    # Try to create visualization
    visualize_enhancement()


if __name__ == '__main__':
    main()
