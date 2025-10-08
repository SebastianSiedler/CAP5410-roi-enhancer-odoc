"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Preprocessing

CLAHE is particularly effective for medical imaging, especially fundus images:
- Enhances local contrast in different regions
- Prevents over-amplification of noise (contrast limiting)
- Adaptive: Works on small tiles rather than entire image
- Improves visibility of optic disc and cup boundaries

Benefits for Retinal Imaging:
1. Better contrast between disc and background
2. Enhanced cup boundaries (often low contrast)
3. Reduced illumination variations
4. Improved feature visibility in dark/bright regions
"""

import cv2
import numpy as np
from PIL import Image


class CLAHEPreprocessor:
    """
    Apply CLAHE preprocessing to fundus images.
    
    CLAHE parameters:
    - clipLimit: Threshold for contrast limiting (higher = more contrast)
    - tileGridSize: Size of grid for histogram equalization (smaller = more local)
    
    For fundus images:
    - clipLimit: 2.0-3.0 works well (prevents over-enhancement)
    - tileGridSize: (8,8) standard for 512x512 images
    """
    
    def __init__(
        self,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        apply_to='LAB'  # Options: 'LAB', 'HSV', 'RGB', 'GREEN'
    ):
        """
        Args:
            clip_limit: Threshold for contrast limiting (1.0-4.0 typical)
            tile_grid_size: Size of grid for adaptive histogram equalization
            apply_to: Which color space to apply CLAHE
                - 'LAB': Apply to L channel (luminance) - RECOMMENDED
                - 'HSV': Apply to V channel (value)
                - 'RGB': Apply to all channels
                - 'GREEN': Apply only to green channel (common in fundus)
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.apply_to = apply_to
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
    
    def __call__(self, image):
        """
        Apply CLAHE preprocessing to image.
        
        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB format
            
        Returns:
            PIL Image with CLAHE applied
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Apply CLAHE based on selected mode
        if self.apply_to == 'LAB':
            enhanced = self._apply_lab(img_array)
        elif self.apply_to == 'HSV':
            enhanced = self._apply_hsv(img_array)
        elif self.apply_to == 'RGB':
            enhanced = self._apply_rgb(img_array)
        elif self.apply_to == 'GREEN':
            enhanced = self._apply_green(img_array)
        else:
            raise ValueError(f"Invalid apply_to: {self.apply_to}")
        
        # Convert back to PIL Image
        return Image.fromarray(enhanced)
    
    def _apply_lab(self, img_rgb):
        """
        Apply CLAHE to L channel in LAB color space.
        This is the RECOMMENDED method for fundus images.
        
        LAB separates luminance (L) from color (A, B), so we only
        enhance brightness/contrast without affecting colors.
        """
        # Convert RGB to LAB
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        
        # Apply CLAHE to L channel
        l_channel_clahe = self.clahe.apply(l_channel)
        
        # Merge channels back
        img_lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
        
        # Convert back to RGB
        img_rgb_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
        
        return img_rgb_clahe
    
    def _apply_hsv(self, img_rgb):
        """
        Apply CLAHE to V channel in HSV color space.
        Alternative method that enhances brightness.
        """
        # Convert RGB to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Split channels
        h_channel, s_channel, v_channel = cv2.split(img_hsv)
        
        # Apply CLAHE to V channel
        v_channel_clahe = self.clahe.apply(v_channel)
        
        # Merge channels back
        img_hsv_clahe = cv2.merge([h_channel, s_channel, v_channel_clahe])
        
        # Convert back to RGB
        img_rgb_clahe = cv2.cvtColor(img_hsv_clahe, cv2.COLOR_HSV2RGB)
        
        return img_rgb_clahe
    
    def _apply_rgb(self, img_rgb):
        """
        Apply CLAHE to all RGB channels separately.
        Can cause color shifts - not recommended for fundus.
        """
        # Split channels
        r_channel, g_channel, b_channel = cv2.split(img_rgb)
        
        # Apply CLAHE to each channel
        r_channel_clahe = self.clahe.apply(r_channel)
        g_channel_clahe = self.clahe.apply(g_channel)
        b_channel_clahe = self.clahe.apply(b_channel)
        
        # Merge channels back
        img_rgb_clahe = cv2.merge([r_channel_clahe, g_channel_clahe, b_channel_clahe])
        
        return img_rgb_clahe
    
    def _apply_green(self, img_rgb):
        """
        Apply CLAHE only to green channel.
        
        Green channel has best contrast in fundus images:
        - Red channel: Too bright (blood vessels saturated)
        - Blue channel: Too dark (low information)
        - Green channel: Best balance
        
        This is a common approach in retinal imaging.
        """
        # Split channels
        r_channel, g_channel, b_channel = cv2.split(img_rgb)
        
        # Apply CLAHE only to green channel
        g_channel_clahe = self.clahe.apply(g_channel)
        
        # Merge channels back
        img_rgb_clahe = cv2.merge([r_channel, g_channel_clahe, b_channel])
        
        return img_rgb_clahe


def compare_clahe_methods(image_path, save_path=None):
    """
    Utility function to compare different CLAHE methods visually.
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save comparison (e.g., 'clahe_comparison.png')
    """
    import matplotlib.pyplot as plt
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Create different CLAHE processors
    clahe_lab = CLAHEPreprocessor(clip_limit=2.0, apply_to='LAB')
    clahe_hsv = CLAHEPreprocessor(clip_limit=2.0, apply_to='HSV')
    clahe_green = CLAHEPreprocessor(clip_limit=2.0, apply_to='GREEN')
    clahe_rgb = CLAHEPreprocessor(clip_limit=2.0, apply_to='RGB')
    
    # Apply each method
    img_lab = clahe_lab(img)
    img_hsv = clahe_hsv(img)
    img_green = clahe_green(img)
    img_rgb = clahe_rgb(img)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_lab)
    axes[0, 1].set_title('CLAHE (LAB) - Recommended')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_hsv)
    axes[0, 2].set_title('CLAHE (HSV)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_green)
    axes[1, 0].set_title('CLAHE (Green Channel)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_rgb)
    axes[1, 1].set_title('CLAHE (RGB)')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()
    
    return img_lab, img_hsv, img_green, img_rgb


# Test function
def test_clahe():
    """Test CLAHE preprocessing with dummy data"""
    print("=== Testing CLAHE Preprocessor ===")
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_img_pil = Image.fromarray(dummy_img)
    
    # Test each method
    methods = ['LAB', 'HSV', 'RGB', 'GREEN']
    
    for method in methods:
        clahe = CLAHEPreprocessor(clip_limit=2.0, apply_to=method)
        result = clahe(dummy_img_pil)
        print(f"✓ CLAHE ({method}): {type(result)}, size={result.size}")
    
    print("\n=== All CLAHE tests passed! ===")


if __name__ == '__main__':
    test_clahe()
