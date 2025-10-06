"""
Defect Simulation for Image Quality Degradation
Simulates various quality issues found in fundus images:
- Gaussian noise
- Gaussian blur
- Contrast reduction
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
import random


def add_gaussian_noise(
    image: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.05
) -> torch.Tensor:
    """
    Add Gaussian noise to image

    Args:
        image: Input image tensor of shape (C, H, W) or (B, C, H, W)
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise

    Returns:
        Noisy image (clipped to [0, 1] range)
    """
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)


def add_gaussian_blur(
    image: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.5
) -> torch.Tensor:
    """
    Apply Gaussian blur to image

    Args:
        image: Input image tensor of shape (C, H, W) or (B, C, H, W)
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Blurred image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create Gaussian kernel
    channels = image.shape[-3] if len(image.shape) == 4 else image.shape[0]
    kernel = _create_gaussian_kernel(kernel_size, sigma, channels)

    # Apply convolution
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        blurred = F.conv2d(
            image, kernel, padding=kernel_size // 2, groups=channels)
        return blurred.squeeze(0)
    else:
        return F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)


def reduce_contrast(
    image: torch.Tensor,
    reduction_factor: float = 0.5
) -> torch.Tensor:
    """
    Reduce image contrast

    Args:
        image: Input image tensor of shape (C, H, W) or (B, C, H, W)
        reduction_factor: Factor to reduce contrast (0-1, where 0=no change, 1=gray)

    Returns:
        Contrast-reduced image
    """
    # Calculate mean intensity
    mean_intensity = image.mean(dim=(-2, -1), keepdim=True)

    # Reduce contrast by moving pixel values toward mean
    reduced = image * (1 - reduction_factor) + \
        mean_intensity * reduction_factor

    return torch.clamp(reduced, 0.0, 1.0)


def adjust_brightness(
    image: torch.Tensor,
    factor: float = 0.8
) -> torch.Tensor:
    """
    Adjust image brightness

    Args:
        image: Input image tensor
        factor: Brightness factor (<1=darker, >1=brighter)

    Returns:
        Brightness-adjusted image
    """
    return torch.clamp(image * factor, 0.0, 1.0)


def _create_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    channels: int
) -> torch.Tensor:
    """Create a 2D Gaussian kernel for convolution"""
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32)
    x = x - (kernel_size - 1) / 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # Create 2D kernel
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()

    return kernel


class DefectSimulator:
    """
    Configurable defect simulator for training
    Can apply single or multiple defects to images
    """

    def __init__(
        self,
        noise_prob: float = 0.8,
        noise_std_range: Tuple[float, float] = (0.01, 0.1),
        blur_prob: float = 0.8,
        blur_kernel_range: Tuple[int, int] = (3, 9),
        blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
        contrast_prob: float = 0.8,
        contrast_range: Tuple[float, float] = (0.3, 0.7),
        brightness_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.7, 0.9),
        num_defects: Union[int, Tuple[int, int]] = (1, 3)
    ):
        """
        Args:
            noise_prob: Probability of applying noise
            noise_std_range: Range of noise standard deviation
            blur_prob: Probability of applying blur
            blur_kernel_range: Range of blur kernel sizes (odd numbers)
            blur_sigma_range: Range of blur sigma values
            contrast_prob: Probability of reducing contrast
            contrast_range: Range of contrast reduction factors
            brightness_prob: Probability of adjusting brightness
            brightness_range: Range of brightness factors
            num_defects: Number of defects to apply (int or tuple for range)
        """
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.blur_sigma_range = blur_sigma_range
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range
        self.num_defects = num_defects

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply random defects to image

        Args:
            image: Input image tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Degraded image
        """
        # Determine number of defects to apply
        if isinstance(self.num_defects, tuple):
            n_defects = random.randint(
                self.num_defects[0], self.num_defects[1])
        else:
            n_defects = self.num_defects

        # Available defects with their probabilities
        defects = []
        if random.random() < self.noise_prob:
            defects.append('noise')
        if random.random() < self.blur_prob:
            defects.append('blur')
        if random.random() < self.contrast_prob:
            defects.append('contrast')
        if random.random() < self.brightness_prob:
            defects.append('brightness')

        # Randomly select defects to apply
        if len(defects) > n_defects:
            defects = random.sample(defects, n_defects)

        # Apply defects
        degraded = image.clone()

        for defect in defects:
            if defect == 'noise':
                std = random.uniform(*self.noise_std_range)
                degraded = add_gaussian_noise(degraded, std=std)

            elif defect == 'blur':
                # Ensure odd kernel size
                kernel_size = random.randint(self.blur_kernel_range[0] // 2,
                                             self.blur_kernel_range[1] // 2) * 2 + 1
                sigma = random.uniform(*self.blur_sigma_range)
                degraded = add_gaussian_blur(
                    degraded, kernel_size=kernel_size, sigma=sigma)

            elif defect == 'contrast':
                factor = random.uniform(*self.contrast_range)
                degraded = reduce_contrast(degraded, reduction_factor=factor)

            elif defect == 'brightness':
                factor = random.uniform(*self.brightness_range)
                degraded = adjust_brightness(degraded, factor=factor)

        return degraded


def simulate_defects(
    image: torch.Tensor,
    defect_types: List[str] = ['noise', 'blur', 'contrast'],
    intensity: str = 'medium'
) -> torch.Tensor:
    """
    Simple interface to apply predefined defects

    Args:
        image: Input image tensor
        defect_types: List of defect types to apply
        intensity: Defect intensity ('low', 'medium', 'high')

    Returns:
        Degraded image
    """
    intensity_params = {
        'low': {'noise': 0.02, 'blur_k': 3, 'blur_s': 0.5, 'contrast': 0.2},
        'medium': {'noise': 0.05, 'blur_k': 5, 'blur_s': 1.0, 'contrast': 0.4},
        'high': {'noise': 0.1, 'blur_k': 7, 'blur_s': 1.5, 'contrast': 0.6}
    }

    params = intensity_params.get(intensity, intensity_params['medium'])
    degraded = image.clone()

    if 'noise' in defect_types:
        degraded = add_gaussian_noise(degraded, std=params['noise'])

    if 'blur' in defect_types:
        degraded = add_gaussian_blur(degraded,
                                     kernel_size=params['blur_k'],
                                     sigma=params['blur_s'])

    if 'contrast' in defect_types:
        degraded = reduce_contrast(
            degraded, reduction_factor=params['contrast'])

    return degraded
