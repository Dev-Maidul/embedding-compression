"""Gaussian smoothing from scratch.

The Gaussian kernel weights a pixel's neighbours by a 2D bell curve: nearby
pixels count more than far ones, and the farther a neighbour is, the faster
its weight falls off (controlled by sigma). Convolving an image with this
kernel blurs it, which is the first step of the Canny edge detector (blur
removes noise before taking derivatives) and a standard denoising tool on
its own.
"""

import numpy as np

from src.ip.convolution import correlate2d


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Build a size x size 2D Gaussian kernel, normalized to sum to 1.

    size must be odd so the kernel has a single centre pixel at (0, 0) in
    kernel-centred coordinates.
    """
    if size % 2 == 0:
        raise ValueError("kernel size must be odd")
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.4, pad_mode: str = "reflect") -> np.ndarray:
    """Blur a 2D grayscale image with a Gaussian kernel (correlation, since
    the kernel is symmetric so correlation and convolution give the same
    result)."""
    kernel = gaussian_kernel(size, sigma)
    return correlate2d(image.astype(np.float64), kernel, pad_mode=pad_mode)
