"""Sobel gradient operator from scratch.

Sobel estimates the image's horizontal and vertical intensity derivatives
with two small 3x3 kernels. Combining both derivatives gives the gradient
magnitude (how sharp an edge is at each pixel) and direction (which way the
edge runs), which Canny uses in its next steps.
"""

import numpy as np

from src.ip.convolution import correlate2d


def sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return (Gx, Gy), the standard 3x3 Sobel kernels for the x and y
    derivatives. These are what cv2.Sobel uses internally (ksize=3)."""
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)
    return gx, gy


def sobel_gradients(image: np.ndarray, pad_mode: str = "reflect") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (gx, gy, magnitude, direction) for a 2D grayscale image.

    magnitude = sqrt(gx^2 + gy^2)
    direction = atan2(gy, gx), in radians, range (-pi, pi]
    Despite the name "convolution" that textbooks use loosely for Sobel,
    cv2.Sobel (and the Sobel kernels as conventionally written, with -1 on
    the "low" side and +1 on the "high" side) apply the kernel as-is, with
    no 180-degree flip. That is correlation, not true convolution, so we
    use correlate2d here to match cv2.Sobel's actual, measurable behaviour
    rather than the textbook name.
    """
    gx_kernel, gy_kernel = sobel_kernels()
    gx = correlate2d(image.astype(np.float64), gx_kernel, pad_mode=pad_mode)
    gy = correlate2d(image.astype(np.float64), gy_kernel, pad_mode=pad_mode)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return gx, gy, magnitude, direction
