"""2D convolution from scratch with NumPy.

A convolution slides a small kernel over an image and, at every pixel,
replaces the pixel with a weighted sum of its neighbours (the weights come
from the kernel). True mathematical convolution flips the kernel 180 degrees
before sliding it; "correlation" (what OpenCV's `cv2.filter2D` actually
computes) does not flip it. For a symmetric kernel like a Gaussian the two
are identical, but for an asymmetric kernel like a Sobel kernel they differ,
so this module implements both and the tests/experiments flip the kernel
when comparing against `cv2.filter2D` to stay honest about what is being
compared. Padding controls what happens at the border, where the kernel
would otherwise fall off the edge of the image.
"""

import numpy as np


def pad2d(image: np.ndarray, pad_h: int, pad_w: int, mode: str = "zero") -> np.ndarray:
    """Pad a 2D array on all four sides.

    mode="zero": border pixels are 0 (cv2.BORDER_CONSTANT with value 0).
    mode="reflect": border pixels mirror the image without repeating the
    edge pixel (NumPy's "reflect", matches cv2.BORDER_REFLECT_101, which is
    OpenCV's default border mode for filtering).
    mode="wrap": the image is treated as periodic (the right edge wraps
    around to the left, etc). This is the boundary condition under which
    spatial convolution exactly equals multiplication in the frequency
    domain (the convolution theorem), so it is used only by
    `src/ip/fourier.py`'s convolution-theorem demonstration, never for
    ordinary image filtering (wrap-around produces visible seams at the
    image border otherwise).
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {image.shape}")
    if mode == "zero":
        np_mode = "constant"
    elif mode == "reflect":
        np_mode = "reflect"
    elif mode == "wrap":
        np_mode = "wrap"
    else:
        raise ValueError(f"unknown padding mode: {mode}")
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=np_mode)


def correlate2d(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """2D cross-correlation: slide the kernel as-is, no flip.

    This is what cv2.filter2D computes. Output has the same (H, W) as the
    input ("same" convolution): the image is padded by half the kernel size
    on each side first.
    """
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel height and width must be odd for a centered 'same' output")
    pad_h, pad_w = kh // 2, kw // 2
    padded = pad2d(image.astype(np.float64), pad_h, pad_w, mode=pad_mode)

    h, w = image.shape
    # sliding_window_view builds a (h, w, kh, kw) view onto `padded` with no
    # copying: windows[i, j] is exactly the kh x kw patch a naive double
    # `for i in range(h): for j in range(w)` loop would slice out by hand.
    # Multiplying by the kernel and summing the last two axes is the same
    # "multiply-and-add over the patch" definition of correlation, just
    # done for every pixel at once instead of one Python loop iteration
    # at a time, which matters once images stop being tiny.
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    out = np.einsum("ijkl,kl->ij", windows, kernel)
    return out


def convolve2d(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """True 2D convolution: flip the kernel 180 degrees, then correlate."""
    flipped = kernel[::-1, ::-1]
    return correlate2d(image, flipped, pad_mode=pad_mode)
