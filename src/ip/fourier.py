"""Frequency-domain filtering and the convolution theorem, using np.fft.

Any image can be written as a sum of 2D sine/cosine waves of different
frequencies (the Fourier transform). Low frequencies are smooth, slowly
changing brightness (the image's overall shape); high frequencies are
fine detail, texture, and noise. A low-pass filter keeps only the low
frequencies (blurring the image); a high-pass filter keeps only the high
frequencies (sharpening/edge-like output).

The convolution theorem says: convolving two signals in the spatial domain
is the same as multiplying their Fourier transforms in the frequency
domain, then transforming back. This module both filters images this way
and directly demonstrates that theorem numerically (same numbers, two
different routes to get them).
"""

import numpy as np

from src.ip.convolution import convolve2d


def _frequency_grid(shape: tuple[int, int]) -> np.ndarray:
    """Distance of every frequency-domain pixel from the center (DC
    component), after fftshift, used to build circular masks."""
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.meshgrid(np.arange(h) - cy, np.arange(w) - cx, indexing="ij")
    return np.sqrt(y ** 2 + x ** 2)


def lowpass_filter(image: np.ndarray, cutoff_frac: float) -> np.ndarray:
    """Ideal circular low-pass filter. cutoff_frac in (0, 1] is the cutoff
    radius as a fraction of the maximum possible radius (corner distance),
    so it is comparable across image sizes."""
    h, w = image.shape
    radius = _frequency_grid((h, w))
    max_radius = np.sqrt((h // 2) ** 2 + (w // 2) ** 2)
    mask = radius <= (cutoff_frac * max_radius)

    spectrum = np.fft.fftshift(np.fft.fft2(image))
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return np.real(filtered)


def highpass_filter(image: np.ndarray, cutoff_frac: float) -> np.ndarray:
    """Ideal circular high-pass filter: the complement of the low-pass
    mask at the same cutoff."""
    h, w = image.shape
    radius = _frequency_grid((h, w))
    max_radius = np.sqrt((h // 2) ** 2 + (w // 2) ** 2)
    mask = radius > (cutoff_frac * max_radius)

    spectrum = np.fft.fftshift(np.fft.fft2(image))
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return np.real(filtered)


def fft_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve `image` with `kernel` by multiplying in the frequency
    domain (the convolution theorem), assuming periodic (wrap-around)
    boundaries -- the condition under which this is exact, not approximate.

    The kernel is placed with its center at array index (0, 0) (via
    np.roll) before transforming, because FFT-based convolution is
    naturally "centered" at the origin, not at the middle of the array the
    way a spatial sliding-window kernel is.
    """
    kh, kw = kernel.shape
    h, w = image.shape
    padded_kernel = np.zeros((h, w), dtype=np.float64)
    padded_kernel[:kh, :kw] = kernel
    # shift so the kernel's center lands at (0, 0), matching the
    # convention that correlate2d/convolve2d use a kernel centered on the
    # pixel being computed.
    padded_kernel = np.roll(padded_kernel, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))

    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(padded_kernel)
    result = np.fft.ifft2(image_fft * kernel_fft)
    return np.real(result)


def convolution_theorem_max_abs_error(image: np.ndarray, kernel: np.ndarray) -> float:
    """Compare spatial (circular/'wrap') convolution against FFT-domain
    multiplication on the same image+kernel; they should agree to
    floating-point precision since both compute the same periodic
    convolution."""
    spatial = convolve2d(image, kernel, pad_mode="wrap")
    frequency = fft_convolve2d(image, kernel)
    return float(np.max(np.abs(spatial - frequency)))
