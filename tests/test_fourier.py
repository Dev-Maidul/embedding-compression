import numpy as np

from src.ip.fourier import (
    convolution_theorem_max_abs_error,
    fft_convolve2d,
    highpass_filter,
    lowpass_filter,
)


def test_lowpass_plus_highpass_reconstructs_image():
    rng = np.random.default_rng(7)
    img = rng.random((32, 32)) * 255
    low = lowpass_filter(img, cutoff_frac=0.3)
    high = highpass_filter(img, cutoff_frac=0.3)
    # ideal low-pass + its complementary high-pass should sum back to the
    # original image (the two masks partition the spectrum exactly)
    assert np.allclose(low + high, img, atol=1e-6)


def test_lowpass_removes_high_frequency_noise():
    rng = np.random.default_rng(8)
    h = w = 64
    y, x = np.meshgrid(np.arange(h), np.arange(w))
    smooth = np.sin(2 * np.pi * x / 64) * 50 + 128  # low-frequency pattern
    noisy = smooth + rng.normal(0, 20, size=(h, w))  # + high-frequency noise
    denoised = lowpass_filter(noisy, cutoff_frac=0.1)
    # denoised should be closer to the smooth ground truth than the noisy input was
    assert np.mean((denoised - smooth) ** 2) < np.mean((noisy - smooth) ** 2)


def test_fft_convolve2d_matches_spatial_circular_convolution():
    rng = np.random.default_rng(9)
    img = rng.random((20, 20))
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16.0
    max_abs_error = convolution_theorem_max_abs_error(img, kernel)
    assert max_abs_error < 1e-9


def test_fft_convolve2d_asymmetric_kernel_also_matches():
    rng = np.random.default_rng(10)
    img = rng.random((16, 16))
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    max_abs_error = convolution_theorem_max_abs_error(img, kernel)
    assert max_abs_error < 1e-9
