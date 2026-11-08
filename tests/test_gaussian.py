import cv2
import numpy as np
import pytest

from src.ip.gaussian import gaussian_blur, gaussian_kernel


def test_gaussian_kernel_sums_to_one():
    k = gaussian_kernel(5, sigma=1.4)
    assert np.isclose(k.sum(), 1.0)


def test_gaussian_kernel_is_symmetric():
    k = gaussian_kernel(7, sigma=2.0)
    assert np.allclose(k, k.T)
    assert np.allclose(k, k[::-1, ::-1])


def test_gaussian_kernel_peaks_at_center():
    k = gaussian_kernel(5, sigma=1.4)
    center = k[2, 2]
    assert center == k.max()


def test_gaussian_kernel_rejects_even_size():
    with pytest.raises(ValueError):
        gaussian_kernel(4, sigma=1.0)


def test_gaussian_blur_constant_image_unchanged():
    img = np.full((10, 10), 42.0)
    out = gaussian_blur(img, size=5, sigma=1.4, pad_mode="reflect")
    assert np.allclose(out, 42.0)


def test_gaussian_blur_matches_opencv():
    rng = np.random.default_rng(3)
    img = (rng.random((40, 40)) * 255).astype(np.float64)
    size, sigma = 5, 1.4
    ours = gaussian_blur(img, size=size, sigma=sigma, pad_mode="reflect")
    ref = cv2.GaussianBlur(img, (size, size), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)
    max_abs_error = np.max(np.abs(ours - ref))
    assert max_abs_error < 1e-3
