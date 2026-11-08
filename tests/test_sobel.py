import cv2
import numpy as np

from src.ip.sobel import sobel_gradients, sobel_kernels


def test_sobel_kernels_shapes():
    gx, gy = sobel_kernels()
    assert gx.shape == (3, 3)
    assert gy.shape == (3, 3)


def test_sobel_kernels_gy_is_gx_transposed():
    gx, gy = sobel_kernels()
    assert np.allclose(gy, gx.T)


def test_sobel_gradients_zero_on_constant_image():
    img = np.full((10, 10), 7.0)
    gx, gy, mag, direction = sobel_gradients(img, pad_mode="reflect")
    assert np.allclose(gx, 0.0, atol=1e-10)
    assert np.allclose(gy, 0.0, atol=1e-10)
    assert np.allclose(mag, 0.0, atol=1e-10)


def test_sobel_gradients_detects_vertical_edge():
    # left half 0, right half 255: a vertical edge, so gx should respond
    # strongly and gy should be ~0 away from the border.
    img = np.zeros((10, 10))
    img[:, 5:] = 255.0
    gx, gy, mag, _ = sobel_gradients(img, pad_mode="zero")
    assert np.abs(gx[5, 5]) > np.abs(gy[5, 5])
    assert mag[5, 5] > 0


def test_sobel_magnitude_matches_opencv():
    rng = np.random.default_rng(4)
    img = (rng.random((40, 40)) * 255).astype(np.float64)
    gx, gy, mag, _ = sobel_gradients(img, pad_mode="reflect")

    ref_gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    ref_gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    ref_mag = np.sqrt(ref_gx ** 2 + ref_gy ** 2)

    assert np.max(np.abs(gx - ref_gx)) < 1e-6
    assert np.max(np.abs(gy - ref_gy)) < 1e-6
    assert np.max(np.abs(mag - ref_mag)) < 1e-6
