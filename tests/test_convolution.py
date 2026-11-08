import numpy as np
import pytest

from src.ip.convolution import convolve2d, correlate2d, pad2d


def test_pad2d_zero_shape_and_values():
    img = np.ones((4, 4))
    padded = pad2d(img, 1, 1, mode="zero")
    assert padded.shape == (6, 6)
    assert padded[0, 0] == 0
    assert padded[1, 1] == 1


def test_pad2d_reflect_matches_numpy():
    img = np.arange(16).reshape(4, 4).astype(float)
    padded = pad2d(img, 2, 2, mode="reflect")
    expected = np.pad(img, 2, mode="reflect")
    assert np.allclose(padded, expected)


def test_pad2d_rejects_bad_mode():
    with pytest.raises(ValueError):
        pad2d(np.zeros((3, 3)), 1, 1, mode="nonsense")


def test_correlate2d_identity_kernel_returns_input():
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0
    img = np.random.default_rng(0).random((10, 10))
    out = correlate2d(img, kernel, pad_mode="zero")
    assert np.allclose(out, img)


def test_correlate2d_box_blur_averages_constant_image():
    img = np.full((8, 8), 5.0)
    kernel = np.ones((3, 3)) / 9.0
    out = correlate2d(img, kernel, pad_mode="reflect")
    # a constant image, reflect-padded, stays constant after any box blur
    assert np.allclose(out, 5.0)


def test_convolve2d_flips_kernel_relative_to_correlate2d():
    # an asymmetric kernel: convolution should flip it before sliding
    kernel = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0]], dtype=float)
    img = np.arange(25).reshape(5, 5).astype(float)
    conv_out = convolve2d(img, kernel, pad_mode="zero")
    corr_out = correlate2d(img, kernel[::-1, ::-1], pad_mode="zero")
    assert np.allclose(conv_out, corr_out)


def test_convolve2d_symmetric_kernel_equals_correlate2d():
    kernel = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype=float) / 16.0
    img = np.random.default_rng(1).random((12, 12))
    assert np.allclose(
        convolve2d(img, kernel, pad_mode="reflect"),
        correlate2d(img, kernel, pad_mode="reflect"),
    )


def test_correlate2d_rejects_even_kernel():
    with pytest.raises(ValueError):
        correlate2d(np.zeros((5, 5)), np.zeros((2, 2)), pad_mode="zero")


def test_correlate2d_output_shape_matches_input():
    img = np.random.default_rng(2).random((17, 23))
    kernel = np.ones((5, 5)) / 25.0
    out = correlate2d(img, kernel, pad_mode="reflect")
    assert out.shape == img.shape
