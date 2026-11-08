import cv2
import numpy as np

from src.ip.histeq import histogram_equalize


def test_histeq_output_range():
    rng = np.random.default_rng(5)
    img = (rng.random((30, 30)) * 100 + 50).astype(np.uint8)  # low contrast
    out = histogram_equalize(img)
    assert out.min() == 0 or out.max() == 255  # should stretch toward full range
    assert out.dtype == np.uint8


def test_histeq_constant_image_stays_constant_ish():
    img = np.full((10, 10), 128, dtype=np.uint8)
    out = histogram_equalize(img)
    # a flat image has one histogram bin; equalization can't invent contrast
    assert np.unique(out).size == 1


def test_histeq_matches_opencv():
    rng = np.random.default_rng(6)
    img = (rng.random((50, 50)) * 255).astype(np.uint8)
    ours = histogram_equalize(img)
    ref = cv2.equalizeHist(img)
    max_abs_error = int(np.max(np.abs(ours.astype(int) - ref.astype(int))))
    assert max_abs_error <= 1  # allow 1-level rounding difference
