import numpy as np
import pytest

from src.features.pixels import extract_pixel_features


def test_pixel_features_shape():
    images = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    feats = extract_pixel_features(images)
    assert feats.shape == (5, 3072)
    assert feats.dtype == np.float32


def test_pixel_features_scaled_to_unit_range():
    images = np.full((2, 32, 32, 3), 255, dtype=np.uint8)
    feats = extract_pixel_features(images)
    assert np.allclose(feats, 1.0)

    images_zero = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    feats_zero = extract_pixel_features(images_zero)
    assert np.allclose(feats_zero, 0.0)


def test_pixel_features_rejects_non_uint8():
    images = np.zeros((2, 32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        extract_pixel_features(images)


def test_pixel_features_preserves_relative_brightness():
    # global (not per-image) scaling: a brighter image should map to a
    # feature vector with a larger sum than a dimmer one.
    dim = np.full((1, 32, 32, 3), 10, dtype=np.uint8)
    bright = np.full((1, 32, 32, 3), 200, dtype=np.uint8)
    feats = extract_pixel_features(np.concatenate([dim, bright], axis=0))
    assert feats[0].sum() < feats[1].sum()
