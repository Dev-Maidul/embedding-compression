import numpy as np

from src.features.hog import extract_hog_features


def test_hog_features_shape():
    rng = np.random.default_rng(12)
    images = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)
    feats = extract_hog_features(images)
    assert feats.shape == (3, 324)
    assert feats.dtype == np.float32


def test_hog_features_deterministic():
    rng = np.random.default_rng(13)
    images = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    feats_a = extract_hog_features(images)
    feats_b = extract_hog_features(images)
    assert np.array_equal(feats_a, feats_b)


def test_hog_features_zero_on_blank_image():
    # a perfectly flat image has no gradients anywhere, so every HOG bin
    # should be zero.
    images = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
    feats = extract_hog_features(images)
    assert np.allclose(feats, 0.0)


def test_hog_features_nonzero_on_edge_image():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, 16:] = 255
    feats = extract_hog_features(img[np.newaxis])
    assert feats.sum() > 0
