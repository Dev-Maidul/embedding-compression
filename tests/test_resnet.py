import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.features.resnet import extract_resnet_features, load_resnet18_feature_extractor

# Loading pretrained weights needs network access the first time (cached
# under ~/.cache/torch afterwards). If that fails (e.g. no internet in a
# CI sandbox), skip rather than fail the whole suite.
try:
    _MODEL = load_resnet18_feature_extractor()
except Exception as exc:  # pragma: no cover
    _MODEL = None
    _LOAD_ERROR = exc


@pytest.mark.skipif(_MODEL is None, reason="could not load pretrained ResNet-18 weights")
def test_resnet_features_shape():
    rng = np.random.default_rng(14)
    images = rng.integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    feats = extract_resnet_features(images, _MODEL, resolution=32, batch_size=2)
    assert feats.shape == (4, 512)
    assert feats.dtype == np.float32


@pytest.mark.skipif(_MODEL is None, reason="could not load pretrained ResNet-18 weights")
def test_resnet_features_deterministic_in_eval_mode():
    rng = np.random.default_rng(15)
    images = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    feats_a = extract_resnet_features(images, _MODEL, resolution=32, batch_size=2)
    feats_b = extract_resnet_features(images, _MODEL, resolution=32, batch_size=2)
    assert np.allclose(feats_a, feats_b)


@pytest.mark.skipif(_MODEL is None, reason="could not load pretrained ResNet-18 weights")
def test_resnet_features_different_images_give_different_features():
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    images[1] = 255
    feats = extract_resnet_features(images, _MODEL, resolution=32, batch_size=2)
    assert not np.allclose(feats[0], feats[1])
