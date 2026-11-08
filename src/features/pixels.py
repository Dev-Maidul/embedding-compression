"""Raw pixel features: the simplest possible image representation.

Every pixel's RGB value, flattened into one long vector, used as-is as a
3072-dimensional (32*32*3) feature. This is a deliberately weak baseline:
it has no notion of shape or texture, is sensitive to shifts and
brightness changes, and exists so later feature types (HOG, ResNet) have
something simple to be compared against.
"""

import numpy as np


def extract_pixel_features(images: np.ndarray) -> np.ndarray:
    """images: uint8 array (N, 32, 32, 3). Returns float32 (N, 3072).

    Normalization: global scaling, dividing every pixel by 255.0 so values
    land in [0, 1]. We use the SAME fixed scale for every image (not a
    per-image min-max or per-image mean/std normalization). Reason: this
    is a retrieval task comparing images against each other by distance;
    per-image normalization would erase real brightness/contrast
    differences between images before any comparison happens, which
    throws away information a nearest-neighbour search over raw pixels is
    otherwise free to use. Global scaling keeps that information and only
    rescales units, which does not change which images are each other's
    nearest neighbours (it is a uniform scale factor on every vector).
    """
    if images.dtype != np.uint8:
        raise ValueError(f"expected uint8 images, got {images.dtype}")
    n = images.shape[0]
    flat = images.reshape(n, -1).astype(np.float32) / 255.0
    return flat
