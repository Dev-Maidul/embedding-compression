"""HOG (Histogram of Oriented Gradients) features, via scikit-image.

HOG describes an image by the distribution of edge directions in small
local cells: for each cell, it builds a histogram of "how much gradient
there is pointing in each of a few directions", then groups cells into
overlapping blocks and normalizes each block's histograms together (to
reduce sensitivity to lighting changes). The result captures local shape
and edge structure while being far more compact and more robust to small
shifts/lighting changes than raw pixels. See Dalal & Triggs (2005).
"""

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

# Parameters, chosen for 32x32 CIFAR-10 images:
#   orientations=9: 9 gradient-direction bins (the original Dalal &
#     Triggs setting, a standard default).
#   pixels_per_cell=(8, 8): a 32x32 image becomes a 4x4 grid of cells.
#   cells_per_block=(2, 2): blocks of 2x2 cells, overlapping by one cell
#     as they slide, giving 3x3 = 9 block positions.
# Output length = 9 block positions * 2*2 cells/block * 9 orientations = 324.
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """images: uint8 array (N, 32, 32, 3). Returns float32 (N, 324).

    Each image is converted to grayscale first (HOG as commonly used
    operates on a single intensity channel; colour HOG variants exist but
    are not what the project brief asks for), then scikit-image's `hog` computes
    the descriptor with the parameters documented above.
    """
    n = images.shape[0]
    features = []
    for i in range(n):
        gray = rgb2gray(images[i])
        feat = hog(
            gray,
            orientations=ORIENTATIONS,
            pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            feature_vector=True,
        )
        features.append(feat)
    return np.stack(features).astype(np.float32)
