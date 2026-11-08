"""Canny edge detector from scratch, built on top of the Gaussian and
Sobel modules in this package.

Canny has four steps: (1) blur the image to reduce noise, (2) compute the
gradient magnitude and direction with Sobel, (3) non-maximum suppression
(NMS) - thin the thick gradient "ridges" down to single-pixel-wide lines by
keeping only pixels that are a local maximum along the gradient direction,
and (4) double threshold with hysteresis - keep strong edges outright,
throw away weak-looking pixels unless they are connected to a strong edge
(so a faint edge that trails off from a clear one is kept, but isolated
noise is not).
"""

import numpy as np
from scipy import ndimage

from src.ip.gaussian import gaussian_blur
from src.ip.sobel import sobel_gradients


def non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin gradient ridges to one pixel wide.

    For each pixel, we quantize its gradient direction into one of four
    sectors (0, 45, 90, 135 degrees) and compare its magnitude to the two
    neighbours that lie along that direction. The pixel survives only if
    its magnitude is >= both neighbours (a local maximum along the
    gradient). This sector method is a standard simplification of the
    "exact" version (which interpolates between neighbours); it is easier
    to reason about and is what most textbook from-scratch implementations
    use.
    """
    h, w = magnitude.shape
    # angle in degrees, folded into [0, 180) since direction and
    # direction+180 point along the same line (we only compare neighbours
    # along the line, not which end it points to).
    angle_deg = np.degrees(direction) % 180

    padded_mag = np.pad(magnitude, 1, mode="constant", constant_values=0)
    out = np.zeros_like(magnitude)

    # sector -> (dy1, dx1), (dy2, dx2): the two neighbour offsets to compare
    sectors = {
        "horizontal": (0, 1),   # angle near 0/180: edge runs vertically, compare left/right
        "diag_45": (1, 1),      # angle near 45: compare along the other diagonal
        "vertical": (1, 0),     # angle near 90: edge runs horizontally, compare up/down
        "diag_135": (1, -1),    # angle near 135
    }

    def neighbour(dy, dx):
        return padded_mag[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    is_horizontal = (angle_deg < 22.5) | (angle_deg >= 157.5)
    is_diag45 = (angle_deg >= 22.5) & (angle_deg < 67.5)
    is_vertical = (angle_deg >= 67.5) & (angle_deg < 112.5)
    is_diag135 = (angle_deg >= 112.5) & (angle_deg < 157.5)

    for mask, (dy, dx) in [
        (is_horizontal, sectors["horizontal"]),
        (is_diag45, sectors["diag_45"]),
        (is_vertical, sectors["vertical"]),
        (is_diag135, sectors["diag_135"]),
    ]:
        n1 = neighbour(dy, dx)
        n2 = neighbour(-dy, -dx)
        is_local_max = (magnitude >= n1) & (magnitude >= n2)
        out = np.where(mask & is_local_max, magnitude, out)

    return out


def double_threshold_hysteresis(nms_magnitude: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep strong edges, keep weak edges only if connected (8-neighbour)
    to a strong edge, drop everything else. Returns a boolean edge map."""
    strong = nms_magnitude >= high
    weak = (nms_magnitude >= low) & (nms_magnitude < high)
    candidate = strong | weak

    # Label connected components of (strong | weak) using 8-connectivity,
    # then keep a whole component if and only if it contains >= 1 strong
    # pixel. This is "hysteresis": a weak pixel is kept only by virtue of
    # being chained to a strong one, directly or through other weak pixels.
    structure = np.ones((3, 3), dtype=int)
    labels, n_components = ndimage.label(candidate, structure=structure)
    if n_components == 0:
        return np.zeros_like(candidate, dtype=bool)

    strong_labels = set(labels[strong].tolist())
    strong_labels.discard(0)
    keep_mask = np.isin(labels, list(strong_labels))
    return keep_mask


def canny_edge_detector(
    image: np.ndarray,
    sigma: float = 1.4,
    gaussian_size: int = 5,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Full Canny pipeline. low_threshold/high_threshold are fractions of
    the maximum gradient magnitude after NMS (so they adapt to image
    contrast, like cv2.Canny's thresholds do when given raw gradient
    values on a per-image basis -- here we normalize instead so the same
    0-1 fractions work across images of different contrast).

    Returns a boolean edge map, True where there is an edge.
    """
    blurred = gaussian_blur(image, size=gaussian_size, sigma=sigma, pad_mode=pad_mode)
    _, _, magnitude, direction = sobel_gradients(blurred, pad_mode=pad_mode)
    suppressed = non_max_suppression(magnitude, direction)

    max_mag = suppressed.max()
    if max_mag == 0:
        return np.zeros_like(suppressed, dtype=bool)
    low = low_threshold * max_mag
    high = high_threshold * max_mag

    return double_threshold_hysteresis(suppressed, low=low, high=high)
