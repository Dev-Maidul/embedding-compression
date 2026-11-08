"""Histogram equalization from scratch.

A dark or low-contrast image has most of its pixel values crowded into a
narrow range. Histogram equalization stretches that range out so pixel
values are spread as evenly as possible across 0-255, which usually makes
detail easier to see. It works purely from the image's own histogram, with
no kernel and no neighbours involved (unlike convolution-based methods).
"""

import numpy as np


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """Equalize a single-channel uint8 (or round-to-uint8) image.

    Steps: build the 256-bin histogram, turn it into a cumulative
    distribution function (CDF), rescale the CDF to [0, 255], then look up
    every pixel's new value in that rescaled CDF. This is exactly what
    `cv2.equalizeHist` does.
    """
    img_u8 = np.clip(np.round(image), 0, 255).astype(np.uint8)
    histogram, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 256))

    cdf = histogram.cumsum()
    cdf_min = cdf[cdf > 0].min()  # smallest nonzero CDF value, for rescaling
    total_pixels = img_u8.size

    if total_pixels == cdf_min:
        # every pixel has the same value: only one histogram bin is
        # nonzero, so there is no contrast to stretch. Map that single
        # value to itself rather than dividing by zero.
        return img_u8

    # Rescale so the CDF spans [0, 255] rather than [cdf_min, total_pixels].
    # This is the standard formula; without subtracting cdf_min, images
    # whose darkest value isn't 0 would never reach a mapped value of 0.
    lookup_table = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)

    return lookup_table[img_u8]
