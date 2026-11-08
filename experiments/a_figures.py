"""Generate the 3 required Part A figures: original vs edges, original vs
histogram equalization, and original vs frequency-domain filtering.

Writes to results/figures/.
"""

import utils.session
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data as skimage_data

from src.ip.canny import canny_edge_detector
from src.ip.fourier import highpass_filter, lowpass_filter
from src.ip.histeq import histogram_equalize

GAUSSIAN_SIZE = 5
GAUSSIAN_SIGMA = 1.4
CANNY_LOW = 0.1
CANNY_HIGH = 0.2


def main():
    image = skimage_data.camera().astype(np.float64)

    # Figure 1: original vs our Canny edges
    edges = canny_edge_detector(image, sigma=GAUSSIAN_SIGMA, gaussian_size=GAUSSIAN_SIZE,
                                 low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny edges (from scratch)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/figures/canny_edges.png", dpi=150)
    plt.close(fig)

    # Figure 2: original vs histogram-equalized
    img_u8 = np.clip(np.round(image), 0, 255).astype(np.uint8)
    equalized = histogram_equalize(img_u8)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_u8, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(equalized, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Histogram equalized (from scratch)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/figures/histogram_equalization.png", dpi=150)
    plt.close(fig)

    # Figure 3: original vs low-pass vs high-pass frequency filtering
    low = lowpass_filter(image, cutoff_frac=0.1)
    high = highpass_filter(image, cutoff_frac=0.1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(low, cmap="gray")
    axes[1].set_title("Low-pass (cutoff=0.1)")
    axes[2].imshow(high, cmap="gray")
    axes[2].set_title("High-pass (cutoff=0.1)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("results/figures/frequency_filtering.png", dpi=150)
    plt.close(fig)

    print("Wrote results/figures/canny_edges.png")
    print("Wrote results/figures/histogram_equalization.png")
    print("Wrote results/figures/frequency_filtering.png")


if __name__ == "__main__":
    main()
