"""Convolution theorem demo: show spatial convolution and frequency-domain
multiplication give the same numbers, and time both as kernel size grows.

Writes results/a_fourier_timing.csv and results/figures/fourier_timing.png.
"""

import time

import numpy as np
import pandas as pd
from skimage import data as skimage_data

from src.ip.convolution import convolve2d
from src.ip.fourier import convolution_theorem_max_abs_error, fft_convolve2d

SEED = 42
KERNEL_SIZES = [3, 5, 7, 9, 11, 15, 21, 31]


def box_kernel(size: int) -> np.ndarray:
    return np.ones((size, size)) / (size * size)


def main():
    np.random.seed(SEED)
    image = skimage_data.camera().astype(np.float64)  # 512x512

    rows = []
    for size in KERNEL_SIZES:
        kernel = box_kernel(size)

        max_abs_error = convolution_theorem_max_abs_error(image, kernel)

        t0 = time.perf_counter()
        convolve2d(image, kernel, pad_mode="wrap")
        t_spatial = time.perf_counter() - t0

        t0 = time.perf_counter()
        fft_convolve2d(image, kernel)
        t_fft = time.perf_counter() - t0

        rows.append({
            "kernel_size": size,
            "max_abs_error_vs_fft": max_abs_error,
            "time_spatial_sec": t_spatial,
            "time_fft_sec": t_fft,
        })
        print(f"kernel {size}x{size}: max_abs_error={max_abs_error:.2e}, "
              f"spatial={t_spatial:.4f}s, fft={t_fft:.4f}s")

    df = pd.DataFrame(rows)
    df.to_csv("results/a_fourier_timing.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["kernel_size"], df["time_spatial_sec"], marker="o", label="spatial convolve2d")
    ax.plot(df["kernel_size"], df["time_fft_sec"], marker="s", label="FFT multiply (fft_convolve2d)")
    ax.set_xlabel("kernel size (k, for a k x k box kernel)")
    ax.set_ylabel("time (seconds)")
    ax.set_yscale("log")
    ax.set_title("Convolution theorem: spatial vs FFT-domain, on a 512x512 image")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/fourier_timing.png", dpi=150)
    print("\nWrote results/a_fourier_timing.csv and results/figures/fourier_timing.png")


if __name__ == "__main__":
    main()
