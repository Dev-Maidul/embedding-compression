"""Benchmark ResNet-18 feature extraction speed at input resolutions
32, 64, and 224, on a subset of CIFAR-10 images, and estimate the time to
extract features for the full 10,000-image retrieval set at each
resolution. The project brief requires measuring this before choosing a
resolution, not guessing.

Writes results/b_resnet_resolution_benchmark.csv.
"""

import time

import numpy as np
import pandas as pd

from src.features.cifar_loader import load_cifar10_test
from src.features.resnet import extract_resnet_features, load_resnet18_feature_extractor

SEED = 42
N_BENCHMARK_IMAGES = 500
N_FULL = 10000
RESOLUTIONS = [32, 64, 224]
BATCH_SIZE = 64


def main():
    np.random.seed(SEED)
    images, _ = load_cifar10_test()
    subset = images[:N_BENCHMARK_IMAGES]

    model = load_resnet18_feature_extractor()

    rows = []
    for resolution in RESOLUTIONS:
        t0 = time.perf_counter()
        feats = extract_resnet_features(subset, model, resolution=resolution, batch_size=BATCH_SIZE)
        elapsed = time.perf_counter() - t0

        sec_per_image = elapsed / N_BENCHMARK_IMAGES
        estimated_full_sec = sec_per_image * N_FULL

        rows.append({
            "resolution": resolution,
            "n_benchmark_images": N_BENCHMARK_IMAGES,
            "elapsed_sec": elapsed,
            "sec_per_image": sec_per_image,
            "estimated_sec_for_10000_images": estimated_full_sec,
            "estimated_min_for_10000_images": estimated_full_sec / 60.0,
            "feature_dim": feats.shape[1],
        })
        print(f"resolution={resolution}: {elapsed:.2f}s for {N_BENCHMARK_IMAGES} images "
              f"({sec_per_image * 1000:.2f} ms/image), "
              f"estimated {estimated_full_sec / 60.0:.1f} min for all {N_FULL} images")

    df = pd.DataFrame(rows)
    df.to_csv("results/b_resnet_resolution_benchmark.csv", index=False)
    print(f"\nWrote results/b_resnet_resolution_benchmark.csv")


if __name__ == "__main__":
    main()
