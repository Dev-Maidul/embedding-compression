"""Estimate intrinsic dimension (ID) for each of the 3 CIFAR-10 feature
types, and as a second level, for each of the 10 classes separately (30
(feature, class) combinations total), using the TwoNN and Levina-Bickel
MLE estimators validated in c_intrinsic_dim_validation.py.

Writes results/c_intrinsic_dim.csv.
"""

import time

import numpy as np
import pandas as pd

from src.eval.intrinsic_dim import mle_id, twonn_id

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

FEATURE_FILES = {
    "pixels": "data/features/pixels.npy",
    "hog": "data/features/hog.npy",
    "resnet": "data/features/resnet.npy",
}


def estimate_row(X: np.ndarray, feature: str, class_name: str) -> dict:
    twonn_estimate = twonn_id(X)
    mle_estimate = mle_id(X, k_min=10, k_max=20)
    return {
        "feature": feature,
        "class": class_name,
        "n_points": X.shape[0],
        "d": X.shape[1],
        "twonn_id": twonn_estimate,
        "mle_id": mle_estimate,
    }


def main():
    labels = np.load("data/features/labels.npy")
    rows = []

    for feature_name, path in FEATURE_FILES.items():
        X = np.load(path)
        t0 = time.perf_counter()

        rows.append(estimate_row(X, feature_name, "all"))
        print(f"{feature_name} / all (N={X.shape[0]}): "
              f"TwoNN={rows[-1]['twonn_id']:.2f}, MLE={rows[-1]['mle_id']:.2f}")

        for class_idx, class_name in enumerate(CLASS_NAMES):
            X_class = X[labels == class_idx]
            row = estimate_row(X_class, feature_name, class_name)
            rows.append(row)
            print(f"  {feature_name} / {class_name} (N={X_class.shape[0]}): "
                  f"TwoNN={row['twonn_id']:.2f}, MLE={row['mle_id']:.2f}")

        print(f"  [{feature_name} done in {time.perf_counter() - t0:.1f}s]")

    df = pd.DataFrame(rows)
    df.to_csv("results/c_intrinsic_dim.csv", index=False)
    n_per_class_rows = len(df[df["class"] != "all"])
    print(f"\nWrote results/c_intrinsic_dim.csv "
          f"({len(df)} rows total, {n_per_class_rows} per-class rows)")


if __name__ == "__main__":
    main()
