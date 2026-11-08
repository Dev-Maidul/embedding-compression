"""Validate TwoNN and the Levina-Bickel MLE on synthetic data with known
intrinsic dimension, BEFORE trusting them on real features. The project brief
requires this validation to exist and to be included in the report.

Two synthetic cases:
  1. A 5-d linear subspace embedded in 100-d ambient space (points are
     drawn as 5-d Gaussians, then rotated into a random 5-d subspace of
     R^100 so the ambient dimension is "wrong" and only a correct
     estimator recovers the true 5).
  2. A 10-d hypercube ([0,1]^10) similarly embedded in 100-d.

Writes results/c_id_validation.csv.
"""

import numpy as np
import pandas as pd

from src.eval.intrinsic_dim import mle_id, twonn_id

SEED = 42
N_POINTS = 5000
AMBIENT_DIM = 100


def embed_in_higher_dim(points: np.ndarray, ambient_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Rotate `points` (N, d) into a random d-dimensional subspace of
    R^ambient_dim via a random orthogonal matrix, so the data's intrinsic
    dimension stays d but it is no longer axis-aligned with the ambient
    coordinates (a harder, more honest test than just zero-padding)."""
    d = points.shape[1]
    random_matrix = rng.standard_normal((ambient_dim, ambient_dim))
    q, _ = np.linalg.qr(random_matrix)
    padded = np.zeros((points.shape[0], ambient_dim))
    padded[:, :d] = points
    return padded @ q.T


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    cases = {
        "5d_linear_subspace_in_100d": (5, lambda n, d, r: r.standard_normal((n, d))),
        "10d_hypercube_in_100d": (10, lambda n, d, r: r.uniform(0, 1, size=(n, d))),
    }

    for case_name, (true_dim, generator) in cases.items():
        points = generator(N_POINTS, true_dim, rng)
        X = embed_in_higher_dim(points, AMBIENT_DIM, rng)

        twonn_estimate = twonn_id(X)
        mle_estimate = mle_id(X, k_min=10, k_max=20)

        rows.append({
            "case": case_name,
            "true_dim": true_dim,
            "ambient_dim": AMBIENT_DIM,
            "n_points": N_POINTS,
            "twonn_estimate": twonn_estimate,
            "twonn_abs_error": abs(twonn_estimate - true_dim),
            "mle_estimate": mle_estimate,
            "mle_abs_error": abs(mle_estimate - true_dim),
        })
        print(f"{case_name}: true={true_dim}, TwoNN={twonn_estimate:.2f}, MLE={mle_estimate:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/c_id_validation.csv", index=False)
    print("\nWrote results/c_id_validation.csv")


if __name__ == "__main__":
    main()
