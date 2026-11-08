"""Sparse (Achlioptas, 2003) random projection.

Achlioptas showed that the dense Gaussian projection matrix can be
replaced with a much sparser, cheaper-to-generate-and-multiply matrix
whose entries are mostly zero, while keeping the same distance-preserving
guarantee (in expectation). This matters in practice because a sparse
matrix needs less memory and fewer multiplications than a dense Gaussian
one, at the scale this project deliberately stays small enough not to
need that benefit -- but the point of including it is to compare whether
a much simpler, discrete random matrix preserves retrieval accuracy as
well as a dense Gaussian one.
"""

import numpy as np


def project(X: np.ndarray, m: int, seed: int) -> np.ndarray:
    """X: (N, d) array. Returns (N, m) projected array.

    Entries of the unscaled matrix are drawn from {+sqrt(3), 0, -sqrt(3)}
    with probabilities {1/6, 2/3, 1/6} (Achlioptas's construction: 2/3 of
    entries are exactly zero, which is the "sparse" part). The whole
    matrix is then scaled by 1/sqrt(m), matching the dense Gaussian
    projection's norm-preserving scale: E[||Rx||^2] = ||x||^2.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    raw = rng.choice([-1.0, 0.0, 1.0], size=(d, m), p=[1 / 6, 2 / 3, 1 / 6])
    R = (np.sqrt(3.0) * raw) / np.sqrt(m)
    return X @ R
