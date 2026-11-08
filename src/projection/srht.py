"""Structured random projection: Subsampled Randomized Hadamard Transform
(SRHT).

Dense and sparse Gaussian-style projections (gaussian.py, sparse.py) need
an explicit d x m random matrix. SRHT instead applies three fast,
structured steps: (1) flip the sign of each input coordinate at random,
(2) apply the Hadamard transform (a fixed, structured, orthogonal
transform -- no randomness, and computable in O(d log d) time instead of
the O(d*m) a dense matrix multiply would cost), (3) keep only a random
subset of m of the resulting d coordinates. Step (1) spreads each input
coordinate's "energy" evenly across all Hadamard outputs before
subsampling (without it, a dataset with one unusually large coordinate
could concentrate too much of its energy into a few Hadamard outputs,
making random subsampling risky). This is what makes keeping only a few
output coordinates still preserve distances approximately, while being
far cheaper than a dense random matrix for large d.
"""

import numpy as np


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fwht(X: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform along the last axis of a 2D array
    (N, D), D must be a power of two. Computes X @ H, where H is the
    unnormalized (+1/-1 entries) Hadamard matrix of size D, without ever
    building that D x D matrix -- the standard iterative butterfly
    algorithm, applied to the whole batch of N rows at once.
    """
    X = X.astype(np.float64).copy()
    n, D = X.shape
    h = 1
    while h < D:
        X = X.reshape(n, D // (2 * h), 2, h)
        a = X[:, :, 0, :].copy()
        b = X[:, :, 1, :].copy()
        X[:, :, 0, :] = a + b
        X[:, :, 1, :] = a - b
        X = X.reshape(n, D)
        h *= 2
    return X


def project(X: np.ndarray, m: int, seed: int) -> np.ndarray:
    """X: (N, d) array. Returns (N, m) projected array.

    Steps: pad d up to D, the next power of two (zero-padding contributes
    nothing extra, it just makes the Hadamard transform applicable);
    multiply every column by a random +-1 sign; apply the normalized
    Hadamard transform (divide by sqrt(D) so it is orthogonal, i.e.
    norm-preserving on its own); keep a random subset of m of the D
    coordinates, scaled by sqrt(D/m) to compensate for keeping only a
    fraction m/D of the energy (so the result is still approximately
    norm-preserving, matching gaussian.py/sparse.py's
    E[||Rx||^2] = ||x||^2 convention).
    """
    n, d = X.shape
    D = _next_power_of_two(d)

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=D)
    selected_idx = rng.choice(D, size=m, replace=False)

    if D > d:
        X_padded = np.zeros((n, D), dtype=np.float64)
        X_padded[:, :d] = X
    else:
        X_padded = X.astype(np.float64)

    X_signed = X_padded * signs[np.newaxis, :]
    X_hadamard = _fwht(X_signed) / np.sqrt(D)
    X_selected = X_hadamard[:, selected_idx] * np.sqrt(D / m)
    return X_selected
