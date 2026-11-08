"""Dense Gaussian random projection (Johnson & Lindenstrauss, 1984).

Projects a d-dimensional vector down to m dimensions by multiplying it
with a random d x m matrix whose entries are drawn independently from a
Gaussian distribution. The Johnson-Lindenstrauss lemma says that, with
high probability, this approximately preserves pairwise distances between
points -- the whole reason compressing features this way might still allow
accurate nearest-neighbour retrieval, which is what Part C measures.
"""

import numpy as np


def project(X: np.ndarray, m: int, seed: int) -> np.ndarray:
    """X: (N, d) array. Returns (N, m) projected array.

    Entries of the projection matrix R (d, m) are drawn from
    N(0, 1/m) (variance 1/m, i.e. std = 1/sqrt(m)). This scaling makes the
    projection norm-preserving in expectation: E[||Rx||^2] = ||x||^2, so
    that comparing distances after projection is on the same scale as
    before, with no separate rescaling step needed downstream.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    R = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(d, m))
    return X @ R
