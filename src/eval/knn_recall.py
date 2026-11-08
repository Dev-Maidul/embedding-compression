"""Exact k-nearest-neighbour retrieval and recall@k, used to measure how
much nearest-neighbour quality survives dimensionality reduction.

The protocol (the project brief): the same set of vectors is both the query set
and the database ("leave the query out" -- a vector is never its own
neighbour). Ground truth is the exact top-k Euclidean neighbours in the
original, unprojected feature space; after projecting to m dimensions, we
recompute the top-k there and measure how much overlap survives.
"""

import numpy as np


def pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """All-pairs squared Euclidean distance matrix for X (N, d), via the
    identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b, which is much faster
    than looping over pairs because it reduces to one matrix multiply."""
    sq_norms = np.sum(X.astype(np.float64) ** 2, axis=1)
    gram = X.astype(np.float64) @ X.astype(np.float64).T
    dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    np.maximum(dist, 0, out=dist)  # clip tiny negative values from float error
    return dist


def exact_topk(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Returns (N, k) int array: for each row, the indices of its k
    nearest OTHER rows (self excluded), sorted nearest-first."""
    n = X.shape[0]
    dist = pairwise_squared_distances(X)
    np.fill_diagonal(dist, np.inf)  # a vector is never its own neighbour

    # argpartition first (O(N) per row) to find the k smallest without a
    # full sort, then sort just those k by true distance for a stable,
    # nearest-first order.
    part_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(dist[row_idx, part_idx], axis=1)
    return part_idx[row_idx, order]


def recall_at_k(true_neighbours: np.ndarray, approx_neighbours: np.ndarray) -> np.ndarray:
    """true_neighbours, approx_neighbours: (N, k) index arrays.
    Returns (N,) array: for each query, |true ∩ approx| / k.
    """
    n, k = true_neighbours.shape
    recalls = np.empty(n, dtype=np.float64)
    for i in range(n):
        recalls[i] = len(set(true_neighbours[i].tolist()) & set(approx_neighbours[i].tolist())) / k
    return recalls
