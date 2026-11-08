"""Intrinsic dimension (ID) estimators: TwoNN (Facco et al., 2017) and the
Levina-Bickel maximum likelihood estimator (Levina & Bickel, 2004).

"Intrinsic dimension" asks: even though a feature vector has, say, 512
numbers in it, how many of those numbers' worth of real, independent
variation is there in the data -- i.e. if the data actually lies near some
lower-dimensional curved surface (manifold) inside that 512-d space, what
is that surface's dimension? Both estimators below answer this using only
local distances between nearby points, with no assumption about what the
surface looks like.
"""

import numpy as np

from src.eval.knn_recall import pairwise_squared_distances


def k_nearest_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Returns (N, k): for each point, the Euclidean distances (not
    squared) to its k nearest OTHER points, sorted nearest-first."""
    n = X.shape[0]
    sqdist = pairwise_squared_distances(X)
    np.fill_diagonal(sqdist, np.inf)
    part_idx = np.argpartition(sqdist, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    part_sqdist = sqdist[row_idx, part_idx]
    order = np.argsort(part_sqdist, axis=1)
    sorted_sqdist = part_sqdist[row_idx, order]
    return np.sqrt(np.maximum(sorted_sqdist, 0))


def twonn_id(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """TwoNN (Facco et al., 2017): estimate intrinsic dimension from only
    the ratio of each point's 2nd- to 1st-nearest-neighbour distance.

    Idea: if the data locally looks like a flat d-dimensional patch, the
    ratio mu = r2/r1 of the two nearest distances follows a known
    distribution whose shape depends only on d (not on the data's
    density, which is what makes this estimator robust to non-uniform
    sampling). Fitting a straight line (through the origin) between
    log(mu) and -log(1 - empirical_CDF(mu)) gives d as the slope.

    discard_fraction: the largest `discard_fraction` of mu values are
    dropped before fitting (the paper's own recommendation), since the
    empirical CDF is noisiest/least reliable at its extreme tail.
    """
    neighbour_dist = k_nearest_distances(X, k=2)
    r1, r2 = neighbour_dist[:, 0], neighbour_dist[:, 1]

    valid = r1 > 0  # guard against exact duplicate points (r1 == 0)
    mu = r2[valid] / r1[valid]
    mu_sorted = np.sort(mu)

    n = len(mu_sorted)
    empirical_cdf = np.arange(1, n + 1) / n
    n_keep = int(n * (1 - discard_fraction))

    x = np.log(mu_sorted[:n_keep])
    y = -np.log(1 - empirical_cdf[:n_keep])

    # least-squares line through the origin: y = d * x -> d = sum(xy)/sum(x^2)
    return float(np.sum(x * y) / np.sum(x * x))


def mle_id(X: np.ndarray, k_min: int = 10, k_max: int = 20) -> float:
    """Levina & Bickel (2004) maximum-likelihood intrinsic dimension
    estimator, averaged over k = k_min..k_max (the paper's own suggested
    fix for the single-k estimator's strong dependence on k: small k is
    noisy, large k is biased by curvature, so averaging a range trades
    off between the two).

    For a fixed k, the local MLE at point i is derived from the k nearest
    neighbour distances T_i,1 <= ... <= T_i,k by treating them as
    following a Poisson process whose rate depends on the intrinsic
    dimension; the dataset-level estimate for that k is

        m_hat(k) = [ (1 / (N*(k-1))) * sum_i sum_{j=1}^{k-1} log(T_i,k / T_i,j) ]^-1

    and the final estimate is the plain average of m_hat(k) over the k
    range.
    """
    n = X.shape[0]
    distances = k_nearest_distances(X, k=k_max)  # (N, k_max), ascending

    m_hat_per_k = []
    for k in range(k_min, k_max + 1):
        t_k = distances[:, k - 1]            # (N,) distance to k-th neighbour
        t_j = distances[:, : k - 1]           # (N, k-1) distances to 1..k-1
        log_ratios = np.log(t_k[:, None] / t_j)
        mean_log_ratio = np.sum(log_ratios) / (n * (k - 1))
        m_hat_per_k.append(1.0 / mean_log_ratio)

    return float(np.mean(m_hat_per_k))
