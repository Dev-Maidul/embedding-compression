import numpy as np

from src.eval.intrinsic_dim import k_nearest_distances, mle_id, twonn_id


def _embed_in_higher_dim(points: np.ndarray, ambient_dim: int, seed: int) -> np.ndarray:
    """Embed `points` (N, d) into `ambient_dim` dimensions via a random
    orthogonal rotation, so the data has a known intrinsic dimension d but
    lives in a higher-dimensional ambient space -- same setup used for the
    Day 5 synthetic validation in experiments/c_intrinsic_dim.py."""
    rng = np.random.default_rng(seed)
    d = points.shape[1]
    random_matrix = rng.standard_normal((ambient_dim, ambient_dim))
    q, _ = np.linalg.qr(random_matrix)
    padded = np.zeros((points.shape[0], ambient_dim))
    padded[:, :d] = points
    return padded @ q.T


def test_k_nearest_distances_shape_and_self_excluded():
    rng = np.random.default_rng(30)
    X = rng.standard_normal((50, 5))
    dist = k_nearest_distances(X, k=4)
    assert dist.shape == (50, 4)
    assert np.all(dist > 0)  # nothing is distance 0 from itself


def test_k_nearest_distances_sorted_ascending():
    rng = np.random.default_rng(31)
    X = rng.standard_normal((40, 3))
    dist = k_nearest_distances(X, k=6)
    assert np.all(np.diff(dist, axis=1) >= 0)


def test_twonn_recovers_known_linear_subspace_dimension():
    rng = np.random.default_rng(32)
    n, true_dim, ambient_dim = 2000, 5, 50
    points = rng.standard_normal((n, true_dim))
    X = _embed_in_higher_dim(points, ambient_dim, seed=1)
    estimate = twonn_id(X)
    assert abs(estimate - true_dim) < 1.0


def test_mle_recovers_known_linear_subspace_dimension():
    rng = np.random.default_rng(33)
    n, true_dim, ambient_dim = 2000, 5, 50
    points = rng.standard_normal((n, true_dim))
    X = _embed_in_higher_dim(points, ambient_dim, seed=2)
    estimate = mle_id(X, k_min=10, k_max=20)
    assert abs(estimate - true_dim) < 1.0


def test_twonn_recovers_1d_line():
    rng = np.random.default_rng(34)
    n = 1000
    points = rng.uniform(0, 1, size=(n, 1))
    X = _embed_in_higher_dim(points, ambient_dim=20, seed=3)
    estimate = twonn_id(X)
    assert 0.5 < estimate < 2.0


def test_mle_on_higher_dim_hypercube_is_larger_than_on_lower_dim():
    rng = np.random.default_rng(35)
    n = 1500
    low = rng.uniform(0, 1, size=(n, 3))
    high = rng.uniform(0, 1, size=(n, 15))
    X_low = _embed_in_higher_dim(low, ambient_dim=40, seed=4)
    X_high = _embed_in_higher_dim(high, ambient_dim=40, seed=4)
    est_low = mle_id(X_low)
    est_high = mle_id(X_high)
    assert est_high > est_low
