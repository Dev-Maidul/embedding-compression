import numpy as np

from src.eval.knn_recall import exact_topk, pairwise_squared_distances, recall_at_k


def test_pairwise_squared_distances_matches_brute_force():
    rng = np.random.default_rng(20)
    X = rng.standard_normal((15, 6))
    fast = pairwise_squared_distances(X)

    n = X.shape[0]
    brute = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            brute[i, j] = np.sum((X[i] - X[j]) ** 2)

    assert np.allclose(fast, brute, atol=1e-8)


def test_exact_topk_excludes_self():
    rng = np.random.default_rng(21)
    X = rng.standard_normal((20, 5))
    neighbours = exact_topk(X, k=5)
    for i in range(20):
        assert i not in neighbours[i]


def test_exact_topk_shape_and_ordering():
    rng = np.random.default_rng(22)
    X = rng.standard_normal((30, 4))
    k = 10
    neighbours = exact_topk(X, k=k)
    assert neighbours.shape == (30, k)

    dist = pairwise_squared_distances(X)
    for i in range(30):
        row_dists = dist[i, neighbours[i]]
        assert np.all(np.diff(row_dists) >= -1e-9)  # nearest-first, non-decreasing


def test_exact_topk_finds_known_nearest_neighbour():
    # point 0 and point 1 are very close; point 2 is far from both.
    X = np.array([
        [0.0, 0.0],
        [0.01, 0.0],
        [100.0, 100.0],
        [-50.0, -50.0],
    ])
    neighbours = exact_topk(X, k=1)
    assert neighbours[0, 0] == 1
    assert neighbours[1, 0] == 0


def test_recall_at_k_perfect_match():
    true_nn = np.array([[1, 2, 3], [0, 2, 3]])
    approx_nn = np.array([[1, 2, 3], [0, 2, 3]])
    recalls = recall_at_k(true_nn, approx_nn)
    assert np.allclose(recalls, 1.0)


def test_recall_at_k_no_overlap():
    true_nn = np.array([[1, 2, 3]])
    approx_nn = np.array([[4, 5, 6]])
    recalls = recall_at_k(true_nn, approx_nn)
    assert np.allclose(recalls, 0.0)


def test_recall_at_k_partial_overlap():
    true_nn = np.array([[1, 2, 3, 4]])
    approx_nn = np.array([[1, 2, 9, 9]])
    recalls = recall_at_k(true_nn, approx_nn)
    assert np.allclose(recalls, 0.5)
