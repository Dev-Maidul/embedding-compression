import numpy as np
import pytest

from src.projection import gaussian, sparse, srht

PROJECTIONS = {
    "gaussian": gaussian.project,
    "sparse": sparse.project,
    "srht": srht.project,
}


@pytest.mark.parametrize("name", PROJECTIONS.keys())
def test_projection_output_shape(name):
    rng = np.random.default_rng(100)
    X = rng.standard_normal((20, 50))
    out = PROJECTIONS[name](X, m=10, seed=0)
    assert out.shape == (20, 10)


@pytest.mark.parametrize("name", PROJECTIONS.keys())
def test_projection_preserves_norm_in_expectation(name):
    """E[||Rx||^2] should be close to ||x||^2, averaged over many random
    projection seeds (the Johnson-Lindenstrauss / Achlioptas guarantee is
    about expectation and concentration, not an exact per-draw equality)."""
    rng = np.random.default_rng(101)
    d, m, n_seeds = 200, 64, 40
    x = rng.standard_normal((1, d))
    original_sqnorm = float(np.sum(x ** 2))

    ratios = []
    for seed in range(n_seeds):
        projected = PROJECTIONS[name](x, m=m, seed=seed)
        ratios.append(float(np.sum(projected ** 2)) / original_sqnorm)

    mean_ratio = np.mean(ratios)
    assert 0.85 < mean_ratio < 1.15, f"{name}: mean squared-norm ratio {mean_ratio} far from 1"


@pytest.mark.parametrize("name", PROJECTIONS.keys())
def test_projection_preserves_pairwise_distance_in_expectation(name):
    rng = np.random.default_rng(102)
    d, m, n_seeds = 300, 100, 30
    x1 = rng.standard_normal((1, d))
    x2 = rng.standard_normal((1, d))
    original_sqdist = float(np.sum((x1 - x2) ** 2))

    ratios = []
    for seed in range(n_seeds):
        p1 = PROJECTIONS[name](x1, m=m, seed=seed)
        p2 = PROJECTIONS[name](x2, m=m, seed=seed)
        projected_sqdist = float(np.sum((p1 - p2) ** 2))
        ratios.append(projected_sqdist / original_sqdist)

    mean_ratio = np.mean(ratios)
    assert 0.8 < mean_ratio < 1.2, f"{name}: mean squared-distance ratio {mean_ratio} far from 1"


@pytest.mark.parametrize("name", PROJECTIONS.keys())
def test_projection_is_deterministic_given_seed(name):
    rng = np.random.default_rng(103)
    X = rng.standard_normal((5, 40))
    out_a = PROJECTIONS[name](X, m=8, seed=7)
    out_b = PROJECTIONS[name](X, m=8, seed=7)
    assert np.array_equal(out_a, out_b)


def test_srht_pads_to_power_of_two_without_error():
    rng = np.random.default_rng(104)
    X = rng.standard_normal((5, 100))  # 100 is not a power of two
    out = srht.project(X, m=16, seed=0)
    assert out.shape == (5, 16)


def test_sparse_matrix_is_mostly_zero():
    # indirect check of the Achlioptas sparsity: with p(zero) = 2/3, the
    # fraction of near-zero projected-coordinate contributions should
    # reflect that most entries of the underlying matrix are exactly 0.
    # We check this by reconstructing the implied matrix for a tiny case.
    rng = np.random.default_rng(0)
    identity_like = np.eye(200)
    R_times_I = sparse.project(identity_like, m=50, seed=42)  # = R itself (up to column order)
    nonzero_fraction = np.mean(R_times_I != 0)
    assert 0.25 < nonzero_fraction < 0.40  # expect close to 1/3 nonzero
