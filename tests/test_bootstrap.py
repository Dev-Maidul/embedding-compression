import numpy as np

from src.eval.bootstrap import bootstrap_mean_ci


def test_bootstrap_mean_matches_simple_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lower, upper = bootstrap_mean_ci(values, n_resamples=500, seed=0)
    assert np.isclose(mean, values.mean())


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(23)
    values = rng.normal(0.7, 0.1, size=200)
    mean, lower, upper = bootstrap_mean_ci(values, n_resamples=1000, seed=1)
    assert lower <= mean <= upper


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(24)
    small = rng.normal(0.5, 0.2, size=20)
    large = rng.normal(0.5, 0.2, size=2000)
    _, lo_small, hi_small = bootstrap_mean_ci(small, n_resamples=1000, seed=2)
    _, lo_large, hi_large = bootstrap_mean_ci(large, n_resamples=1000, seed=2)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_bootstrap_zero_variance_gives_zero_width_ci():
    values = np.full(50, 0.42)
    mean, lower, upper = bootstrap_mean_ci(values, n_resamples=200, seed=3)
    assert np.isclose(mean, 0.42)
    assert np.isclose(lower, 0.42)
    assert np.isclose(upper, 0.42)
