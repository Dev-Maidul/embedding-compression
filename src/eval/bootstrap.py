"""Bootstrap confidence intervals over queries.

The project brief requires never reporting a single number without uncertainty:
every recall@10 value is a mean plus a 95% bootstrap confidence interval,
computed by resampling queries with replacement many times and looking at
the spread of the resulting means.
"""

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray, n_resamples: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """values: 1D array of per-query measurements (e.g. recall@10 for
    each query). Returns (mean, ci_lower, ci_upper).

    Method: draw `n_resamples` bootstrap samples, each the same size as
    `values`, sampled with replacement; the mean of each resample gives an
    empirical distribution of the mean statistic; the (2.5, 97.5)
    percentiles of that distribution are the 95% CI bounds (the standard
    percentile bootstrap).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    resample_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = values[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(resample_means, 100 * alpha))
    upper = float(np.percentile(resample_means, 100 * (1 - alpha)))
    return float(values.mean()), lower, upper
