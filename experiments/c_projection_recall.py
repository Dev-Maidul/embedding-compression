"""Part C: how much does projecting features to m dimensions hurt
nearest-neighbour retrieval?

For each feature type (pixels, HOG, resnet) and each projection method
(gaussian, sparse, srht), and each projection dimension m < d, we run 10
projection seeds, measure recall@10 against the exact (unprojected)
ground truth for every one of the 10,000 queries, pool all
10 * 10,000 per-query recall values, and report the mean plus a 95%
bootstrap CI over that pool.

This is the slow experiment the project brief asks to flag before running and to
log progress for: estimated ~15 minutes on this machine (measured on a
subset before committing to the full run; see docs/DECISIONS.md). Writes
results/c_recall.csv and logs progress to results/c_recall_progress.log.
"""

import itertools
import time

import numpy as np
import pandas as pd

from src.eval.bootstrap import bootstrap_mean_ci
from src.eval.knn_recall import exact_topk, recall_at_k
from src.projection import gaussian, sparse, srht

SEED = 42
K = 10
N_PROJECTION_SEEDS = 10
N_BOOTSTRAP_RESAMPLES = 1000
PROJECTION_DIMS = [4, 8, 16, 32, 64, 128, 256]

FEATURE_FILES = {
    "pixels": "data/features/pixels.npy",
    "hog": "data/features/hog.npy",
    "resnet": "data/features/resnet.npy",
}

PROJECTIONS = {
    "gaussian": gaussian.project,
    "sparse": sparse.project,
    "srht": srht.project,
}

LOG_PATH = "results/c_recall_progress.log"


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    np.random.seed(SEED)
    open(LOG_PATH, "w").close()  # truncate
    t_start = time.perf_counter()

    rows = []
    for feature_name, path in FEATURE_FILES.items():
        X = np.load(path)
        d = X.shape[1]
        log(f"=== feature={feature_name}, d={d}, N={X.shape[0]} ===")

        t0 = time.perf_counter()
        ground_truth = exact_topk(X, k=K)
        log(f"  ground truth top-{K} computed in {time.perf_counter() - t0:.1f}s")

        valid_dims = [m for m in PROJECTION_DIMS if m < d]

        for projection_name, project_fn in PROJECTIONS.items():
            for m in valid_dims:
                t0 = time.perf_counter()
                pooled_recalls = []
                for seed in range(N_PROJECTION_SEEDS):
                    X_proj = project_fn(X, m=m, seed=seed)
                    approx_neighbours = exact_topk(X_proj, k=K)
                    recalls = recall_at_k(ground_truth, approx_neighbours)
                    pooled_recalls.append(recalls)
                pooled_recalls = np.concatenate(pooled_recalls)

                mean, lo, hi = bootstrap_mean_ci(
                    pooled_recalls, n_resamples=N_BOOTSTRAP_RESAMPLES, seed=SEED
                )

                rows.append({
                    "feature": feature_name,
                    "d": d,
                    "projection": projection_name,
                    "m": m,
                    "m_over_d": m / d,
                    "n_seeds": N_PROJECTION_SEEDS,
                    "n_queries": X.shape[0],
                    "recall_at_10_mean": mean,
                    "recall_at_10_ci_lower": lo,
                    "recall_at_10_ci_upper": hi,
                })
                elapsed = time.perf_counter() - t0
                log(f"  {projection_name:8s} m={m:4d} ({m}/{d}): "
                    f"recall@10={mean:.3f} [{lo:.3f}, {hi:.3f}]  ({elapsed:.1f}s, {N_PROJECTION_SEEDS} seeds)")

    df = pd.DataFrame(rows)
    df.to_csv("results/c_recall.csv", index=False)
    total_elapsed = time.perf_counter() - t_start
    log(f"Wrote results/c_recall.csv ({len(df)} rows) in {total_elapsed / 60:.1f} minutes total")


if __name__ == "__main__":
    main()
