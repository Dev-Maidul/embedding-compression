"""Part C's central hypothesis test: does recall@10 at a fixed projection
dimension m depend mainly on intrinsic dimension (ID), such that recall
curves for different feature types collapse onto a single curve when
plotted against m/ID_hat (rather than against raw m or m/d)?

This is explicitly exploratory (the project brief) -- it may fail, and a clean
"it does not hold" is a valid, reportable finding, not something to hide
or paper over.

Writes results/c_collapse.csv and three comparison figures:
results/figures/c_collapse_vs_m.png
results/figures/c_collapse_vs_m_over_d.png
results/figures/c_collapse_vs_m_over_id.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ID_ESTIMATOR = "twonn_id"  # see docs/DECISIONS.md for why TwoNN, not MLE
FIXED_M_FOR_CORRELATION = 64


def load_feature_curves():
    """Returns {feature: (d, id_hat, m_array, recall_array)}, where
    recall_array is the recall@10 mean AVERAGED OVER THE 3 PROJECTION
    METHODS at each m (the collapse hypothesis is about feature types,
    not about which projection was used, so we average that axis out;
    see docs/DECISIONS.md)."""
    recall_df = pd.read_csv("results/c_recall.csv")
    id_df = pd.read_csv("results/c_intrinsic_dim.csv")
    id_df_all = id_df[id_df["class"] == "all"].set_index("feature")

    curves = {}
    for feature, feature_df in recall_df.groupby("feature"):
        avg_over_projections = (
            feature_df.groupby("m")["recall_at_10_mean"].mean().sort_index()
        )
        d = int(feature_df["d"].iloc[0])
        id_hat = float(id_df_all.loc[feature, ID_ESTIMATOR])
        curves[feature] = (d, id_hat, avg_over_projections.index.values.astype(float),
                            avg_over_projections.values)
    return curves


def collapse_spread_statistic(curves: dict, x_transform) -> float:
    """Pre-declared collapse statistic: map each feature's m values
    through x_transform(feature_data) to get its x-axis values, build a
    common log-spaced grid spanning the OVERLAP of all curves' x-ranges,
    linearly interpolate each curve (in log-x) onto that grid, then
    average the across-feature standard deviation at each grid point.
    Lower = curves collapse onto each other more tightly; 0 would be a
    perfect collapse.
    """
    xs_by_feature = {}
    for feature, (d, id_hat, m_arr, recall_arr) in curves.items():
        xs_by_feature[feature] = (x_transform(m_arr, d, id_hat), recall_arr)

    lo = max(x.min() for x, _ in xs_by_feature.values())
    hi = min(x.max() for x, _ in xs_by_feature.values())
    if lo >= hi:
        return float("nan")  # no overlapping range -- cannot compare curves at all

    grid = np.logspace(np.log10(lo), np.log10(hi), num=20)

    interpolated = []
    for feature, (x, y) in xs_by_feature.items():
        order = np.argsort(x)
        interpolated.append(np.interp(grid, x[order], y[order]))
    interpolated = np.stack(interpolated, axis=0)  # (n_features, n_grid)

    spread_per_point = np.std(interpolated, axis=0)
    return float(np.mean(spread_per_point))


def plot_curves(curves: dict, x_transform, xlabel: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for feature, (d, id_hat, m_arr, recall_arr) in curves.items():
        x = x_transform(m_arr, d, id_hat)
        order = np.argsort(x)
        ax.plot(x[order], recall_arr[order], marker="o",
                label=f"{feature} (d={d}, ID≈{id_hat:.0f})")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("recall@10 (mean over 3 projections)")
    ax.set_title(f"Collapse check: recall@10 vs {xlabel}")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    curves = load_feature_curves()

    transforms = {
        "m": (lambda m, d, id_hat: m, "projection dimension m"),
        "m_over_d": (lambda m, d, id_hat: m / d, "m / d"),
        "m_over_id": (lambda m, d, id_hat: m / id_hat, "m / ID_hat (TwoNN)"),
    }

    rows = []
    for name, (transform, xlabel) in transforms.items():
        stat = collapse_spread_statistic(curves, transform)
        rows.append({"x_axis": name, "collapse_spread_statistic": stat})
        plot_curves(curves, transform, xlabel, f"results/figures/c_collapse_vs_{name}.png")
        print(f"{name}: collapse spread statistic = {stat:.4f} (lower = more collapse)")

    # Spearman correlation between ID_hat and recall-loss at a fixed m,
    # across the 3 feature types. NOTE: n=3 data points is very small for
    # a correlation -- reported honestly as a weak/suggestive signal only,
    # not a statistically meaningful test (see docs/DECISIONS.md).
    id_hats, recall_losses = [], []
    for feature, (d, id_hat, m_arr, recall_arr) in curves.items():
        if FIXED_M_FOR_CORRELATION in m_arr:
            recall_at_m = recall_arr[m_arr == FIXED_M_FOR_CORRELATION][0]
            id_hats.append(id_hat)
            recall_losses.append(1.0 - recall_at_m)

    if len(id_hats) >= 3:
        corr, p_value = spearmanr(id_hats, recall_losses)
    else:
        corr, p_value = float("nan"), float("nan")

    rows.append({
        "x_axis": f"spearman_id_vs_recall_loss_at_m={FIXED_M_FOR_CORRELATION}_n={len(id_hats)}",
        "collapse_spread_statistic": corr,
    })
    print(f"\nSpearman(ID_hat, recall_loss @ m={FIXED_M_FOR_CORRELATION}) = {corr:.3f} "
          f"(p={p_value:.3f}, n={len(id_hats)} -- very small n, interpret with caution)")

    df = pd.DataFrame(rows)
    df.to_csv("results/c_collapse.csv", index=False)
    print("\nWrote results/c_collapse.csv")


if __name__ == "__main__":
    main()
