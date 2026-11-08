"""One recall-vs-m plot per feature type, with all three projection
methods' curves and their 95% bootstrap CI bands, read from
results/c_recall.csv (which must already exist -- run
c_projection_recall.py first)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECTION_COLORS = {
    "gaussian": "#1f77b4",
    "sparse": "#ff7f0e",
    "srht": "#2ca02c",
}


def main():
    df = pd.read_csv("results/c_recall.csv")

    for feature_name, feature_df in df.groupby("feature"):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for projection_name, proj_df in feature_df.groupby("projection"):
            proj_df = proj_df.sort_values("m")
            color = PROJECTION_COLORS[projection_name]
            ax.plot(proj_df["m"], proj_df["recall_at_10_mean"], marker="o",
                    label=projection_name, color=color)
            ax.fill_between(proj_df["m"], proj_df["recall_at_10_ci_lower"],
                             proj_df["recall_at_10_ci_upper"], alpha=0.2, color=color)

        d = feature_df["d"].iloc[0]
        ax.set_xscale("log", base=2)
        ax.set_xlabel("projection dimension m")
        ax.set_ylabel("recall@10 (mean, 95% bootstrap CI)")
        ax.set_title(f"{feature_name} (d={d}): recall@10 vs projection dimension")
        ax.set_ylim(0, 1.02)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = f"results/figures/c_recall_vs_m_{feature_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
