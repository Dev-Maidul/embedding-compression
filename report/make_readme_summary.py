"""Regenerate the README's "Results summary" section from results/*.csv.

Writes only the text between the <!-- RESULTS_SUMMARY_START --> and
<!-- RESULTS_SUMMARY_END --> markers in README.md; everything else in the
README is left untouched. This is what makes the README's numbers
generated, not hand-typed, per the project brief's rule #1.
"""

import pandas as pd

START_MARKER = "<!-- RESULTS_SUMMARY_START -->"
END_MARKER = "<!-- RESULTS_SUMMARY_END -->"


def build_summary() -> str:
    a = pd.read_csv("results/a_validation.csv")
    b_feat = pd.read_csv("results/b_feature_extraction.csv")
    b_bench = pd.read_csv("results/b_resnet_resolution_benchmark.csv")
    c_recall = pd.read_csv("results/c_recall.csv")
    c_id = pd.read_csv("results/c_intrinsic_dim.csv")
    c_collapse = pd.read_csv("results/c_collapse.csv")
    c_id_val = pd.read_csv("results/c_id_validation.csv")

    lines = [START_MARKER, "", "## Results summary", "",
             "_Regenerated from `results/*.csv` by `report/make_readme_summary.py` -- not hand-typed._", ""]

    lines.append("**Part A validation vs OpenCV** (max abs error across 4 validation images):")
    for check, group in a.groupby("check"):
        lines.append(f"- `{check}`: max abs error {group['max_abs_error'].max():.2e}")
    lines.append("")

    lines.append("**Part B feature extraction** (all 10,000 images):")
    for _, row in b_feat.iterrows():
        lines.append(f"- {row['feature']}: shape {row['shape']}, {row['elapsed_sec']:.1f}s")
    best_res = int(b_bench.loc[b_bench["resolution"] == 224, "resolution"].iloc[0])
    lines.append(f"- ResNet-18 input resolution used: {best_res}px "
                  f"(benchmarked 32/64/224px first, see `results/b_resnet_resolution_benchmark.csv`)")
    lines.append("")

    lines.append("**Part C intrinsic dimension estimator validation** (synthetic data, known true dim):")
    for _, row in c_id_val.iterrows():
        lines.append(f"- {row['case']}: true={int(row['true_dim'])}, "
                      f"TwoNN={row['twonn_estimate']:.2f}, MLE={row['mle_estimate']:.2f}")
    lines.append("")

    lines.append("**Part C intrinsic dimension by feature type** (full 10,000-image set):")
    overall = c_id[c_id["class"] == "all"]
    for _, row in overall.iterrows():
        lines.append(f"- {row['feature']} (d={int(row['d'])}): "
                      f"TwoNN={row['twonn_id']:.2f}, MLE={row['mle_id']:.2f}")
    lines.append("")

    lines.append("**Part C recall@10 at m=64** (mean, 95% bootstrap CI):")
    subset = c_recall[c_recall["m"] == 64].sort_values(["feature", "projection"])
    for _, row in subset.iterrows():
        lines.append(f"- {row['feature']} / {row['projection']}: "
                      f"{row['recall_at_10_mean']:.3f} "
                      f"[{row['recall_at_10_ci_lower']:.3f}, {row['recall_at_10_ci_upper']:.3f}]")
    lines.append("")

    lines.append("**Part C collapse-spread statistic** (lower = better collapse across feature types):")
    for _, row in c_collapse.iterrows():
        lines.append(f"- {row['x_axis']}: {row['collapse_spread_statistic']:.4f}")
    lines.append("")

    lines.append(END_MARKER)
    return "\n".join(lines)


def main():
    with open("README.md") as f:
        content = f.read()

    if START_MARKER not in content or END_MARKER not in content:
        raise RuntimeError(
            f"README.md must contain both {START_MARKER} and {END_MARKER} markers "
            "for this script to know where to write the summary."
        )

    before = content.split(START_MARKER)[0]
    after = content.split(END_MARKER)[1]
    new_content = before + build_summary() + after

    with open("README.md", "w") as f:
        f.write(new_content)
    print("Updated README.md results summary section.")


if __name__ == "__main__":
    main()
