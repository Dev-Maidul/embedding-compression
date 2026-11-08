"""Generate every LaTeX table in the report from the CSVs in results/.

The project brief's rule #1: no number in the report is typed by hand. This
script is the only place table numbers are formatted; report.tex only
\\input's the .tex fragments this script writes to report/tables/.
Re-run this (and then pdflatex) any time an upstream results/*.csv
changes.
"""

import os

import pandas as pd

OUT_DIR = "report/tables"


def write_table(filename: str, latex: str):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        f.write(latex)
    print(f"Wrote {path}")


def table_a_validation():
    df = pd.read_csv("results/a_validation.csv")
    summary = df.groupby("check").agg(
        max_abs_error=("max_abs_error", "max"),
        mean_abs_error=("mean_abs_error", "mean"),
        pixel_agreement=("pixel_agreement", "mean"),
    ).reset_index()

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Check (ours vs OpenCV) & Max abs error & Mean abs error & Pixel agreement \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        agreement = f"{row['pixel_agreement']:.3f}" if pd.notna(row["pixel_agreement"]) else "--"
        lines.append(
            f"{row['check'].replace('_', r'\_')} & "
            f"{row['max_abs_error']:.2e} & {row['mean_abs_error']:.2e} & {agreement} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("a_validation.tex", "\n".join(lines) + "\n")


def table_b_features():
    extraction = pd.read_csv("results/b_feature_extraction.csv")
    benchmark = pd.read_csv("results/b_resnet_resolution_benchmark.csv")

    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Feature & Shape & Extraction time (10{,}000 images) \\",
        r"\midrule",
    ]
    for _, row in extraction.iterrows():
        lines.append(f"{row['feature']} & {row['shape']} & {row['elapsed_sec']:.1f}s \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("b_features.tex", "\n".join(lines) + "\n")

    lines2 = [
        r"\begin{tabular}{rr}",
        r"\toprule",
        r"ResNet-18 input resolution & Estimated time, 10{,}000 images \\",
        r"\midrule",
    ]
    for _, row in benchmark.iterrows():
        lines2.append(f"{int(row['resolution'])}px & {row['estimated_min_for_10000_images']:.1f} min \\\\")
    lines2 += [r"\bottomrule", r"\end{tabular}"]
    write_table("b_resnet_benchmark.tex", "\n".join(lines2) + "\n")


def table_c_id_validation():
    df = pd.read_csv("results/c_id_validation.csv")
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Synthetic case & True dim. & TwoNN estimate & MLE estimate \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        case = row["case"].replace("_", " ")
        lines.append(f"{case} & {int(row['true_dim'])} & {row['twonn_estimate']:.2f} & {row['mle_estimate']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("c_id_validation.tex", "\n".join(lines) + "\n")


def table_c_intrinsic_dim():
    df = pd.read_csv("results/c_intrinsic_dim.csv")
    overall = df[df["class"] == "all"]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Feature & $d$ & TwoNN $\hat{ID}$ & MLE $\hat{ID}$ \\",
        r"\midrule",
    ]
    for _, row in overall.iterrows():
        lines.append(f"{row['feature']} & {int(row['d'])} & {row['twonn_id']:.2f} & {row['mle_id']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("c_intrinsic_dim_overall.tex", "\n".join(lines) + "\n")

    per_class = df[df["class"] != "all"]
    lines2 = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Feature & Class & TwoNN $\hat{ID}$ & MLE $\hat{ID}$ \\",
        r"\midrule",
    ]
    for _, row in per_class.iterrows():
        lines2.append(f"{row['feature']} & {row['class']} & {row['twonn_id']:.2f} & {row['mle_id']:.2f} \\\\")
    lines2 += [r"\bottomrule", r"\end{tabular}"]
    write_table("c_intrinsic_dim_per_class.tex", "\n".join(lines2) + "\n")


def table_c_recall_at_m():
    df = pd.read_csv("results/c_recall.csv")
    fixed_m = 64
    subset = df[df["m"] == fixed_m].sort_values(["feature", "projection"])
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        rf"Feature & Projection & recall@10 (m={fixed_m}) & CI lower & CI upper \\",
        r"\midrule",
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"{row['feature']} & {row['projection']} & {row['recall_at_10_mean']:.3f} & "
            f"{row['recall_at_10_ci_lower']:.3f} & {row['recall_at_10_ci_upper']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("c_recall_at_m64.tex", "\n".join(lines) + "\n")


def table_c_collapse():
    df = pd.read_csv("results/c_collapse.csv")
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Comparison & Value \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        label = row["x_axis"].replace("_", r"\_")
        lines.append(f"{label} & {row['collapse_spread_statistic']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("c_collapse.tex", "\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    table_a_validation()
    table_b_features()
    table_c_id_validation()
    table_c_intrinsic_dim()
    table_c_recall_at_m()
    table_c_collapse()


if __name__ == "__main__":
    main()
