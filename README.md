# How much can image features be compressed?

_Created: November 2026_

A small, honest, reproducible study: from-scratch image/signal processing
(validated against OpenCV), classic and learned CIFAR-10 features, and one
research question about how much nearest-neighbour retrieval quality is
lost when features are compressed by random projection, and whether that
loss can be predicted from intrinsic dimension. This is a small empirical
study, not a state-of-the-art or novel method.

## Overview

The question: when an image feature vector is compressed from its full
dimension $d$ down to $m$ dimensions by a random projection, how much
nearest-neighbour retrieval quality is lost, and can that loss be
predicted ahead of time from the feature's intrinsic dimension (how many
real degrees of freedom the data varies along, as opposed to how many
numbers it happens to be stored with)? The study is split into three
parts: foundational image/signal processing implemented from scratch and
validated against OpenCV, three feature types extracted for CIFAR-10, and
the random-projection retrieval experiment itself together with intrinsic
dimension estimation.

## What is implemented

- **Part A -- image and signal processing, from scratch in NumPy**:
  2D convolution/correlation, Gaussian blur, Sobel gradients, Canny edge
  detection (non-maximum suppression and double-threshold hysteresis),
  histogram equalization, and FFT-domain low/high-pass filtering. Every
  function is validated against OpenCV (and, for the convolution theorem,
  against a direct FFT-domain computation) with measured error numbers in
  `results/a_validation.csv`, not just visual inspection.
- **Part B -- features on the CIFAR-10 test set (10,000 images)**: raw
  pixels (3072-d, globally normalized), HOG (324-d, scikit-image), and
  pretrained ResNet-18 penultimate-layer features (512-d). The ResNet-18
  input resolution (224px) was chosen after benchmarking 32/64/224px
  first, not assumed (`results/b_resnet_resolution_benchmark.csv`).
- **Part C -- random projection, retrieval, and intrinsic dimension**:
  three random projections (dense Gaussian, sparse Achlioptas, and a
  structured Subsampled Randomized Hadamard Transform), each measured for
  recall@10 against exact nearest-neighbour ground truth across a grid of
  projection dimensions and 10 seeds with bootstrap confidence intervals;
  and intrinsic dimension estimated with TwoNN and the Levina-Bickel MLE,
  both validated on synthetic data of known dimension before use on real
  features.

<!-- RESULTS_SUMMARY_START -->

## Results summary

_Regenerated from `results/*.csv` by `report/make_readme_summary.py` -- not hand-typed._

**Part A validation vs OpenCV** (max abs error across 4 validation images):
- `Canny_vs_canny_edge_detector`: max abs error 1.00e+00
- `GaussianBlur_vs_gaussian_blur`: max abs error 1.14e-13
- `Sobel_vs_sobel_gradients`: max abs error 0.00e+00
- `equalizeHist_vs_histogram_equalize`: max abs error 0.00e+00
- `filter2D_vs_correlate2d`: max abs error 1.71e-13

**Part B feature extraction** (all 10,000 images):
- pixels: shape (10000, 3072), 0.0s
- hog: shape (10000, 324), 1.2s
- resnet: shape (10000, 512), 213.4s
- ResNet-18 input resolution used: 224px (benchmarked 32/64/224px first, see `results/b_resnet_resolution_benchmark.csv`)

**Part C intrinsic dimension estimator validation** (synthetic data, known true dim):
- 5d_linear_subspace_in_100d: true=5, TwoNN=5.00, MLE=5.03
- 10d_hypercube_in_100d: true=10, TwoNN=8.75, MLE=8.38

**Part C intrinsic dimension by feature type** (full 10,000-image set):
- pixels (d=3072): TwoNN=31.63, MLE=25.11
- hog (d=324): TwoNN=39.95, MLE=36.09
- resnet (d=512): TwoNN=30.09, MLE=26.77

**Part C recall@10 at m=64** (mean, 95% bootstrap CI):
- hog / gaussian: 0.196 [0.195, 0.196]
- hog / sparse: 0.195 [0.194, 0.195]
- hog / srht: 0.210 [0.209, 0.211]
- pixels / gaussian: 0.378 [0.377, 0.380]
- pixels / sparse: 0.375 [0.374, 0.376]
- pixels / srht: 0.383 [0.381, 0.384]
- resnet / gaussian: 0.297 [0.296, 0.298]
- resnet / sparse: 0.300 [0.299, 0.301]
- resnet / srht: 0.322 [0.321, 0.323]

**Part C collapse-spread statistic** (lower = better collapse across feature types):
- m: 0.0425
- m_over_d: 0.1897
- m_over_id: 0.0305
- spearman_id_vs_recall_loss_at_m=64_n=3: 0.5000

<!-- RESULTS_SUMMARY_END -->

## Findings

**Recall vs. projection dimension.** At a representative projection
dimension of $m=64$, recall@10 ranges from about 0.20 (HOG) to about 0.38
(raw pixels), with ResNet features in between at about 0.30 (exact means
and 95% CIs for every (feature, projection) pair are in the Results
summary above and `results/c_recall.csv`). All three projection methods
(Gaussian, sparse, SRHT) give nearly identical recall for pixels and HOG.

**SRHT vs. Gaussian/sparse on ResNet features at large $m$.** For ResNet
features at $m=256$ (the largest projection dimension tested here, since
only $m<d$ was run), SRHT reaches recall@10 $\approx$ 0.685 versus
$\approx$ 0.582 for Gaussian/sparse projection -- a difference clearly
outside the reported confidence intervals. We do not have an established
explanation for this. One untested hypothesis: SRHT samples coordinates
without replacement from a fixed orthogonal transform, which could reduce
variance relative to Gaussian/sparse projection when $m/D$ is large; this
is speculation, not something this study tested or confirmed.

**Intrinsic dimension and the collapse hypothesis.** TwoNN-estimated
intrinsic dimension is 31.6 for pixels, 40.0 for HOG, and 30.1 for ResNet
features. The central hypothesis -- that recall@10 at fixed $m$ depends
mainly on intrinsic dimension, so that recall curves for different
feature types should collapse onto one curve when plotted against
$m/\hat{ID}$ rather than raw $m$ -- gets only weak, preliminary support
from this data. Pixels and ResNet features have almost the same
estimated intrinsic dimension (31.6 vs. 30.1) but noticeably different
recall at $m=64$ (0.378 vs. 0.297), which the hypothesis alone does not
explain. A pre-declared spread statistic is lower for $m/\hat{ID}$
(0.031) than for raw $m$ (0.043), but only slightly, and largely because
the three feature types' TwoNN estimates (31.6, 40.0, 30.1) are
themselves fairly close together -- so there is limited dynamic range for
$m/\hat{ID}$ to actually differ from $m$ across feature types. This
result rests on only three curves (one per feature type) with no
uncertainty estimate on the spread statistic itself, so it should be read
as a preliminary, exploratory observation, not a confirmed effect.

## Limitations

- Single dataset (CIFAR-10), a fixed 10,000-image subset, one random
  seed for data splits.
- CPU-only, which bounded some choices (e.g. ResNet-18 instead of a
  larger backbone) but was not otherwise a binding constraint -- every
  experiment here, including the slowest (about 18.5 minutes), finished
  well within an hour.
- The collapse analysis above uses only feature-type-level recall curves
  (three points), not separate recall sweeps for each of the 10 CIFAR-10
  classes, since the per-class version would cost roughly 10x the
  runtime of the feature-type-level sweep. Per-class intrinsic dimension
  estimates (30 (feature, class) combinations) were still computed in
  full and are in `results/c_intrinsic_dim.csv`.
- The intrinsic-dimension/collapse hypothesis is exploratory and
  received only weak, preliminary support (see Findings above), not
  confirmation. The accompanying Spearman correlation check uses only
  $n=3$ data points and is not statistically meaningful at that sample
  size.
- Canny does not pixel-match OpenCV's `cv2.Canny` exactly, by design (a
  different, also-standard non-maximum-suppression and threshold
  convention); this is a measured, documented difference
  (`results/a_validation.csv`), not an unverified assumption.
- The 10-d hypercube intrinsic-dimension validation case underestimates
  the true dimension (TwoNN 8.75, MLE 8.38, vs. true 10), a known
  boundary-bias effect for bounded distributions documented in
  `docs/DECISIONS.md`, not a bug in the estimators.

## Reproducing

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` recreates the venv, downloads CIFAR-10 if needed, runs all
tests, and regenerates every CSV, figure, and report table in `results/`
and `report/` from scratch. Nothing in this README or the report is typed
by hand; every number is produced by a script reading from `results/`.

Each stage can also be run individually (all assume
`source venv/bin/activate` and running from the repo root):

```bash
python experiments/make_environment_txt.py        # results/environment.txt
python experiments/a_validate_ip.py                # results/a_validation.csv
python experiments/a_fourier_timing.py              # results/a_fourier_timing.csv + figure
python experiments/a_figures.py                     # Part A figures
python experiments/b_benchmark_resnet.py             # results/b_resnet_resolution_benchmark.csv
python experiments/b_extract_features.py             # data/features/*.npy + results/b_feature_extraction.csv
python experiments/c_projection_recall.py            # results/c_recall.csv (~15-20 min, logs progress)
python experiments/c_recall_plots.py                 # recall-vs-m figures
python experiments/c_intrinsic_dim_validation.py      # results/c_id_validation.csv
python experiments/c_intrinsic_dim.py                 # results/c_intrinsic_dim.csv
python experiments/c_collapse_analysis.py             # results/c_collapse.csv + collapse figures
python report/make_tables.py                          # report/tables/*.tex
python report/make_readme_summary.py                  # regenerates the section above
cd report && pdflatex report.tex && pdflatex report.tex && cd ..
pytest tests/ -q                                      # all unit tests (75 passing)
```

Seed 42 is used everywhere a seed applies. Machine and package versions
are recorded in `results/environment.txt` (generated by
`experiments/make_environment_txt.py`); all experiments were run
CPU-only.

## Repository layout

```
src/
  ip/          convolution, Gaussian, Sobel, Canny, histogram eq, Fourier
  features/    pixel, HOG, ResNet-18 feature extraction, CIFAR-10 loading
  projection/  Gaussian, sparse (Achlioptas), and SRHT random projections
  eval/        exact kNN recall, bootstrap CI, intrinsic dimension (TwoNN/MLE)
experiments/   one script per experiment, producing files in results/
tests/         pytest unit tests for every src/ module
results/       CSVs, figures, environment.txt (committed; regenerated by run_all.sh)
report/        report.tex, make_tables.py, make_readme_summary.py, tables/
docs/          DECISIONS.md (every parameter choice and why), EXPLAINERS.md
configs/       default.yaml (seeds, grids, paths)
data/          CIFAR-10 and cached features (gitignored, regenerated locally)
```

## Development note

Parts of the code and documentation were written with an AI coding
assistant (Claude Code). The author defined the study, reviewed the
code, ran the experiments and is responsible for the results.
