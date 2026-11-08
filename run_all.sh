#!/usr/bin/env bash
# Runs the whole project from a clean clone, in order. Extended each day
# as more experiments are added. Safe to re-run: it recreates the venv and
# regenerates every result file.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt

export PYTHONPATH=.

echo "== tests =="
pytest tests/ -q

echo "== environment info =="
python experiments/make_environment_txt.py

echo "== Part A: validate image/signal processing against OpenCV =="
python experiments/a_validate_ip.py

echo "== Part A: convolution theorem timing demo =="
python experiments/a_fourier_timing.py

echo "== Part A: figures =="
mkdir -p results/figures
python experiments/a_figures.py

echo "== Part B: ResNet resolution benchmark =="
python experiments/b_benchmark_resnet.py

echo "== Part B: extract and cache pixel/HOG/ResNet features =="
python experiments/b_extract_features.py

echo "== Part C: projection + recall sweep (long, ~15-20 min, logs to results/c_recall_progress.log) =="
python experiments/c_projection_recall.py

echo "== Part C: recall-vs-m plots =="
python experiments/c_recall_plots.py

echo "== Part C: intrinsic dimension estimator validation on synthetic data =="
python experiments/c_intrinsic_dim_validation.py

echo "== Part C: intrinsic dimension per feature type and per class =="
python experiments/c_intrinsic_dim.py

echo "== Part C: collapse analysis (m vs m/d vs m/ID_hat) =="
python experiments/c_collapse_analysis.py

echo "== Report: generate tables and README summary from CSVs =="
python report/make_tables.py
python report/make_readme_summary.py

echo "== Report: compile report.tex (twice, for cross-references) =="
if command -v pdflatex >/dev/null 2>&1; then
  ( cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex )
else
  echo "pdflatex not found -- skipping PDF compile. Install a LaTeX distribution (e.g. 'brew install --cask basictex') to compile report/report.tex yourself."
fi

echo "Day 1-6 steps complete."
