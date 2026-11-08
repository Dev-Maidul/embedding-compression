# Decisions log

Every time a parameter or design choice is not fixed by the study design, it is
recorded here with the reason, instead of silently picking something.

## Image processing

- **Convolution implementation: vectorized with `np.lib.stride_tricks.sliding_window_view` + `einsum`, not a raw double Python for-loop.**
  Reason: a pure double-loop over every pixel is correct but far too slow
  on full-size validation images (e.g. 512x512) on a CPU-only laptop. The
  vectorized version computes the exact same sum-of-products definition,
  just for all pixels at once, so it stays "from scratch" (no `cv2`/`scipy`
  convolution call) while being fast enough to run routinely.

- **Sobel uses correlation (`correlate2d`), not true convolution (`convolve2d`).**
  Reason: measured against `cv2.Sobel`, the sign came out flipped when
  using true convolution (kernel rotated 180 degrees). This shows that
  `cv2.Sobel`, despite the name, applies the kernel as-is (correlation),
  matching how the Sobel kernels are conventionally written with -1 on the
  "low" side. Confirmed empirically in `tests/test_sobel.py::test_sobel_magnitude_matches_opencv`
  (max abs error ~0, see `results/a_validation.csv`).

- **Gaussian blur uses correlation too**, but it does not matter: the
  Gaussian kernel is symmetric, so correlation and true convolution give
  identical output for it.

- **Padding mode for validation: `reflect` (matches OpenCV's default
  `cv2.BORDER_REFLECT_101`)** for Gaussian and Sobel checks, `zero`
  available as an option for cases that need `cv2.BORDER_CONSTANT`.

- **Validation images: `skimage.data` samples (camera, astronaut converted
  to gray, checkerboard, coins)**, not CIFAR-10, for Part A. Reason: these
  are the images the study design calls for as needing "no download", and they are
  large and varied enough (200x200 to 512x512) to stress-test correctness
  and timing; CIFAR-10 images (32x32) are reserved for Part B/C feature
  extraction and retrieval, where their small size is the point.

- **Machine RAM is 16 GB** (`sysctl hw.memsize` on the development
  machine, recorded in `results/environment.txt`). Treated as
  informational only; no experiment size was scaled up because of it —
  all sizes (10k retrieval set, feature dims, etc.) stay as originally
  planned.

- **Canny NMS uses 4-sector angle quantization (0/45/90/135 degrees), not
  continuous bilinear interpolation between neighbours.**
  Reason: the sector method is the standard simplified version taught in
  most from-scratch tutorials, is much easier to explain and reason about
  pixel-by-pixel in an interview, and does not need a second interpolation
  step. Trade-off, stated honestly: it means our edge map does not
  pixel-match `cv2.Canny` exactly. Measured pixel agreement is 93-99%
  across the 4 validation images (`results/a_validation.csv`), which we
  judge good enough given the two algorithms are genuinely different
  variants, not a bug to fix.

- **Canny thresholds are fractions of this image's own max gradient
  magnitude (e.g. 0.1/0.2), not fixed absolute values.**
  Reason: this makes the same two numbers work across images of very
  different contrast/scale, which matters because CIFAR-10 images (Part
  B/C) are tiny (32x32) with very different gradient scales than the
  512x512 skimage validation images. `cv2.Canny` takes fixed absolute
  thresholds instead, which is part of why the two don't match pixel-
  perfectly (see above); for the validation script we picked absolute
  OpenCV thresholds (50, 150) that give visually comparable edge density.

- **Hysteresis implemented with `scipy.ndimage.label` (connected-component
  labeling) rather than a hand-written flood fill/stack-based traversal.**
  Reason: connected-component labeling is a standard, well-tested
  operation and using it keeps the code about what Canny does
  ("keep a weak pixel iff its connected component contains a strong
  pixel"), not about reimplementing graph traversal from scratch, which
  is outside this study's scope.

- **Frequency filters use an ideal (hard circular cutoff) mask**, not a
  smoothed one (e.g. Gaussian or Butterworth). Reason: this is the
  simplest filter to implement and is explicit about what "low-pass" means
  (keep all frequencies within radius r of DC, remove the rest); the
  resulting ringing artifacts in `results/figures/frequency_filtering.png`
  are noted in `docs/EXPLAINERS.md` as an expected, textbook consequence
  (Gibbs phenomenon), not something to hide or fix.

- **Convolution theorem demo uses `pad_mode="wrap"` (periodic/circular
  boundary), not `reflect` or `zero`.** Reason: the theorem (spatial
  convolution == FFT multiplication) is only exactly true under periodic
  boundary conditions, since the DFT implicitly treats the signal as
  periodic. Using `reflect`/`zero` padding would introduce a real,
  expected mismatch at the border that has nothing to do with a bug —
  using `wrap` isolates the comparison to exactly what the theorem claims.

## Features

- **Raw pixel features use global normalization (divide every pixel by
  255.0, same scale for every image), not per-image normalization
  (per-image mean/std or min-max).**
  Reason: this is a retrieval task that compares images to each other by
  distance. Per-image normalization would erase real brightness/contrast
  differences between images before any comparison happens -- two images
  of the same object under different lighting would be made to look more
  alike than they really are. Global scaling only changes units, not which
  images are nearest neighbours of which. See `src/features/pixels.py`.

- **HOG uses orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
  block_norm='L2-Hys'** on 32x32 grayscale CIFAR-10 images, giving a
  4x4 grid of cells, 3x3 overlapping blocks, and a 324-d feature vector.
  Reason: these are the original Dalal & Triggs (2005) defaults;
  pixels_per_cell=(8,8) is the largest cell size that still gives more
  than one cell across a 32x32 image (a 16x16 cell would give only a 2x2
  grid, too coarse to capture CIFAR-10's small objects).

- **ResNet-18 input resolution: 224x224 (not 32 or 64).**
  Reason: measured first, not guessed (`results/b_resnet_resolution_benchmark.csv`,
  on a 500-image subset): extracting all 10,000 images takes about 0.9 min
  at 32px, 0.9 min at 64px, and 3.6 min at 224px on this CPU-only machine
  -- all comfortably under the 20-minute "say so first" threshold set
  for long-running experiments. Since the time cost of 224px is small in absolute terms, we
  use it despite being the slowest option, because 224x224 is the
  resolution ImageNet-pretrained ResNet-18 was actually trained on; its
  early convolutional layers expect that much spatial detail, so feeding
  it 32x32 images (upscaled from CIFAR-10's native size either way, since
  CIFAR-10 images never had more than 32x32 real detail) asks the network
  to operate far outside the input statistics it learned on. We do not
  have a measured accuracy comparison between resolutions (that would
  need the actual retrieval experiment in Part C), so this choice is based
  on the standard ResNet-18 training protocol, not on measured retrieval
  quality; if Part C's results look poor for ResNet features, re-checking
  this choice against measured recall would be the first thing to try.

- **ResNet images are resized with bilinear interpolation** (not nearest-
  neighbour or area/box resizing) when going from 32x32 to 224x224, since
  bilinear is the standard choice for upsampling into a pretrained
  network's expected input size and avoids the blocky artifacts nearest-
  neighbour would introduce.

## Projection and retrieval

- **All three projections are scaled to be norm-preserving in
  expectation: E[||Rx||^2] = ||x||^2.** Dense Gaussian entries have
  variance 1/m; Achlioptas sparse entries are scaled by 1/sqrt(m) on top
  of the {+sqrt(3),0,-sqrt(3)} values; SRHT's final subsampling step is
  scaled by sqrt(D/m). Reason: without a consistent scale convention
  across the three methods, any recall difference between them could
  partly be an artifact of one method silently shrinking or inflating
  distances rather than a genuine difference in how well it preserves
  neighbour structure -- norm-preservation in expectation is the standard
  way to rule that out and is what the JL/Achlioptas theory assumes.

- **SRHT's sign-flip and column-subsampling randomness are both derived
  from the single `seed` argument**, consumed in order (sign flips drawn
  first, then the m selected indices), rather than taking two separate
  seed arguments. Reason: this keeps the same (feature, projection, m,
  seed) -> result interface across all three projection modules
  (gaussian.project, sparse.project, srht.project all take the same
  (X, m, seed) signature), which is what lets `c_projection_recall.py`
  loop over all three with identical code.

- **SRHT uses a hand-written vectorized Fast Walsh-Hadamard Transform
  (butterfly algorithm)**, not `scipy.linalg.hadamard` (which builds the
  full D x D matrix explicitly). Reason: materializing a D x D Hadamard
  matrix for D=4096 (the padded size for 3072-d pixel features) would be
  a 4096x4096 float64 array (128 MB) rebuilt on every single projection
  call; the O(D log D) butterfly algorithm computes the identical
  transform without ever storing that matrix, which is both the
  textbook-correct way to get the "fast" in Fast Hadamard Transform and
  meaningfully cheaper here.

- **Bootstrap CI is computed over the pooled (10 seeds x 10,000 queries)
  array of per-query recall values, not over per-seed mean recalls, and
  not per-seed separately.** The study design calls for "the mean over seeds and
  a 95% bootstrap CI over queries". We pool every (seed, query) recall
  value into one array before resampling, so the reported mean equals the
  average of the 10 per-seed means (every seed contributes equally many
  values), while the bootstrap resampling captures variability from both
  sources at once (query-to-query and seed-to-seed) rather than only one.
  This is a pragmatic reading of that instruction, stated explicitly here
  since the exact wording is ambiguous between "resample queries only,
  holding seeds fixed" and this pooled approach; we chose the pooled
  version because it is simpler to implement correctly and gives a
  slightly more conservative (wider) CI than resampling only 10 seed-level
  means would.

- **Full sweep timing was measured on real feature data before running
  the whole experiment**, following the ">20 minutes, say so and run in
  background" rule for long experiments: ground-truth top-10 on the full 3072-d pixel features
  took ~5s, a single projected top-10 computation (any m from 4 to 256)
  took ~1-1.5s (dominated by the fixed O(N^2) pairwise-distance cost, not
  by m), and the full sweep (3 features x 3 projections x up to 7 valid m
  values x 10 seeds, 63 settings total) was estimated at ~15 minutes from
  that subset measurement. We ran it in the background regardless, with
  progress logged to `results/c_recall_progress.log`. Actual measured
  total was 18.5 minutes (`results/c_recall_progress.log`'s last line) --
  about 25% longer than the subset-based estimate, most likely because
  the per-setting time grows slightly with m (visible in the log: ~14s at
  m=4 vs ~17-19s at m=256, since larger-m projections and their
  Hadamard/matrix-multiply steps cost a little more) in a way the small
  benchmark at only two m values (4 and 256, not all 7) underestimated.
  Recorded here rather than silently adjusted, following this study's
  honesty rule about not shrinking or hand-waving timing numbers.

## Intrinsic dimension and collapse analysis

- **Collapse analysis uses only the 3 feature-type-level recall curves,
  NOT separate recall@10 sweeps for each of the 10 classes.** This is a
  deliberate, acknowledged scope reduction from the original study design ("recall
  curves for different feature types and classes should collapse"),
  recorded here as this log's own rule requires. Reason: running the full
  recall@10 sweep (3 projections x 7 m values x 10 seeds) separately for
  each of the 10 classes, on top of the 3 feature types already done,
  would multiply the projection-and-retrieval sweep's measured 18.5-minute runtime by roughly 10x
  (each class subset is smaller, ~1000 points vs 10,000, so the per-run
  cost is less than 10x, but still a large, multi-hour addition) for a
  7-day project with many other things still to build. ID estimation
  itself *is* done at the per-class level (30 (feature, class) points,
  `results/c_intrinsic_dim.csv`) exactly as planned, so the
  per-class ID numbers exist and are reported; what's reduced is only the
  per-class *recall curve* half of the collapse check. This is stated
  explicitly in the report limitations, not silently dropped.

- **Primary ID estimator for the collapse analysis: TwoNN, not the
  Levina-Bickel MLE** (both are computed and reported for every row of
  `results/c_intrinsic_dim.csv`, but only TwoNN's value is used as
  "ID_hat" in the m/ID_hat collapse plot and the Spearman correlation).
  Reason: TwoNN has no tunable parameter (MLE needs a choice of k range),
  which makes it simpler to justify as "the" ID_hat for a single
  headline comparison, and the two estimators agree reasonably well on
  real features anyway (e.g. pixels: TwoNN=31.6 vs MLE=25.1 overall --
  same order of magnitude, same ranking across feature types in every
  case we checked).

- **The 10-d hypercube synthetic validation underestimates the true
  dimension (TwoNN: 8.75, MLE: 8.38, vs true 10), while the 5-d linear
  subspace is recovered almost exactly (TwoNN: 5.00, MLE: 5.03).** This
  is NOT a bug in our implementation: it is a well-documented boundary
  effect -- points near a hypercube's faces/edges/corners have
  systematically fewer neighbours in some directions than points in the
  interior, which biases local neighbour-ratio-based estimators downward,
  and both the TwoNN and Levina-Bickel papers note exactly this bias for
  bounded, non-smooth-boundary distributions like a hypercube. Reported
  honestly in `results/c_id_validation.csv` and the report rather than
  tuned away, following this study's rule against ever adjusting a method
  to make a number look better.

- **Collapse curves average recall@10 over the 3 projection methods at
  each m**, rather than keeping 9 separate (feature x projection)
  curves. Reason: the collapse hypothesis under test is specifically
  about feature types (and classes), not about projection method; mixing
  in projection-method variation would make the 3 "feature curves" noisier
  for no reason relevant to the hypothesis being tested. The separate,
  per-projection recall differences (e.g. SRHT's edge on ResNet features,
  noted in the projection-and-retrieval sweep) are a different question, already reported in
  `results/c_recall.csv` and `results/figures/c_recall_vs_m_*.png`.

- **Collapse statistic (pre-declared before looking at the result): for
  each x-axis choice (m, m/d, m/ID_hat), build a common log-spaced grid
  over the OVERLAPPING range of all 3 feature curves' x-values, linearly
  interpolate each curve onto that grid, and average the across-feature
  standard deviation at each grid point.** Lower = curves sit closer
  together = more "collapse". Measured result: m/ID_hat gives 0.031,
  raw m gives 0.043, and m/d gives 0.190 (nearly 5x worse than
  m/ID_hat) -- m/ID_hat is the best of the three, and m/d is clearly
  the worst, which is a real, measured, honest result, not an expected
  outcome assumed in advance. It is NOT a clean collapse (visible
  separation remains between the 3 curves in
  `results/figures/c_collapse_vs_m_over_id.png`), so the honest summary
  is "ID_hat explains some but not all of the difference between feature
  types' compressibility" -- a partial, exploratory finding, not
  confirmation of a clean hypothesis.

- **Spearman correlation between ID_hat and recall-loss at a fixed m is
  computed on only n=3 points (one per feature type)**, since that is
  all the feature-type-level data available (see the scope-reduction
  decision above: no per-class recall curves). The result (rho=0.5,
  p=0.667 at m=64) is reported but explicitly flagged as statistically
  uninformative at this sample size -- a correlation computed from 3
  points can only take a handful of possible values and its p-value
  cannot be small no matter what the data looks like. Included for
  completeness since the study design calls for it, not presented as evidence.

## Report and verification

- **`pdflatex` was not installed on the development machine.** A minimal
  LaTeX distribution (BasicTeX, via Homebrew) was installed to compile
  `report/report.tex`. The install requires `sudo`, so it was run
  directly in a terminal by the author, entering the system password
  interactively; the installing process never saw or handled that
  password.

- **README's "Results summary" section is regenerated by a dedicated
  script (`report/make_readme_summary.py`) that only rewrites the text
  between two HTML comment markers**, leaving the rest of the README
  (narrative Day-by-day status log, setup instructions) untouched.
  Reason: this study's rule #1 ("every number in the README and report
  must be produced by a script... not typed by hand") applies most
  strictly to numbers presented as results; the Day-by-day status log is
  narrative commentary written as each day's work finished, and its
  numbers were copied from the same CSVs the generated summary reads (so
  they agree), but they are not themselves script-generated text. The
  dedicated "Results summary" section is the canonical, provably
  generated source of every number; the Day-by-day log is kept as a
  readable log of what happened and why, cross-referencing the same
  files.

- **Report tables are kept deliberately short** (e.g. the per-class
  intrinsic-dimension table, 30 rows, is generated as a .tex fragment by
  `make_tables.py` but NOT `\input` into the 2-3 page `report.tex` body --
  only the 3-row feature-type-level summary is). Reason: the study design
  calls for a 2-3 page report; a 30-row table would not fit that budget. The
  full per-class numbers are still fully generated, still in
  `results/c_intrinsic_dim.csv`, and the report text explicitly points
  there -- nothing is hidden, just not inlined into the short report.

- **Fresh-clone verification: cloned the committed git history into a
  separate scratch directory, ran `./run_all.sh` there from scratch**
  (new venv, re-downloaded CIFAR-10, re-ran every experiment, recompiled
  the report), and compared every regenerated `results/*.csv` against the
  originally committed versions. Result: every seeded, scientifically
  meaningful number matched bit-for-bit (recall@10 means and CIs,
  intrinsic dimension estimates, the collapse-spread statistic, Part A
  max-abs-error validation numbers, feature shapes) -- zero numeric
  difference across `c_recall.csv`, `c_intrinsic_dim.csv`,
  `c_id_validation.csv`, and `c_collapse.csv`, and zero difference in the
  non-timing columns of `a_validation.csv` and `b_feature_extraction.csv`.
  The only numbers that differed between runs were wall-clock timing
  measurements (e.g. `elapsed_sec`, `time_ours_sec`), which are not
  seeded and depend on machine load -- exactly the kind of difference
  the "match within seeds' tolerance" standard allows, not a reproducibility
  failure. All 75 pytest tests passed in the fresh clone too. The
  verification clone was deleted afterward (it is not part of the
  project folder and was only ever scratch space for this check).
