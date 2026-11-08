# Explainers

One short, plain-English section per module. Read this before an interview
question about that module; it is the "why", not just the "what" (the code
itself already says what).

## `src/ip/convolution.py`

What it does: slides a small grid of numbers (the kernel) over an image
and, at each pixel, replaces it with a weighted sum of its neighbours.

Key formula (in words): output(i, j) = sum over all (di, dj) in the kernel
of image(i + di, j + dj) * kernel(di, dj). That is correlation. True
convolution flips the kernel 180 degrees first (so di, dj become -di, -dj).
For a symmetric kernel (like Gaussian) the two give the same answer; for an
asymmetric kernel (like Sobel) they differ by a sign/mirror.

Why it matters: this one function underlies every other Part A operation
(blurring, edge detection). Getting padding and the flip/no-flip choice
right, and being able to say exactly which one `cv2.filter2D` and
`cv2.Sobel` use, is the first thing an interviewer will probe.

Implementation note: instead of a literal double `for` loop over pixels
(correct but slow on a CPU laptop for 512x512 images), we build all
sliding windows at once with `np.lib.stride_tricks.sliding_window_view`
and multiply-and-sum them with `np.einsum`. It computes the exact same
numbers as the loop would, just without doing one Python-level loop
iteration per pixel.

## `src/ip/gaussian.py`

What it does: builds a 2D Gaussian "bell curve" kernel and uses it to blur
an image.

Key formula: kernel(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2)), then
normalized so all the weights sum to 1 (so a blurred constant image stays
that same constant — no energy added or removed).

Why it matters: blurring removes small noise and fine detail before taking
derivatives, which is exactly why Canny's first step is a Gaussian blur.
Sigma controls how much: larger sigma = more blur = removes more noise but
also more real detail.

## `src/ip/sobel.py`

What it does: estimates how fast the image intensity changes left-right
(gx) and up-down (gy) at every pixel, using two small 3x3 kernels, then
combines them into an edge strength (magnitude) and edge direction.

Key formula: magnitude = sqrt(gx^2 + gy^2), direction = atan2(gy, gx).

Why it matters: this is the "gradient" step that both classic edge
detection (Canny) and some hand-crafted feature descriptors (HOG, used in
Part B) are built on.

Measured, not assumed: cv2.Sobel applies its kernel without flipping it
(correlation), even though people often call it a "convolution" kernel.
We checked this directly — flipping the kernel (true convolution) gave the
exact negative of cv2.Sobel's output on a random test image, so our
`sobel_gradients` uses correlation (no flip) to match. See
`docs/DECISIONS.md` and `results/a_validation.csv`.

## `src/ip/histeq.py`

What it does: spreads out a low-contrast image's pixel values so they use
the full 0-255 range as evenly as possible, using only the image's own
histogram (no neighbours, no kernel).

Key formula (in words): build the histogram, turn it into a cumulative
distribution function (CDF) by running total, then rescale that CDF so it
spans exactly 0 to 255, and use it as a lookup table: new_value =
rescaled_CDF[old_value].

Why it matters: a dark or washed-out photo can have most of its pixels
crammed into a narrow slice of 0-255; equalization redistributes them so
detail that was invisible becomes visible, with no extra information
needed beyond the image itself.

Measured: matches `cv2.equalizeHist` exactly (100% pixel agreement, 0 max
abs error) on all validation images, since both implement the identical
CDF-rescaling formula.

## `src/ip/canny.py`

What it does: the full Canny edge detector, built from pieces already in
this package: Gaussian blur (noise removal) -> Sobel gradients (edge
strength and direction) -> non-maximum suppression, NMS (thin thick
gradient ridges to 1-pixel-wide lines) -> double threshold + hysteresis
(keep confident edges, keep faint edges only if chained to a confident
one, discard isolated faint responses).

Key idea for NMS: a real edge is a ridge in the gradient-magnitude surface.
At each pixel, look at its magnitude compared to the two neighbours lying
along the gradient direction (quantized to the nearest of 0/45/90/135
degrees); keep the pixel only if it's the local peak of that ridge, else
suppress it to 0.

Key idea for hysteresis: pixels with magnitude above the high threshold
are "strong" and always kept; pixels between low and high are "weak" and
kept only if they are 8-connected (directly or through a chain of other
weak pixels) to at least one strong pixel. We implement the connectivity
check with `scipy.ndimage.label` (connected-component labeling) rather
than a hand-written flood fill — same result, standard tool.

Honest comparison: our output does not pixel-match `cv2.Canny` exactly
(93-99% pixel agreement across validation images, see
`results/a_validation.csv`), because the two use different NMS
interpolation (we quantize to 4 sectors, OpenCV interpolates continuously)
and different threshold conventions (we threshold as a fraction of this
image's own max gradient; OpenCV takes fixed absolute gradient values).
Both are valid, standard variants of the same algorithm family; see
`docs/DECISIONS.md`.

## `src/ip/fourier.py`

What it does: filters images by their frequency content using `np.fft`,
and demonstrates the convolution theorem (spatial convolution ==
frequency-domain multiplication).

Key idea: any image is a sum of 2D sine waves at different frequencies.
Low frequencies are smooth brightness changes; high frequencies are fine
detail, edges, and noise. A low-pass filter zeroes out the high-frequency
part of the spectrum (blurs); a high-pass filter zeroes out the low part
(keeps only detail/edges).

Convolution theorem, in words: convolving an image with a kernel in the
spatial domain gives exactly the same result as (1) taking the FFT of
both, (2) multiplying them pointwise, (3) taking the inverse FFT. We
verify this numerically: `results/a_fourier_timing.csv` shows max abs
error around 1e-13 (floating-point noise) between the two routes, for
kernel sizes from 3x3 to 31x31 on a 512x512 image.

Why it matters for speed: spatial convolution cost grows with kernel size
(more multiply-adds per pixel); FFT-based convolution cost is
~independent of kernel size (dominated by the fixed-cost FFT itself,
O(N log N) in the number of pixels). The timing plot
(`results/figures/fourier_timing.png`) shows spatial time growing roughly
linearly with kernel area while FFT time stays flat — so for large
kernels, FFT-based convolution wins.

One subtlety: our ideal (hard-cutoff, "brick wall") low-pass filter
produces visible ringing artifacts around edges in the figure
(`results/figures/frequency_filtering.png`) — a classic, expected
consequence of a sharp cutoff in frequency space (Gibbs phenomenon), not a
bug. A smoother cutoff (e.g. a Gaussian mask) would reduce the ringing at
the cost of a less sharp frequency boundary.

## `src/features/pixels.py`

What it does: the simplest possible image feature — flattens every pixel
of a 32x32x3 image into one 3072-length vector, scaled to [0, 1].

Why it matters: it is a deliberately weak baseline. It has no notion of
shape, is very sensitive to small shifts or lighting changes, but is
cheap and gives later feature types (HOG, ResNet) something to clearly
beat in the Part C retrieval experiments.

Decision worth remembering: normalization is global (divide by 255 for
every image, the same scale always), not per-image. Per-image
normalization (e.g. subtracting each image's own mean) would throw away
real brightness differences between images before any distance is even
computed, which matters for a retrieval task built entirely on comparing
images to each other.

## `src/features/hog.py`

What it does: Histogram of Oriented Gradients (Dalal & Triggs, 2005).
Splits the image into small cells, and in each cell builds a histogram of
"how much edge-gradient points in each of 9 directions"; neighbouring
cells are grouped into overlapping blocks and normalized together to
reduce sensitivity to lighting.

Key formula (in words): for each pixel, compute the gradient direction and
magnitude (same idea as Sobel); for each cell, add every pixel's magnitude
into the histogram bin matching its direction; concatenate and
block-normalize.

Why it matters: unlike raw pixels, HOG describes *shape* (edge layout)
rather than exact pixel values, so it is far more robust to small
lighting/contrast changes and is a classic hand-crafted alternative to
learned features like ResNet's.

Parameters used here (32x32 images): 9 orientation bins, 8x8-pixel cells
(giving a 4x4 cell grid), 2x2-cell blocks (giving 3x3 overlapping block
positions), L2-Hys block normalization -> 324-dimensional output. See
`docs/DECISIONS.md` for why these exact numbers.

## `src/features/resnet.py`

What it does: runs an image through a ResNet-18 convolutional network
(He et al., 2016) pretrained on ImageNet, and reads off the
512-dimensional vector from just before its final classification layer
(the "penultimate" layer, right after global average pooling).

Why it matters: this network learned, from 1.2 million natural images
across 1000 categories, a general-purpose way of describing visual
content (edges -> textures -> parts -> whole-object concepts, in
increasingly deep layers). The penultimate layer's output is a common,
effective "generic" feature representation, usually much stronger for
retrieval than either raw pixels or HOG, without any training on
CIFAR-10 itself (we only use it for inference -- "transfer learning" by
feature extraction, not fine-tuning).

Resolution decision, measured not assumed: the project brief asks to benchmark
speed before picking an input resolution (32, 64, or 224). On a 500-image
subset (`results/b_resnet_resolution_benchmark.csv`), extracting features
for all 10,000 CIFAR-10 test images takes about 0.9 minutes at 32px,
0.9 minutes at 64px, and 3.6 minutes at 224px on this CPU-only laptop.
Since even the slowest option is well under the project brief's 20-minute
"flag it first" threshold, we use 224px -- the resolution ResNet-18 was
actually trained on -- rather than optimizing for speed we did not need to
trade away. See `docs/DECISIONS.md`.

## `src/projection/gaussian.py`, `sparse.py`, `srht.py`

What they do: each takes an (N, d) array of feature vectors and a target
dimension m < d, and returns an (N, m) array -- a compressed version of
the same N vectors, using random (not learned) linear maps.

Key idea, shared by all three (Johnson-Lindenstrauss, 1984; Achlioptas,
2003): a *random* linear projection to a lower dimension approximately
preserves pairwise distances between points, with high probability, as
long as m is not too small relative to how many points there are and how
much distortion is tolerable. This is surprising -- there is no learning,
no looking at the data's structure at all, yet distances survive
reasonably well. Part C's whole question is exactly how well, and whether
that "how well" can be predicted from the data's intrinsic dimension.

- **Dense Gaussian**: every one of the d*m matrix entries is an
  independent Gaussian draw. Simple, strong theoretical guarantees, but
  O(d*m) to generate and multiply.
- **Sparse (Achlioptas)**: the same idea, but roughly 2/3 of the matrix
  entries are exactly 0, drawn from a 3-value distribution instead of a
  continuous Gaussian. Cheaper to generate and multiply (sparse matrix
  math can skip the zeros), same expected distance-preservation property.
- **SRHT (structured)**: instead of a random matrix at all, apply a
  fixed, structured, fast transform (sign flip -> Hadamard transform ->
  random coordinate subsampling). The Hadamard transform spreads
  information from every input coordinate evenly across every output
  coordinate, so subsampling a few outputs afterward still captures a
  representative mix of the whole input -- in O(d log d) time instead of
  O(d*m).

All three are implemented to be norm-preserving in expectation
(E[||Rx||^2] = ||x||^2), so recall differences between them in Part C
reflect genuine differences in how they preserve neighbour structure, not
one method silently using a different distance scale. See
`docs/DECISIONS.md`.

## `src/eval/knn_recall.py`

What it does: computes exact nearest neighbours (brute-force, not
approximate) for every point against every other point, and measures how
much of that neighbour set survives after projecting to fewer dimensions.

Key formula: squared Euclidean distance between a and b equals
||a||^2 + ||b||^2 - 2*(a . b). Computing this for all pairs at once
reduces to one matrix multiply (the Gram matrix a_i . a_j for all i, j)
plus some vector arithmetic, instead of a loop over every pair -- which is
what makes exact (not approximate) nearest-neighbour search on 10,000
points practical here.

recall@10 for one query = (number of its true top-10 neighbours that are
still in its top-10 after projection) / 10. This is the project's central
quality measure: 1.0 means projection didn't change this query's
neighbours at all; 0.0 means none of them survived.

## `src/eval/bootstrap.py`

What it does: turns a list of per-query measurements (like 10,000 recall
values) into a mean plus a range of plausible means (a 95% confidence
interval), without assuming anything about the shape of their
distribution.

Key idea: if we had collected a different, equally plausible sample of
10,000 queries, we'd get a slightly different mean recall. Bootstrapping
simulates "what if" by repeatedly resampling the queries we do have, with
replacement, and looking at how much the resulting mean actually varies.
The middle 95% of those resampled means is the 95% CI -- a report of how
much we should trust the exact mean we measured, given a finite sample.

## `src/eval/intrinsic_dim.py`

What it does: estimates a dataset's "intrinsic dimension" (ID) -- how many
real degrees of freedom the data actually varies along, even if it is
stored using many more numbers than that -- using two different
estimators, both based only on distances between nearby points.

**TwoNN (Facco et al., 2017)**, in words: for each point, look at only
its first and second nearest neighbours, and take the ratio of their
distances (mu = r2/r1). If the data locally looks like a flat patch of
dimension d, this ratio follows a specific, known probability
distribution that depends only on d. Plotting the data's actual mu values
against that theoretical distribution and fitting a line gives d as the
slope. The appeal: it uses only 2 neighbours per point, so it is
unaffected by how the local point-density varies across the dataset
(which would bias estimators that use many neighbours at once).

**Levina-Bickel MLE (2004)**, in words: treats the nearby points around
each point as if they were generated by a Poisson process whose rate
depends on the dimension, and derives the dimension value that maximizes
the likelihood of seeing the k nearest-neighbour distances actually
observed. Done for several different values of k (we use k=10 to 20,
averaged), since small k is noisy and large k gets biased by the data's
curvature (a locally flat approximation breaks down further from the
point).

Validated before trusting it on real data: both estimators were first run
on synthetic data with a known true dimension (`results/c_id_validation.csv`):
a 5-d Gaussian subspace embedded in 100-d space (both estimators recovered
~5.0, almost exactly right) and a 10-d hypercube embedded in 100-d
(both estimators gave ~8.4-8.75, an underestimate). The hypercube
underestimate is an expected, documented boundary effect (points near a
cube's corners/edges have asymmetric neighbourhoods), not a bug -- see
`docs/DECISIONS.md`.

## Day 5 experiments: collapse analysis

What it does (`experiments/c_collapse_analysis.py`): tests the project's
central, exploratory hypothesis -- that recall@10 at a given projection
size m depends mainly on a feature type's intrinsic dimension, so that
plotting recall against m/ID_hat (instead of raw m, or m/d) should make
the three feature types' curves collapse onto roughly the same curve.

Why m/ID_hat specifically: if a d-dimensional feature actually only has
ID_hat "real" degrees of freedom, then m/ID_hat is a rough measure of how
much of that real information survives compression to m dimensions --
whereas raw m ignores how much real structure there was to begin with,
and m/d assumes every one of the d stored numbers carries equally real
information, which intrinsic-dimension estimates exist specifically to
question.

## `report/make_tables.py` and `report/make_readme_summary.py`

What they do: turn `results/*.csv` into the exact LaTeX tables in
`report/report.tex` and the "Results summary" section of this README,
with no numbers typed by hand anywhere in either document (the project brief's
rule #1). `make_tables.py` writes small `.tex` fragments to
`report/tables/`, which `report.tex` pulls in with `\input`;
`make_readme_summary.py` rewrites only the text between two HTML comment
markers in `README.md`, leaving the rest of the file alone. Re-run both
any time an upstream CSV changes, then recompile `report.tex`, so the
written-out numbers never drift from the measured ones.

What we found, honestly: m/ID_hat gave the lowest (best) value of a
pre-declared "collapse spread" statistic (0.031) compared to raw m
(0.043) and m/d (0.190, by far the worst) -- so the hypothesis has real,
measured support. But the curves do not collapse cleanly onto one line;
visible separation remains between feature types even against m/ID_hat.
The Spearman correlation check (ID_hat vs recall-loss at a fixed m) uses
only 3 data points (one per feature type) and is reported but flagged as
statistically uninformative at that sample size -- see `docs/DECISIONS.md`
for why a larger version of this check (per-class recall curves, 30 more
data points) was out of scope for this project's time budget.
