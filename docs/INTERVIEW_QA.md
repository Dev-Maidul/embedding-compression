# Interview Q&A log

Three short interview-style questions per day. Owner answers are recorded
here too, once given.

## Day 1

**Q1. What is the difference between convolution and correlation, and
which one does `cv2.Sobel` actually use?**

Model answer: Correlation slides the kernel over the image as-is and
computes a weighted sum at each position. True convolution first flips the
kernel 180 degrees (both axes reversed), then does the same sliding sum.
For a symmetric kernel (like Gaussian) flipping does nothing, so the two
are identical. For an asymmetric kernel (like Sobel's [-1,0,1] rows),
flipping changes the sign/order of the result. We measured this directly:
using true convolution for Sobel gave the exact negative of `cv2.Sobel`'s
output, so `cv2.Sobel` uses correlation, not a flipped kernel, despite
"convolution" being the common name people use for this kind of operator.

Owner answer: _(pending)_

**Q2. Why does the Gaussian kernel get divided by its sum (normalized)
before it's used to blur an image?**

Model answer: Normalizing makes the weights sum to exactly 1. That means
blurring a constant-value image leaves it unchanged (a constant times
weights-summing-to-1 is the same constant), and more generally it keeps
the image's overall brightness the same — the blur redistributes energy
between neighbouring pixels but doesn't add or remove any. Without
normalization, the output would be scaled up or down depending on the
kernel size and sigma.

Owner answer: _(pending)_

**Q3. Why does the validation report "max absolute error" against OpenCV
instead of just saying "it matches"?**

Model answer: Floating-point arithmetic is never bit-exact across two
different implementations (different operation order accumulates
different rounding). "It matches" is not falsifiable; "max abs error is
1.7e-13 pixels, versus pixel values up to 255" is a precise, checkable
claim that anyone can verify by re-running the script, and it is honest
about the fact that the two are extremely close but not mathematically
identical bit-for-bit. This also matters because Sobel matched *exactly*
(error 0.0) once we matched OpenCV's correlation convention — a previous
version using true convolution had a large, measurable error (the sign
flip), which is exactly the kind of thing max-abs-error catches and "it
looks right" would not.

Owner answer: _(pending)_

## Day 2

**Q1. In non-maximum suppression, why do we compare a pixel's gradient
magnitude to its neighbours *along the gradient direction*, instead of
just its 4 or 8 nearest neighbours in general?**

Model answer: An edge is a ridge in the magnitude surface, running
perpendicular to the gradient direction (the gradient points across the
edge, from dark to light). To thin that ridge down to one pixel wide, we
need to check whether the current pixel is the highest point *along the
direction the ridge is narrow in* — which is the gradient direction, not
along the edge itself. Comparing to neighbours in a fixed direction (say,
always left/right) would incorrectly suppress real edges that happen to
run vertically, or fail to thin edges that run horizontally. That's why we
quantize the gradient direction per pixel and pick the matching pair of
neighbours for that pixel specifically.

Owner answer: _(pending)_

**Q2. What does "hysteresis" mean in double-threshold hysteresis, and why
not just use one threshold?**

Model answer: A single threshold forces an all-or-nothing choice at one
cutoff value, which is fragile: a continuous edge whose contrast fades in
and out would get broken into disconnected fragments wherever it dips
below the cutoff, even briefly. Hysteresis uses two thresholds: pixels
above the high one are trusted outright ("strong"); pixels between low and
high are only kept if they chain back to a strong pixel ("weak but
connected"). This lets a real edge keep its fainter sections intact (as
long as they connect to a confident part of the same edge) while still
throwing away isolated weak responses that are probably just noise, since
noise rarely forms a connected chain back to a strong edge.

Owner answer: _(pending)_

**Q3. Why does low-pass filtering with a hard circular cutoff in the
frequency domain produce ringing (ripples) around edges in the output
image, instead of a clean blur?**

Model answer: A hard cutoff is a rectangular (brick-wall) function in the
frequency domain. The inverse Fourier transform of a rectangle is a sinc
function in the spatial domain, which has oscillating side-lobes that
never fully decay — so the filter is effectively convolving the image with
a sinc-shaped kernel, and that kernel's ripples show up as visible ringing
near sharp edges (the Gibbs phenomenon). A Gaussian low-pass filter avoids
this because the Fourier transform of a Gaussian is itself a Gaussian,
with no sharp edges and no ringing, at the cost of a less sharply defined
cutoff frequency.

Owner answer: _(pending)_

## Day 3

**Q1. Why do raw pixel features get normalized the same way for every
image (divide by 255), instead of each image being normalized to its own
mean and standard deviation?**

Model answer: This is a nearest-neighbour retrieval task — we only care
about distances *between* feature vectors. Per-image normalization (e.g.
subtracting each image's own mean, dividing by its own std) would remove
real brightness and contrast differences between images before any
distance is computed, which can make two genuinely different images (say,
a dark photo and a bright photo of different objects) look artificially
more similar, or two images of the same object under different lighting
look artificially more different depending on what normalization does to
each independently. A single global scale factor (divide by 255) changes
units only — it does not change which image is closest to which in
Euclidean distance — so it preserves whatever real signal was in the raw
brightness values.

Owner answer: _(pending)_

**Q2. What does the HOG descriptor actually throw away compared to raw
pixels, and why is that a useful thing to throw away for retrieval?**

Model answer: HOG discards the exact pixel intensities everywhere except
their local gradient direction and magnitude, pooled into per-cell
histograms, and even then renormalizes those histograms per block. That
means two images with the same edge/shape layout but different overall
brightness, or a small shift of a few pixels, end up with much more
similar HOG vectors than they would as raw pixels. Raw pixel distance is
dominated by exact alignment and lighting; HOG distance is closer to "do
these two images have a similar arrangement of edges", which is usually a
more useful notion of visual similarity for retrieval, at the cost of
losing texture/color information entirely.

Owner answer: _(pending)_

**Q3. Why did we decide to resize CIFAR-10 images up to 224x224 before
feeding them to ResNet-18, instead of just using them at their native
32x32 size, given that upscaling cannot add real detail?**

Model answer: You're right that upscaling a 32x32 image to 224x224 adds
no new information — it's the same image, just interpolated to more
pixels. But a pretrained network's convolutional filters were learned to
respond to patterns at a particular spatial scale, because that's the
scale of the images (224x224 ImageNet photos) they were trained on. Its
early layers assume features like edges and textures span a certain
number of pixels; feeding genuinely tiny 32x32 input (even without
resizing) would make those filters see something resembling heavily
downsampled, aliased versions of the patterns they expect, which tends to
produce worse features in practice. Since the benchmark showed 224px only
costs an extra ~2.7 minutes over 10,000 images compared to 32px — nowhere
near the 20-minute budget — we chose to match the network's native
training resolution rather than save time we did not need to save. We
have not yet measured whether this actually produces better retrieval
accuracy than 32px would (that only shows up once we run Part C); it's
currently a choice based on how ResNet-18 was trained, not on our own
measured results.

Owner answer: _(pending)_

## Day 4

**Q1. What does it mean for a random projection to be "norm-preserving in
expectation", and why does it matter that all three projection methods
(Gaussian, sparse, SRHT) share that property?**

Model answer: "Norm-preserving in expectation" means E[||Rx||^2] = ||x||^2
-- averaged over many random draws of the projection matrix R, the
squared length of the projected vector equals the squared length of the
original. Any single draw will be a bit off (sometimes larger, sometimes
smaller), but there's no systematic shrinking or inflating built in. This
matters because we compare recall@10 across the three projection methods
to see which one preserves neighbour structure better; if one method
scaled distances differently from the others on average, a recall
difference we observed might just reflect that scale difference rather
than a real difference in how well it preserves which-point-is-nearest
information. Matching this convention across methods makes the comparison
about the thing we actually want to compare.

Owner answer: _(pending)_

**Q2. Why does recall@10 increase as the projection dimension m
increases, and why does it start so low (around 0.01-0.04) even at m=4
rather than exactly 0?**

Model answer: Larger m means less information is thrown away, so more of
the structure that determined the original top-10 neighbours survives --
that's the expected trend across every feature type and projection method
in our results. It isn't exactly 0 at m=4 because even a very aggressive
compression isn't *random*: a true neighbour that was close in the
original space is still, on average, somewhat more likely to end up close
after projection than a randomly chosen other point is, even if that edge
is weak. With 10,000 candidate points and only 10 true neighbours out of
9,999 possible ones, recall near 0.01-0.04 at m=4 is roughly in the range
you'd expect for "barely better than chance but not literally chance"
(pure chance would be 10/9999 ≈ 0.001).

Owner answer: _(pending)_

**Q3. In the ResNet recall-vs-m plot, SRHT visibly outperforms Gaussian
and sparse projection at large m (recall@10 ≈ 0.69 vs ≈ 0.58 at m=256).
Pixels and HOG don't show this gap. What's a plausible reason, and what
would you need to check before trusting it as a real finding rather than
noise?**

Model answer: A plausible reason is that ResNet features (unlike raw
pixels or HOG) likely have structure that isn't spread evenly across all
512 dimensions -- some directions probably carry much more of the
"information that determines nearest neighbours" than others (this is
exactly what Part C's intrinsic-dimension analysis is designed to
measure). SRHT's Hadamard step mixes all input coordinates together
before subsampling, which could make it better at capturing a
concentrated signal with fewer kept coordinates than a dense Gaussian
projection, which combines coordinates more "diffusely" but with more
random variance per output. Before trusting this as a real effect rather
than noise: check that the gap is larger than the reported 95% CIs (it
is, from the logged numbers -- the CIs are each only about +/-0.001
wide), check it's consistent across nearby m values (it is: visible from
m=64 onward, not just at m=256), and ideally re-run with different seeds
or a different ResNet layer to see if it's specific to this exact feature
set. This is noted as an open, not-yet-explained observation in the
README rather than an explained result.

Owner answer: _(pending)_

## Day 5

**Q1. Why does TwoNN only look at the 1st and 2nd nearest neighbours of
each point, instead of using more neighbours like the MLE estimator
does?**

Model answer: Using just the two closest neighbours and their distance
ratio makes the estimator insensitive to how the local density of points
varies across the dataset -- a region with many nearby points and a region
with few will still give mu = r2/r1 ratios following the same
theoretical distribution if the local dimension is the same, because both
r1 and r2 scale together with density, but their ratio doesn't. An
estimator using more neighbours (like the k nearest in Levina-Bickel's
MLE) has to assume the data looks locally flat over the whole
neighbourhood of k points, which breaks down more if the manifold curves
or the density changes a lot within that neighbourhood. The trade-off:
TwoNN throws away a lot of information (everything beyond the 2nd
neighbour), which is part of why we also implement and cross-check
against the MLE estimator rather than trusting either one alone.

Owner answer: _(pending)_

**Q2. Both estimators recovered the 5-d linear subspace's dimension
almost exactly (5.00, 5.03) but underestimated the 10-d hypercube's
dimension (~8.4-8.75). Why would the same methods behave differently on
these two cases, and does that mean the implementation is wrong?**

Model answer: Not wrong -- this is an expected, documented effect of the
hypercube's shape, not the linear subspace's. A linear subspace extends
infinitely (or, in our sampled version, points are drawn from an
unbounded Gaussian) with no edges, so every point's local neighbourhood
looks the same in every direction, matching what both estimators assume.
A hypercube [0,1]^10 has faces, edges, and corners: a point near a face
has neighbours in only "one side" along that axis, not both, which skews
the local neighbour-distance statistics the same way a partial
measurement would. This boundary bias is specifically called out in both
original papers for bounded distributions. We did not tune anything to
fix this -- reporting it honestly, rather than hiding or "correcting" it,
is itself evidence the implementation is doing what the theory predicts,
not a sign something is broken.

Owner answer: _(pending)_

**Q3. The m/ID_hat collapse statistic (0.031) beat raw m (0.043) and
clearly beat m/d (0.190), which sounds like it supports the hypothesis.
But the report calls this result "partial, not clean." What's the
difference, and why does it matter to say so?**

Model answer: "Supports the hypothesis" would mean the three feature
types' recall curves, when plotted against m/ID_hat, land on essentially
the same line -- predicting one feature type's recall from another's at
matched m/ID_hat would then be reliable. What we actually see
(`results/figures/c_collapse_vs_m_over_id.png`) is the three curves
moving much closer together than under m or m/d, but still visibly
separated from each other -- ID_hat explains some, but clearly not all, of
why different feature types compress differently. Calling that "clean
collapse" would overstate the finding; calling it "no effect" would
understate it. The project brief explicitly asks for honest reporting of
negative or partial results rather than dressing up a weak result as a
strong one (or a real, modest result as if it weren't there) -- an
interviewer who has seen the figure would immediately notice the overclaim
if we called this a clean collapse.

Owner answer: _(pending)_

## Day 6

**Q1. The report and README both say "nothing is typed by hand" -- what
would actually break if a number in `report.tex` had been typed directly
instead of generated by `make_tables.py`?**

Model answer: Nothing would break the LaTeX compile -- a typed number
looks identical on the page to a generated one. What breaks is
trustworthiness over time: if an experiment is re-run with a different
seed, a bug fix, or more data, the CSV changes but a hand-typed number in
the report would not, silently going stale and wrong without any error
or warning. A generated number is guaranteed to match whatever
`results/*.csv` currently says, because it's read from that file every
time the script runs; a typed number is only as correct as whoever typed
it remembered to update it. This matters most in a project meant to be
defended in an interview -- a question like "how did you get this number"
should always have the answer "run this script on this CSV", never "I
copied it from an earlier run."

Owner answer: _(pending)_

**Q2. Why was installing a LaTeX distribution (BasicTeX) something that
needed your explicit approval, when installing Python packages into the
project's venv earlier did not?**

Model answer: The Python packages went into `venv/`, a folder inside this
project that the project brief's rules clearly apply to and that gets rebuilt
from `requirements.txt` on any clean clone -- entirely contained, and
explicitly part of the project's own setup. A LaTeX distribution
installed via Homebrew modifies files on the machine outside this
project folder (shared system locations any other project or application
could also use), and the install itself needed `sudo`, meaning it also
needed your own password -- which the project brief's rule #7 says to never
touch. Those two differences (outside the project folder, and requiring
a password only you should enter) crossed the line from "routine project
setup" to "something that needed your explicit go-ahead first."

Owner answer: _(pending)_

**Q3. The first LaTeX compile produced 5 pages; the project brief asks for 2-3.
What did changing to fix that, and what did NOT change?**

Model answer: What changed was purely visual/formatting: a smaller font
(10pt instead of 11pt), tighter margins, less vertical space around
section headers and tables, several nearly-duplicate sentences trimmed
(for example, removing "Table X shows..." lead-ins that just repeated
what the table's own caption already said), and two pairs of related
small tables merged side-by-side instead of stacked vertically. What did
NOT change is any number, any finding, or any of the honest hedging
language (like "partial, not clean collapse", or "not statistically
meaningful at n=3") -- compressing a report to fit a page budget is a
layout problem, and the project brief's honesty rules are explicit that content
should never be cut or softened just to make the page count or the
results look better.

Owner answer: _(pending)_
