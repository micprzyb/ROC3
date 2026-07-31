# Ideas log

Everything considered while building `roc3`, in the order the questions came up: what to
plot, what to integrate, how to compute it, how to draw it, how to attach uncertainty —
with the verdict for each, and the numbers that decided it. Rejected ideas are kept, with
the reason. Bugs found along the way are at the end.

Companion documents: [`PLAN.md`](PLAN.md) (the design that came out of this),
[`REFERENCES.md`](REFERENCES.md) (annotated sources).

Legend: **[USED]** shipped · **[REJECTED]** tried/analysed and discarded ·
**[DEFERRED]** sound but out of scope · **[FINDING]** empirical result · **[OPEN]** unresolved.

**Reading the codes.** Entries are numbered by section — `A1…A9` for what to plot, `B1…B9`
for what to integrate, `C` computation, `D` visual design, `E` uncertainty, `F` findings,
`G` bugs, `H` open questions, `I` the ordering constraint. **These are this document's own
numbers and do not line up with [`PLAN.md`](PLAN.md)'s** — always cite across documents as
"IDEAS_LOG §B6" or "PLAN §A4", never a bare code.

**Acronyms used below.** VUS = volume under the surface. AUC = area under the curve.
HUM = hypervolume under the manifold (the VUS past three classes). EEU = equal error
utility, the cost assumption that makes the rule family two-parameter (PLAN §A5).
LP = linear program. CVD = colour vision deficiency. OKLCH / OKLab = a perceptually
uniform colour space; **ΔE** is a distance in it (×100 throughout) and **ΔL** a difference
in its lightness coordinate. SE = standard error.

---

## A. What object should the "3-D ROC plot" be?

**A1 — Three stacked one-vs-rest ROC curves.** **[REJECTED as the answer, USED as a panel]**
Three thresholds `t₁,t₂,t₃` do not define a classifier: a point can clear two of them or
none. The three curves cannot show the coupling, which is the entire content of a 3-class
trade-off. Kept as dashboard panel F with an explicit caption saying so, because it is
what people expect to see and its absence would be read as an oversight.

**A2 — Hand–Till M / macro pairwise AUC.** **[REJECTED as the answer, USED as a number]**
Scalar; chance level 1/2, so it is not the volume of anything in 3-D. Reported in the
summary table for calibration of intuition.

**A3 — AUCμ (Kleiman & Page 2019).** **[DEFERRED]** A genuinely better scalar than A2
(it uses the whole probability vector rather than one column at a time), but still a
scalar with chance 1/2. Would be a nice addition to the comparator table; not implemented
because it adds a dependency-free but non-trivial partition solver for no gain in the
plot.

**A4 — The full 3-class ROC hypersurface.** **[REJECTED]**
A 3×3 row-stochastic confusion matrix has 6 free off-diagonal entries; the unrestricted
Bayes family has 5 free parameters, so the honest object is a 5-D manifold in 6-D. Two
independent reasons to reject: it cannot be drawn, and — decisively — Edwards & Metz
(2012) show the ideal observer's decision lines can meet the likelihood-ratio curve in at
most six places, so the manifold has **at most 4 degrees of freedom** and its 5-D
hypervolume degenerates. The restriction in A5 is therefore not a convenience, it is
forced.

**A5 — Mossman/Scurfield surface under Equal Error Utility.** **[USED — this is the answer]**
Assume the cost of misclassifying a class-`k` case depends only on `k`, not on which wrong
label was picked. The Bayes rule then collapses to `argmax_k w_k p_k(x)` with `w` on the
2-simplex — a 2-parameter family, tracing a 2-D surface in the 3-D space of per-class
sensitivities. Derivation: with `U_{i|k} = u_k·1[i=k] + v_k·1[i≠k]`, the expected utility
of deciding `i` is `(u_i − v_i)·p_i(x) + Σ_k v_k p_k(x)`, and the second term does not
depend on `i`. So `w_k ∝ u_k − v_k`. For two classes this *is* the threshold sweep.

**A6 — Pairwise/tournament thresholds (3 pairwise comparisons + vote).** **[REJECTED]**
Three free parameters (over-parameterised for a 3-D display) and, worse, the tournament
can cycle (A beats B beats C beats A), so the rule is undefined without an arbitrary
tie-break. Discarded before implementation.

**A7 — Arbitrary partitions of the probability simplex.** **[REJECTED]** Not a
2-parameter family, no canonical sweep, therefore no surface.

**A8 — Two cut-points on a single ordinal marker.** **[USED as a second mode]**
For genuinely ordered classes this is more actionable than A5: the two parameters are
literal cut-off values you can type into production. Shipped as `roc3.ordinal` with its
own dashboard. The panel-B analogue (the `(t₁,t₂)` plane) is arguably the single most
directly usable picture in the whole package.

**A9 — Plot in the log-weight plane `(a,b)` instead of the simplex.** **[REJECTED for
A5, USED for A8]** The `(a,b)` plane is where the rule actually varies, but it is
unbounded and the axes ("log of a weight ratio") mean nothing to a clinician. The simplex
is bounded, shows the whole family at once, and reads as "how much do I up-weight each
class". For the ordinal mode the raw parameter plane *is* interpretable (they are marker
values), so there it is used directly.

---

## B. What exactly is "the volume"?

**B1 — Down-set hypervolume of the achievable operating points.** **[USED — primary]**
`VUS = vol( ⋃_w [0,S₁(w)]×[0,S₂(w)]×[0,S₃(w)] )`. Two readings:
*geometric* — the probability that a uniformly random requirement triple "Se₁ ≥ u,
Se₂ ≥ v, Se₃ ≥ z" is satisfiable by some rule in the family; *reductive* — for two
classes it is **exactly** the trapezoidal AUC (verified to machine precision, §F1).

**B2 — Down-set of the convex hull (randomised rules allowed).** **[USED — secondary]**
`VUS_ch ≥ VUS`; the gap is head-room available from re-thresholding alone. First
implementation was **wrong** — see §G3.

**B3 — Rank / 3-alternative-forced-choice.** **[USED — primary secondary estimator]**
`P(the identity permutation maximises Π_k p_{σ(k)}(x_k))`. Threshold-free, a U-statistic
(so unbiased with a closed-form variance), and it is Mossman's own reading of the VUS.
Algebraic characterisation derived by hand (five conditions, PLAN §3.3) and then verified
against a brute-force six-permutation argmax (§F5).

**B4 — Ordinal `P(s₁ < s₂ < s₃)`.** **[USED — ordinal mode]** Nakas & Yiannoutsos.
Computed in `O(n log n)` by sweeping the middle class and multiplying "how many below" ×
"how many above"; tie-corrected with the usual ½ credit. Verified against brute force.

**B5 — Trinormal parametric VUS.** **[REJECTED]** Requires three Gaussians on a single
decision axis. Reasonable for tiny samples in a lab, wrong as a default for ML models
whose scores are neither Gaussian nor one-dimensional.

**B6 — Scurfield's information-theoretic `D`.** **[REJECTED]** Elegant (it is the channel
capacity of the n-interval forced-choice observer) but not on a `[0,1]` scale with
perfect = 1, which was an explicit requirement.

**B7 — Chance correction.** **[USED]** `VUS_adj = (VUS − 1/6)/(1 − 1/6)`. Necessary
because 1/6 is an unintuitive floor: a VUS of 0.5 is a *decent* 3-class model, not a coin
flip. Reported next to the raw value everywhere. This is the 3-class analogue of
Gini = 2·AUC − 1.

**B8 — Partial VUS on a box `[ℓ₁,1]×[ℓ₂,1]×[ℓ₃,1]`.** **[USED]** The analogue of partial
AUC, for when only the high-sensitivity corner matters. Normalised by the box volume so
"can meet everything in the box" still scores 1.

**B9 — Volume under the surface measured from `(1,1,1)` instead of the origin.**
**[REJECTED]** Would make *smaller* better and break the AUC analogy. Mentioned only
because "distance to the perfect corner" survives as a *threshold-selection criterion*
(`closest_to_perfection`), which is the right place for it.

---

## C. Computation

**C1 — Grid over the rule family.** **[USED]** The rule changes only when `a`, `b` or
`b−a` crosses one of `3n` critical values, so a uniform grid wastes resolution. Cut points
are placed strictly *between* consecutive distinct data values (quantile-thinned to the
requested resolution), plus one far outside each end — which makes the three degenerate
corners ("always predict k", operating points `e₁,e₂,e₃`) automatically reachable and
makes every strict inequality unambiguous.

**C2 — Fast sensitivity evaluation by 2-D cumulative histograms.** **[REJECTED after
analysis]** `S₁` is a 2-D dominance count in `(D₁₂, D₁₃)` and *is* histogram-able on the
`(a,b)` grid in `O(n + G)`. But `S₂` needs `(D₁₂, D₂₃)` against `(a, b−a)` and `S₃` needs
`(D₁₃, D₂₃)` against `(b, b−a)` — sheared lattices. Re-parameterising to `(a, c=b−a)` makes
`S₂` cheap and breaks `S₁`. The problem is symmetric: no single Cartesian grid makes all
three cheap. Abandoned in favour of C3.

**C3 — Chunked direct evaluation.** **[USED]** `O(n·G)` with a memory-bounded chunk loop.
Measured: 410 881 rules × 900 samples in **1.2 s**; 25 921 rules in 0.08 s. Fast enough
that the cleverness in C2 would have bought nothing but bugs.

**C4 — Full 3×3 confusion per rule instead of just the diagonal.** **[USED]** Costs ~2×
(six comparisons per sample-rule instead of three) and stores `9G` int64s (≈1 MB at
G = 26k). Buys: arbitrary cost matrices as a selection criterion, off-diagonal diagnostics
("at this operating point, how often is malignant called benign?"), and the
operating-point card. Clearly worth it.

**C5 — Exact 3-D hypervolume by dimension sweep.** **[USED]** Sort by `S₃` descending;
maintain the 2-D staircase of `(S₁,S₂)` incrementally with `bisect`; accumulate
`area × Δz`. Every point is inserted and deleted at most once, so it is `O(N log N)` plus
C-level list splices. Cross-checked three ways (§F2).

**C6 — Pareto pre-filter.** **[USED where it helps]** Not needed for correctness of C5
(the staircase drops dominated points on its own) but required for C7 and for plotting.
Implemented as an `O(n log n)` sweep reusing the same staircase structure, with a
coordinate shift so that zero coordinates still take part in the domination test.

**C7 — Convex-hull VUS.** **[USED]** Proof that made it tractable: the down-set of
`conv(S)` equals `conv(⋃ boxes)`, because if `q ≤ Σλᵢpᵢ` then setting
`t_j = q_j/(Σλᵢp_{i,j})` and `qᵢ = (t₁p_{i,1}, t₂p_{i,2}, t₃p_{i,3})` gives `qᵢ ≤ pᵢ` with
`Σλᵢqᵢ = q`. So the region is convex and its volume is just
`ConvexHull(all 8 vertices of every Pareto box).volume`.

**C8 — Monotone envelope `Z(u,v)` by scatter-max + reverse cumulative max.**
**[USED]** `O(N + R²)`. Serves double duty: it is the surface that gets plotted, and
`mean(Z)` is an independent estimate of the VUS. At R = 801 it agrees with the exact
hypervolume to `1e-5` (§F4).

**C9 — 3AFC: exact vs pruned vs Monte Carlo.** **[ALL THREE USED]**
`exact` does a tie-aware argmax over the six permutation scores for all `n₁n₂n₃` triples;
`exact_fast` uses the five algebraic conditions and prunes on the two that involve only
two indices (typically 5–20× faster, but counts ties as losses); `mc` samples triples.
`auto` picks `exact` under 2·10⁶ triples and `mc` above. All three agree (§F5).

**C10 — Generic HUM for `c > 3`.** **[USED]** The winning permutation is the solution of a
linear assignment problem; enumerated for `c ≤ 7`, `scipy.optimize.linear_sum_assignment`
above. Chance level `1/c!`. There is no surface past `c = 3` — that is a geometric fact,
not a missing feature — but the scalar survives, and at `c = 2` it is exactly the AUC.

---

## D. Visual design

**D1 — Parametric `trisurf` of `{(S₁,S₂,S₃)(w)}`.** **[REJECTED as the main panel]**
Faithful to Mossman but the parametric surface can fold, and matplotlib's painter's
algorithm renders folds as garbage.

**D2 — The monotone envelope as the 3-D panel.** **[USED]** `Z(u,v)` is a genuine graph,
monotone by construction, and the volume under it *is* the VUS — so the picture and the
number are the same object rather than two related ones.

**D3 — Translucent chance plane in 3-D.** **[REJECTED]** z-fighting with the surface
produced a spurious dark wedge that read as a feature of the data. Replaced by the
outline of the triangle `e₁–e₂–e₃` (the intersection of `ΣS = 1` with the unit cube) —
one thin line, no ambiguity, no depth conflict.

**D4 — Ternary map of the weight simplex.** **[USED — the panel that does the work]**
The dual of the surface: every point is a `w`, filled with the chosen objective. This is
where a user actually picks a rule. Evaluated on a regular barycentric lattice
(`lattice_surface`) rather than the quantile grid, so the contours are clean; log ratios
are clipped to the range in which the rule can still change, so the lattice corners really
are the "always predict k" rules.

**D5 — Iso-sensitivity contours overlaid on the objective fill.** **[REJECTED, split
into two panels]** The fill uses the sequential blue ramp and the class contours use
categorical slots 1–3, the first of which is also blue. Overlaying them put blue on blue.
Split into panel B (objective fill) and panel C (three contour families on a plain
surface): each panel then does exactly one job.

**D6 — Topographic contour map of `Z(u,v)`.** **[USED]** The readable twin of the 3-D
panel — you cannot get numbers off a rotated surface, but you can off a contour map.

**D7 — Iso-slice frontiers.** **[USED]** For `z ∈ {0.70, 0.80, 0.90, 0.95}`, the achievable
`(S₁,S₂)` staircase subject to `S₃ ≥ z`. Directly answers "if I must guarantee the
dangerous class, what do I give up?". Drawn as staircases, not smoothed curves, because
the achievable set really is a staircase.

**D8 — Six levels of iso-contour per class.** **[REJECTED]** 18 labelled contour families
in one triangle is unreadable. Cut to three levels (0.60/0.80/0.90) with sparse inline
labels.

**D9 — Five-step ordinal ramp for the iso-slice levels.** **[REJECTED — failed the
validator]** The step-separation check demands OKLCH ΔL ≥ 0.06 between adjacent steps;
the natural five steps `#86b6ef,#5598e7,#2a78d6,#256abf,#184f95` gave a minimum ΔL of
**0.048** (the 450↔500 pair). Dropped to four steps `#86b6ef,#5598e7,#2a78d6,#184f95`,
minimum ΔL **0.093** — passes. Hence four iso-levels, not five.

**D10 — Palette validation.** **[USED, with a substitution]** The bundled
`validate_palette.js` could not run — no JS runtime in this environment. Rather than
eyeball it, the checks were **ported to Python** (same Machado–Oliveira–Fernandes 2009
severity-1.0 matrices, same OKLab ΔE×100 metric, same thresholds) and run. Results for
the three class hues at *all* pairs:
| mode | lightness | chroma | CVD ΔE | normal-vision ΔE | contrast |
|---|---|---|---|---|---|
| light `#2a78d6 #eb6834 #1baf7a` | pass | pass | 9.2 (deutan) | 24.0 | **relief**: aqua 2.74:1 |
| dark `#3987e5 #d95926 #199e70` | pass | pass | 9.4 (deutan) | 20.9 | pass |
The light-mode aqua sits below 3:1, so the relief rule applies: every panel using it
ships visible direct labels (inline contour labels, endpoint labels on the price curve)
and the summary text panel is the table view.

**D11 — Interactive matplotlib widget with sliders.** **[REJECTED]** Does not survive
being saved to a file, which is how these figures get shared. Replaced by a self-contained
plotly HTML (`interactive_surface`) where hovering a point shows `w` and the full
sensitivity triple.

**D12 — Dark mode.** **[USED]** Separately stepped, not an automatic inversion:
`THEMES["dark"]` takes the dark column of the categorical palette and reverses the
sequential ramp. Validated against the dark surface independently.

**D13 — Cost-space dual (Drummond–Holte style).** **[DEFERRED]** For three classes the
cost space is also 2-D, so the ternary weight map already plays this role. Would be worth
revisiting if per-pair costs (outside the EEU family) ever get supported.

---

## E. Uncertainty

**E1 — Closed-form U-statistic variance for the 3AFC VUS.** **[USED]** The Hoeffding
projections `ζ₁,ζ₂,ζ₃` are the variances of the three sets of conditional means, and
those come out of the triple sweep for free — accumulate the row/column/tube sums of the
same indicator that produces the point estimate. No resampling. Agrees with the bootstrap
(§F6).

**E2 — Stratified bootstrap.** **[USED]** Resample within class. The general-purpose
fallback; also the only practical route for the geometric VUS, which is not a U-statistic.

**E3 — Permutation test of `H₀: VUS = 1/6`.** **[USED]** Shuffle labels. Verified to sit
at 1/6 under the null and not to reject (§F6).

**E4 — Confidence region for the operating point itself.** **[USED]**
`bootstrap_operating_point` holds the *rule* fixed and resamples the data, which is the
honest uncertainty of what you deploy. Documented explicitly as *not* including the
selection optimism from having chosen `w` on the same sample.

**E5 — Paired bootstrap for a difference in VUS between two models.** **[USED in the case
study]** Both models scored on the same resampled cases so the shared sampling noise
cancels. On the demo data it separates LDA (0.673) from a random forest (0.645) with a
95% CI of (−0.048, −0.009), which two independent intervals would have called a tie.

---

## F. Empirical findings

**F1 — The binary reduction is exact.** Running the identical construction (sweep the
rule family, collect operating points, take the volume of the down-set) on two-class
problems reproduces `sklearn.metrics.roc_auc_score` with **max |diff| = 2.2 × 10⁻¹⁶** over
eight random problems. `HUM` at `c = 2` likewise matches the AUC. This is the strongest
single piece of evidence that the generalisation is the right one.

**F2 — The hypervolume kernel is right.** Exact against hand-computed inclusion–exclusion
cases; within 4 SE of Monte-Carlo dominance sampling on random clouds; unchanged by Pareto
pre-filtering. A dense sample of the chance plane gives 0.1654 against the true 1/6 =
0.1667 (staircase, converges from below).

**F3 — Grid convergence is `O(1/R)` from below.**
| R | rules | VUS | gap to 3AFC | time |
|---|---|---|---|---|
| 20 | 441 | 0.64984 | −0.0333 | 0.00 s |
| 40 | 1 681 | 0.66536 | −0.0178 | 0.01 s |
| 80 | 6 561 | 0.67472 | −0.0084 | 0.02 s |
| 160 | 25 921 | 0.67810 | −0.0051 | 0.08 s |
| 320 | 103 041 | 0.67995 | −0.0032 | 0.32 s |
| 640 | 410 881 | 0.68084 | −0.0023 | 1.18 s |
Successive gaps halve as `R` doubles. Default `resolution=160` therefore costs about
0.005 of VUS; the demos use 240–300. Richardson extrapolation of the tail lands within
0.0015 of the rank estimate.

**F4 — The gridded envelope integrates to the exact hypervolume.** At R = 801,
`mean(Z) = 0.88752` vs exact `0.88751`. So the picture in panel A and the number in the
title are the same quantity, not two approximations of it.

**F5 — The two VUS definitions agree — including for mis-calibrated scorers.**
This was the main open question. Scurfield/Mossman prove geometric = 3AFC for the *ideal
observer*; nothing guarantees it otherwise.
* Ideal observer (true posteriors), separations 0.4 → 2.4 at R = 400: max |geom − 3AFC| =
  **0.0012**, always with geom slightly *below* (the residual grid bias of F3).
* Deliberately mis-calibrated scorers (random temperature `p^α`, `α ∈ [0.4,2.5]`; random
  per-class rescaling; an extra monotone warp on one column): max |geom − 3AFC| =
  **0.0016**.
So within grid error the two coincide even outside the regime the theorem covers. That is
worth stating as an observation, not a theorem — the experiment is a handful of
distribution families, not a proof.

**F6 — Inference is calibrated.** Closed-form SE 0.0242 vs stratified-bootstrap SE 0.0229
on the same data (within 6%). The permutation null centres on 0.1685 ≈ 1/6 and does not
reject on a null model (p = 0.34).

**F7 — The estimators separate at the extremes exactly as required.**
perfect → 1.00000, adversarial (cyclic) → 0.00000, uninformative → 0.186 at n = 400/class
(the small positive bias is the usual finite-sample optimism of a max-over-rules
statistic; it shrinks with n and the permutation test accounts for it).

**F8 — Criteria genuinely disagree, which is the point.** On the demo problem
(600/400/200, benign/indeterminate/malignant):
| criterion | Se benign | Se indet | Se malig |
|---|---|---|---|
| youden ≡ balanced accuracy | 0.798 | 0.445 | 0.830 |
| closest to perfection | 0.730 | 0.615 | 0.690 |
| maximin | 0.692 | 0.665 | 0.655 |
| accuracy | 0.862 | 0.525 | 0.625 |
| max volume `S₁S₂S₃` | 0.727 | 0.548 | 0.780 |
`youden` and `balanced_accuracy` are affine transforms of each other and always pick the
same rule — documented rather than hidden, since both names are in common use.

**F9 — The "price of a guarantee" is not linear.** Requiring `Se[malignant] ≥ z` is
**free up to z ≈ 0.85** (the balanced-accuracy optimum already delivers 0.865) and then
bites hard: at z = 0.975 the indeterminate class collapses from 0.40 to 0.23 while benign
barely moves. The `w[malignant]` multiplier has to rise by ~15× across that range. This is
the kind of thing the surface exists to reveal and a single scalar cannot.

---

## G. Bugs found and fixed

**G1 — Broken incremental area update in the staircase.** The first version of
`_Staircase.add` forgot that inserting a point changes the *successor's* predecessor, and
contained a dead branch. Rewritten with an explicit successor fix-up
(`area += ys[i]·(old_prev − x)`) and verified against hand-computed inclusion–exclusion.

**G2 — Ties resolved inconsistently.** The partition of the `(a,b)` plane used strict
inequalities throughout, so a fully tied sample matched none of the three wedges, while
`OperatingSurface.confusion_at` used `np.argmax` (lowest index wins). Fixed by orienting
the inequalities as `≥ / < , ≥` to match `argmax` exactly. For continuous scores the grid
never lands on a data value so this is invisible; for degenerate scorers it matters.

**G3 — Convex-hull VUS computed the wrong thing.** The first implementation took the
*hypervolume of the hull's vertices*, which is the staircase over those vertices, not the
volume of the hull's down-set — an underestimate. Fixed with the argument in C7.

**G4 — Constant scorer: the estimators legitimately disagree.** Not a bug, but it looked
like one. With `p ≡ (⅓,⅓,⅓)` the staircase VUS is **0** while the 3AFC VUS and the
convex-hull VUS are both **1/6**. This is the exact 3-class echo of the binary tie
convention: deterministic thresholding of a constant score reaches only the three corner
rules (whose down-set has zero volume), whereas Mann–Whitney/AUC gives ties ½ credit and
so returns 0.5. Resolution: keep the staircase as primary because it reproduces the AUC
exactly when there are no ties (F1), and always report the hull and rank values beside it.
Documented in `PLAN.md` §7.

**G5 — `np.math.factorial` no longer exists** in NumPy 2.x. Replaced with a local helper.

**G9 — The probability floor was silently destroying dynamic range.**
`prepare_scores` clipped probabilities at `1e-12` before taking logs. For ordinary models
that is invisible; it surfaced only when the ordered-gauge work
([`CONSTRAINED.md`](CONSTRAINED.md)) started pushing one column far down, at which point
the *provable* class-rescaling invariance failed numerically — VUS moved from 0.9971 to
0.8637 under a transformation that cannot change it. The floor is now `1e-300`, which
keeps every float64-representable probability intact. Two lessons: a theorem that must
hold exactly is the sharpest possible test of the numerics, and a clipping constant
chosen for "safety" is a silent lossy compression.

**G6 — Wine with all thirteen features is separable.** VUS ≈ 0.99, surface pinned to the
perfect corner, nothing to look at. Switched the demo to two features (`alcohol`,
`alcalinity_of_ash`, VUS ≈ 0.81) — still real, honest out-of-fold data, but with a surface
that has structure. `features=None` restores the full set.

**G7 — Class labels vs class indices.** `prepare_scores` requires the values in `y` to
appear in `classes`; passing display names as `classes` alongside integer `y` raised. Left
as a hard error (silently guessing would be worse) and the demos map labels explicitly.

**G8 — `OrdinalOperatingPoint` lacked `.rates`,** so the shared operating-point card
crashed on the ordinal path. Added, and the card now branches on whether the rule is a
weight vector or a pair of cut-points.

---

---

## I. The logical constraint `0 < p₁ < p₂ < p₃ < 1`

Full write-up in [`CONSTRAINED.md`](CONSTRAINED.md); this is the idea-by-idea log.

**I1 — Read the sentence two ways before doing any maths.** **[USED]** "The probabilities
must satisfy…" can constrain the *model's output* or the *true posterior*. The two give
opposite answers (nothing changes / everything changes), so the first deliverable was the
disambiguation, not a formula. A diagnostic distinguishes them empirically: an ordered
true posterior forces `π₁<π₂<π₃`, so unordered prevalences prove you are in the
output-only reading.

**I2 — Output-ordering is a gauge.** **[USED — the main negative result]** Class-wise
rescaling `p_k → c_k p_k` is absorbed into the rule's weights, so it cannot move the
surface. Choosing the shifts as `−(max_x A + m)`, `−(max_x C + m)` orders *every* row.
Hence every model has an ordered twin with identical VUS. Verified bit-for-bit.

**I3 — Temperature scaling as part of the gauge.** **[REJECTED as unnecessary]** I first
thought the ordered gauge needed `p → p^ε` as well as a class rescaling, to shrink the
spread before shifting. It does not — a shift alone suffices, which makes the theorem
cleaner (one class rescaling, nothing else). Temperature would only be needed if the goal
were a *bounded* log-odds spread, which it is not.

**I4 — "The constraint must reduce the dimension."** **[REJECTED]** The order chamber is
an open subset of the simplex, still 2-D. No degrees of freedom are lost. (It *would* be
1-D if the outputs were additionally forced onto a curve, e.g. `p ∝ softmax(θ·(1,2,3))` —
that collapses to the ordinal single-marker mode of `roc3.ordinal`. Worth noting as a
separate constraint, not implied by this one.)

**I5 — "The chance level must change because the output space is 1/6 of the simplex."**
**[REJECTED — tested]** No. If the scores are independent of the label, all six
permutations are equally likely by exchangeability whatever region they live in. Measured
0.172 at n = 400/class.

**I6 — The master lemma: integrate the pointwise constraint over a decision region.**
**[USED — the key move]** `π_i f_i < π_j f_j` integrated over `R_m` gives
`π_i C[i,m] ≤ π_j C[j,m]`. Nine linear inequalities that bind *every* classifier. Once
this was written down everything else fell out as corollaries.

**I7 — Bounded likelihood ratio ⟹ bounded AUC.** **[USED]** A change-of-measure argument
gives `AUC ≤ 1 − 1/(2R)` for `L = f_i/f_j ≤ R`, tight. Under the constraint `R = π_j/π_i`.
Derived independently of I6 — and the two agree exactly in the binary case
(`1 − π_i/(2π_j)`, verified to 9 decimals), which is what made me trust both.

**I8 — Computing the ceiling by grid + LP.** **[USED as a cross-check]** `R²` tiny LPs.
Slower than I9 but produces the ceiling as a *surface*, which is what the figure needs.

**I9 — Computing the ceiling exactly by vertex enumeration.** **[USED — primary]** The
feasible set is a polytope in 6 free variables with 18 facets; enumerate `C(18,6)` =
18 564 six-subsets, solve, keep feasible points, project onto the diagonal, take the
down-set volume of the hull (reusing `convex_hull_vus`). Exact, no resolution parameter.
Agrees with I8 to `1e-5`.

**I10 — Is the ceiling attainable by a single model?** **[OPEN — measured]** No, not
exactly: the ceiling ranges over all classifiers on all worlds with the given priors,
whereas one world exposes only a 2-parameter Bayes family. Constructed worlds captured
37% → **89.3%** of the attainable volume as the posterior distribution was enriched.
Concentrating mass near the chamber's extreme points beats a uniform lattice by 36 points,
which is what the LP predicts (the optimum is at vertices). 80 random constrained worlds
all respected the ceiling.

**I11 — The order chamber's vertices are the whole intuition.** **[USED]**
`T = {p₁≤p₂≤p₃}` is the triangle with vertices `(⅓,⅓,⅓)`, `(0,½,½)`, `(0,0,1)`. So the
constraint says: *you can never be more than 1/3 sure of class 1, or more than 1/2 sure of
class 2.* Every quantitative result is that sentence in another form. This also fixed a
buggy first attempt at generating constrained worlds — parameterising the chamber
barycentrically by these three vertices makes ordered posteriors automatic, where my
hand-rolled formula produced violations.

**I12 — Strictness ties.** **[FINDING]** Two of the three chamber vertices have ties
(`p₁=p₂=p₃` and `p₂=p₃`), so a uniform shrink toward the interior does *not* produce
strict ordering. The offset has to be increasing in `k`: `p·(1−6ε) + ε·(1,2,3)`.

**I13 — Renormalise the index.** **[USED — the practical payoff]**
`VUS_normalized = (VUS − 1/6)/(VUS_max(π) − 1/6)`. On the worked example a model scoring
0.550 — mediocre against 1.000 — is capturing **86.6%** of everything attainable.

**I14 — Degenerate case as a consistency check.** **[FINDING]** If `π_i = π_j`, the lemma
forces `C[i,·] = C[j,·]`: the two classes are indistinguishable to every classifier and
the ceiling collapses to chance. An ordered posterior with balanced priors is not a hard
problem, it is an impossible one — which is exactly why the §0 diagnostic works.

**I15 — Ceilings on the scalar metrics too.** **[USED]** `max accuracy = π₃` in closed
form (the constant rule is Bayes-optimal and unbeatable — verified against the LP to 6
decimals); balanced accuracy and worst-class sensitivity by LP.

---

## H. Open questions / what I would do next

1. **Is geometric VUS = 3AFC VUS exactly, for any scorer?** F5 says the gap is under
   0.002 across everything tried, and all of it is explainable as grid bias. A proof (or a
   counterexample) would settle whether `exact_fast` can simply replace the grid for the
   headline number.
2. **A cheap exact algorithm for the geometric VUS** — enumerate the cells of the line
   arrangement in the `(a,b)` plane (`O(n²)` cells) instead of gridding. Removes the
   resolution parameter entirely. Worth it only if question 1 is answered "no".
3. **Costs outside the EEU family.** The selection criterion already handles a full 3×3
   cost matrix by reading the stored off-diagonals, but the *surface* only covers the EEU
   rule family. The extra rules live on the degenerate object of A4; a principled 2-D
   summary of them is, as far as the literature goes, still open.
4. **Umbrella orderings** (Nakas 2007) for the ordinal mode — where the marker is not
   monotone in severity.
5. **AUCμ** in the comparator table (A3).
6. **Selection optimism.** `bootstrap_operating_point` holds the rule fixed. A nested
   bootstrap that re-selects `w` inside each replicate would quantify how much of a chosen
   operating point is wishful thinking; expensive but straightforward.
