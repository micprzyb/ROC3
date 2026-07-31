# A 3-Dimensional Analogue of the ROC Curve and AUC for 3-Class Classifiers

**Design document / research plan.**
Companion files: [`IDEAS_LOG.md`](IDEAS_LOG.md) (every idea tried, incl. dead ends and numbers),
[`REFERENCES.md`](REFERENCES.md) (annotated bibliography).

---

## 0. Problem statement

Given a 3-class probabilistic classifier producing scores `P ∈ R^{n×3}` and true labels
`y ∈ {0,1,2}^n`, we want:

1. **A 2-D surface embedded in 3-D** that plays the role the ROC curve plays for binary
   problems: it must display the *joint* trade-off between the three "cut-off" choices
   and let a practitioner *pick an operating point*.
2. **A scalar "volume under the surface" (VUS)** that summarises overall model quality,
   equal to **1 for a perfect model**.
3. Both must be *actionable*: reading a point off the surface must yield a concrete,
   deployable decision rule.

---

## 1. Reverse-engineering the binary ROC: which properties must survive?

Before generalising we list what the binary ROC/AUC actually *is*, so we can check the
generalisation against each property.

| # | Property of binary ROC/AUC |
|---|---|
| **P1** | **Axes = achievable trade-off coordinates.** One axis per "kind of correctness": TPR = Se₁ (sensitivity for the positive class), 1−FPR = Se₀ (specificity). |
| **P2** | **The curve is the image of a 1-parameter family of decision rules** — the threshold sweep `ŷ = 1[s(x) > t]`. dim(parameter space) = 1 = c−1 for c=2. |
| **P3** | **Every point on the curve ↔ a concrete deployable rule** (a value of `t`). This is what makes it a *threshold-selection* tool. |
| **P4** | **The curve is the upper boundary (Pareto frontier) of the achievable set**; the region under it is a *down-set* (if `(f,t)` is achievable then everything componentwise worse is too). |
| **P5** | **AUC = volume of that down-set**, and it has two equivalent readings: <br>(a) *geometric*: `P((U,V) ~ Unif[0,1]² is dominated by some achievable (Se₀,Se₁))` — "the chance that a randomly demanded pair of requirements is satisfiable"; <br>(b) *rank / forced-choice*: `P(s(X⁺) > s(X⁻))` — probability of correctly ordering a random positive/negative pair — the **2-alternative forced choice (2AFC)** probability. |
| **P6** | **Scale:** perfect = 1, chance = 1/2 = 1/2!, adversarial → 0. |
| **P7** | **Invariance:** AUC does not change under class re-weighting (prior shift) or any strictly monotone rescaling of the score. |
| **P8** | **Convex-hull reading:** randomised rules fill in the hull; area under hull ≥ AUC, and the gap measures how much a better *thresholding* (not a better model) could buy. |

Any candidate 3-class construction is graded against P1–P8 below.

---

## 2. The design space — every approach considered

### A1. Stack three one-vs-rest ROC curves
Plot three ordinary 2-D ROCs (class k vs rest). Summary = macro-average OvR AUC.

*Pros:* trivial, familiar, `sklearn` supports it.
*Cons:* **fails P1/P2/P3 jointly.** Three independent thresholds `t₁,t₂,t₃` do not define a
classifier — a point may exceed two thresholds or none. It cannot express the *coupling*
between the three decisions, which is exactly what a 3-class trade-off is. It is also
prior-dependent (the "rest" class is a mixture).
**Verdict: rejected as the answer; kept as a context/diagnostic panel.**

### A2. Hand & Till M (macro pairwise / one-vs-one AUC)
`M = 2/(c(c−1)) · Σ_{i<j} Â(i,j)`, `Â(i,j) = ½(A(i|j) + A(j|i))`.

*Pros:* prior-insensitive (its selling point), perfect = 1, chance = 1/2, cheap.
*Cons:* **scalar only — there is no surface** (fails P1–P4). Chance = 1/2 not 1/6 means it
is *not* the volume of anything in 3-D. Insensitive to genuinely 3-way confusions
(a model that is pairwise-good can still be 3-way-bad).
**Verdict: rejected as the answer; reported as a reference number.**

### A3. AUCμ (Kleiman & Page, ICML 2019)
Partition-based multiclass AUC: for each ordered class pair it projects the probability
vector onto a separating direction and computes a binary AUC there, then averages.

*Pros:* single scalar in [0,1], chance = 1/2, handles class imbalance, respects the simplex
geometry (unlike A2 it uses the *whole* probability vector).
*Cons:* still a scalar, no surface, no threshold read-off.
**Verdict: reference number only.**

### A4. The *full* 3-class ROC hypersurface
The 3×3 row-stochastic confusion matrix has `c(c−1) = 6` free off-diagonal entries. The
unrestricted family of Bayes rules (arbitrary utility matrix `U ∈ R^{3×3}`) has
`c(c−1) − 1 = 5` free parameters, so in principle one sweeps a **5-D manifold inside a 6-D
space** and takes its 5-D hypervolume.

*Pros:* the "honest" complete description.
*Cons:* **unplottable**, and Edwards & Metz (2012) show the manifold is *degenerate* — the
ideal-observer decision lines can intersect the likelihood-ratio curve in at most six
places, so the surface has **at most 4 degrees of freedom**, not 5, making the 5-D
hypervolume identically zero / ill-behaved as a figure of merit.
**Verdict: rejected (documented in IDEAS_LOG as the reason the EEU restriction is not a
cop-out but a necessity).**

### A5. ★ The Mossman / Scurfield ROC **surface** under Equal Error Utility — **CHOSEN**

Restrict the utility matrix by the **Equal Error Utility (EEU)** assumption: the cost of
misclassifying a class-`k` case depends only on `k`, not on which wrong label was chosen
(`U_{i|k} = u_k` if `i=k`, else `v_k`). Then the Bayes rule collapses to

```
ŷ(x; w) = argmax_k  w_k · p_k(x),      w ∈ Δ² = {w ≥ 0, Σw = 1},   w_k ∝ (u_k − v_k)
```

— a **2-parameter family** (only ratios of `w` matter). This is *exactly* the binary
threshold sweep when c = 2: `w₁p₁ > w₀p₀ ⟺ p₁ > t`, `t = w₀/(w₀+w₁)`.

The operating point is the triple of **true class fractions** (per-class sensitivities)

```
S_k(w) = P( ŷ(X; w) = k | Y = k ),        (S₁,S₂,S₃) ∈ [0,1]³
```

and sweeping `w` over `Δ²` traces a **2-D surface in the 3-D unit cube** — the answer to
requirement (1). Reference objects:

* **Chance plane** `S₁+S₂+S₃ = 1` (any uninformative rule lands there; a rule that assigns
  labels independently of `x` gives exactly `Σ S_k = 1`).
* **Perfect corner** `(1,1,1)`.

Checks: P1 ✓ (three "kinds of correctness"), P2 ✓ (c−1 = 2 parameters), P3 ✓✓ (`w` *is* the
deployable rule), P4 ✓ (take the down-set), P5 ✓ (§3.3), P6 ✓ (perfect 1, chance 1/6 = 1/3!),
P7 ✓ (§3.4), P8 ✓ (convex-hull variant).

**Sub-parameterisations considered:**

* **A5a — weighted argmax over `Δ²`** ⟺ EEU-Bayes. **Chosen as the default.** Directly
  interpretable ("multiply class-k probability by `w_k` before taking the argmax", i.e.
  class-specific cost/prior multipliers).
* **A5b — two cut-points on a single ordinal marker** (`ŷ = 1 if s<t₁; 2 if t₁≤s<t₂; 3 if
  s≥t₂`). This is Mossman's/Nakas' setting for *ordered* classes (mild < moderate < severe).
  **Chosen as a second mode** (`roc3.ordinal`), because for ordinal problems it is more
  natural and yields literal cut-off numbers `t₁, t₂`.
* **A5c — pairwise/tournament thresholds** (`c(c−1)/2 = 3` pairwise comparisons, majority
  vote). *Rejected:* the tournament can be intransitive (rock-paper-scissors cycles), so the
  rule is not well-defined without an ad-hoc tie-break, and the family has 3 parameters
  (surface would be over-parameterised for 3-D display).
* **A5d — arbitrary partitions of the simplex** (AUCμ-style). *Rejected:* not a 2-parameter
  family; no canonical sweep; not a surface.

### A6. Definitions of "the volume"

| id | definition | status |
|---|---|---|
| **V1** | **Down-set hypervolume**: `vol( ⋃_w [0,S₁(w)]×[0,S₂(w)]×[0,S₃(w)] )`. Equals binary AUC exactly when c = 2 (verified in §6). Interpretation: *probability that a uniformly random requirement triple "Se₁ ≥ u, Se₂ ≥ v, Se₃ ≥ z" is satisfiable by some setting of `w`.* | **PRIMARY** |
| **V2** | Down-set of the **convex hull** of achievable points = what randomised rules can reach. Analogue of "area under the ROC convex hull". `VUS_ch − VUS ≥ 0` measures head-room from better thresholding alone. | secondary diagnostic |
| **V3** | **Rank / 3AFC (`HUM`)**: probability that, given one case from each class in random order, the maximum-likelihood assignment picks the correct permutation. Threshold-free U-statistic ⇒ unbiased, closed-form variance. This is Mossman's reading of VUS ("probability of correctly sorting a trio"). | **PRIMARY (secondary estimator)** |
| **V4** | **Ordinal VUS** `P(s(X₁) < s(X₂) < s(X₃))` (Nakas & Yiannoutsos 2004) for the A5b mode. | ordinal mode |
| **V5** | **Trinormal parametric VUS** — assume three Gaussians, integrate analytically. *Rejected as default:* strong distributional assumption; useful only for small samples. | rejected (noted) |
| **V6** | Scurfield's information-theoretic `D` (channel capacity of the n-interval forced-choice observer). *Rejected:* not on a [0,1] scale with perfect = 1. | rejected (noted) |

### A7. Normalisations
* `VUS ∈ [0,1]`, perfect = 1, chance = **1/6 = 1/3!**, adversarial → 0.
* `VUS_adj = (VUS − 1/6) / (1 − 1/6) = (6·VUS − 1)/5` — chance 0, perfect 1
  (the 3-class analogue of Gini / Somers' D = 2·AUC − 1).
* **Partial VUS** over a box `[ℓ₁,1]×[ℓ₂,1]×[ℓ₃,1]`, normalised by the box volume — for when
  only the high-sensitivity corner matters (analogue of partial AUC).

### A8. Visualisation variants (all implemented; see §4)
1. Parametric `trisurf` of `{(S₁,S₂,S₃)(w)}` — faithful to Mossman, but can fold / be hard to read.
2. **Monotone envelope** `Z(u,v) = max{ S₃(w) : S₁(w) ≥ u, S₂(w) ≥ v }` — the surface *as a
   graph*, guaranteed monotone; `∫∫Z du dv = VUS` *literally*. **This is the primary 3-D panel.**
3. **Ternary weight map** — the *dual* picture: barycentric plot of `w ∈ Δ²` with iso-contours
   of `S₁,S₂,S₃`. This is the panel you actually read thresholds off.
4. **Iso-slices** — for fixed `S₃ ≥ z`, the achievable `(S₁,S₂)` frontier; a contour map of the surface.
5. Cost-space dual (Drummond–Holte style) — *considered, deferred*: for 3 classes the cost
   space is also 2-D and the ternary map (3) already serves the purpose.
6. Interactive HTML (plotly) with hover showing `w`, `(S₁,S₂,S₃)` and the confusion matrix.

---

## 3. The chosen algorithm, in full

### 3.1 Rule family and operating points

Let `ℓ_k(x) = log p_k(x)` (probabilities clipped to `[ε, 1]`). With `λ = log w` and only
differences mattering, set

```
a = log(w₂/w₁),   b = log(w₃/w₁)          (2 free parameters)
D₁₂(x) = ℓ₁−ℓ₂,   D₁₃(x) = ℓ₁−ℓ₃,   D₂₃(x) = ℓ₂−ℓ₃  =  D₁₃ − D₁₂
```

The rule `argmax_k (ℓ_k + λ_k)` partitions the `(a,b)` plane, **per sample**, into three
wedges meeting at the point `(D₁₂(x), D₁₃(x))`:

```
predict 1  ⟺  a < D₁₂(x)  and  b < D₁₃(x)
predict 2  ⟺  a > D₁₂(x)  and  b − a < D₂₃(x)
predict 3  ⟺  b > D₁₃(x)  and  b − a > D₂₃(x)
```

Hence

```
S₁(a,b) = (1/n₁) · #{ i ∈ C₁ : D₁₂ᵢ > a , D₁₃ᵢ > b }
S₂(a,b) = (1/n₂) · #{ j ∈ C₂ : D₁₂ⱼ < a , D₂₃ⱼ > b − a }
S₃(a,b) = (1/n₃) · #{ k ∈ C₃ : D₁₃ₖ < b , D₂₃ₖ < b − a }
```

**Grid.** The rule only changes when `a`, `b` or `b−a` crosses one of the `3n` critical
values. A uniform grid wastes resolution; we use **quantile grids** of the pooled `D₁₂` (for
`a`) and `D₁₃` (for `b`), padded with ±∞-surrogates so the three degenerate corners
("always predict k", operating points `e₁,e₂,e₃`) are always in the achievable set.
Cost: `O(n·G)` with `G = R²` grid points, evaluated in memory-bounded chunks.
Exactness w.r.t. grid resolution is validated empirically (see IDEAS_LOG §Grid stability).

### 3.2 Volume: exact 3-D hypervolume of the down-set

`VUS = vol( ⋃_g [0,S₁ᵍ]×[0,S₂ᵍ]×[0,S₃ᵍ] )` — this is exactly the **hypervolume indicator**
from multi-objective optimisation with reference point at the origin.

Algorithm (`O(N log N)` after Pareto filtering):
1. Keep only Pareto-maximal operating points (a point is dominated if another is ≥ in all
   three coordinates).
2. Sort by `S₃` descending; sweep. Between consecutive `S₃` levels the cross-section is the
   union of origin-anchored rectangles `⋃[0,S₁]×[0,S₂]` over the already-processed points;
   maintain that 2-D staircase incrementally (insert, drop dominated, update area by delta).
3. `VUS = Σ_m Area(staircase after m points) · (S₃₍ₘ₎ − S₃₍ₘ₊₁₎)`, last slab down to 0.

Cross-checked against (i) Monte-Carlo dominance sampling and (ii) integration of the
gridded envelope `Z(u,v)`.

### 3.3 Volume: rank-based 3AFC estimator

Draw one case from each class, `x₁∈C₁, x₂∈C₂, x₃∈C₃`, present them in unknown order, and
assign the labels `{1,2,3}` bijectively to maximise `Π_k p_{σ(k)}(x_k)`. Writing
`A = ℓ₁−ℓ₂`, `B = ℓ₁−ℓ₃`, `C = ℓ₂−ℓ₃` (so `B = A + C`), the identity permutation wins iff
**all five** of

```
(1) A(x₁) > A(x₂)                (swap 1↔2)
(2) C(x₂) > C(x₃)                (swap 2↔3)
(3) B(x₁) > B(x₃)                (swap 1↔3)
(4) A(x₁) + C(x₂) > B(x₃)        (3-cycle 1→2→3)
(5) B(x₁) > A(x₂) + C(x₃)        (3-cycle 1→3→2)
```

`VUS₃AFC = P(all five)`, estimated by the U-statistic over all `n₁n₂n₃` triples (exact) or
Monte-Carlo. Perfect ⇒ 1, uninformative ⇒ 1/6 (exactly one of the 6 permutations wins
a.s., and by symmetry each with prob. 1/6). The 6 permutation probabilities sum to 1 and
form a useful **"3AFC confusion profile"** diagnostic.

### 3.4 Invariances (P7)
`VUS₃AFC` is invariant to
* **temperature scaling** `p ↦ p^α` (all five conditions are homogeneous of degree 1), and
* **class re-weighting / prior shift** `p_k ↦ c_k p_k` (each condition's two sides shift by the
  same constant — verified algebraically, e.g. (4) shifts by `log(c₁/c₃)` on both sides).

The geometric `VUS` is invariant to prior shift *by construction* (it is a max over all `w`)
and to any monotone transform that preserves the ordering of the `D`'s.

### 3.5 Inference
* **Stratified bootstrap** (resample within each class) → percentile CI for any of the
  estimators, and for any derived operating point.
* **Closed-form U-statistic variance** for `VUS₃AFC` (Hoeffding projection):
  `Var ≈ ζ₁₀₀/n₁ + ζ₀₁₀/n₂ + ζ₀₀₁/n₃` where `ζ₁₀₀ = Var_i(mean_{j,k} ψ_{ijk})`, etc.
* **Permutation test** of `H₀: VUS = 1/6` by shuffling labels.

---

## 4. The plot suite (what makes it *useful for choosing thresholds*)

A 3-D render alone is a poor decision tool. The deliverable is a **6-panel dashboard**:

| Panel | Content | Question it answers |
|---|---|---|
| **A** | 3-D ROC surface = graph of the monotone envelope `Z(u,v)`, with the chance plane `ΣS=1`, the perfect corner, the plain-argmax point, and the shaded volume. Title carries VUS. | *How good is the model overall, and what shape is the trade-off?* |
| **B** | **Ternary weight map**: `w ∈ Δ²` filled-contoured by a chosen objective, overlaid with iso-`S₁`/`S₂`/`S₃` contours; optimum + default marked. | ***Where do I set my thresholds?*** |
| **C** | **Iso-slices**: for `z ∈ {0.7,0.8,0.9,0.95}` the achievable `(S₁,S₂)` frontier subject to `S₃ ≥ z` (and the two rotations). | *If I must guarantee class-3 sensitivity, what do I give up?* |
| **D** | **Topographic map** of `Z(u,v)` with filled contours (the same object as A, read as a contour map — much easier to read numbers off). | *Given requirements on classes 1 and 2, what is the best class-3 sensitivity?* |
| **E** | Reference panel: three OvR ROC curves + three OvO ROC curves with AUCs; metric table (VUS, VUS_adj, VUS₃AFC, VUS_ch, Hand–Till M, macro-OvR AUC, accuracy, balanced accuracy). | *Sanity / comparison with familiar numbers.* |
| **F** | **Operating-point card** at the selected `w`: the 3×3 confusion matrix (counts + row-normalised), per-class Se/PPV/F1, and the rule written out explicitly. | *What exactly do I deploy, and what will it do?* |

Plus an **interactive plotly HTML** where hovering a surface point shows `w`, `(S₁,S₂,S₃)`
and the confusion matrix.

---

## 5. Threshold-selection criteria implemented

| criterion | objective |
|---|---|
| `youden` (Generalised Youden Index, Nakas et al.) | maximise `S₁+S₂+S₃−1` |
| `balanced_accuracy` | maximise `(S₁+S₂+S₃)/3` — *note: same argmax as `youden`; both provided for naming familiarity, and the equivalence is documented rather than hidden* |
| `closest_to_perfection` | minimise `‖(1,1,1) − (S₁,S₂,S₃)‖₂` |
| `maximin` | maximise `min_k S_k` (egalitarian / worst-class guarantee) |
| `expected_utility` | maximise `Σ_k π_k Σ_i U_{i|k} P(ŷ=i|k)` for a user cost matrix and priors |
| `accuracy` | maximise `Σ_k π̂_k S_k` (prevalence-weighted) |
| `constrained` | maximise any of the above subject to `S_k ≥ ℓ_k` |

---

## 6. Validation plan

1. **Perfect model** ⇒ VUS = 1 (all estimators).
2. **Uninformative model** (scores independent of label) ⇒ VUS → 1/6.
3. **Adversarial / cyclic model** ⇒ VUS → 0.
4. **Monotonicity**: VUS increases monotonically with class separation in a controlled
   Gaussian simulation.
5. **Binary reduction**: run the machinery on 2 classes ⇒ `VUS == sklearn roc_auc_score`
   to numerical precision.
6. **Estimator agreement**: geometric `VUS` vs `VUS₃AFC` across many random models — quantify
   agreement and characterise where they diverge (they answer subtly different questions;
   see IDEAS_LOG).
7. **Grid stability**: VUS vs grid resolution `R` — must converge.
8. **Hypervolume correctness**: exact sweep vs Monte-Carlo vs gridded-envelope integration.
9. **Ordinal mode**: `VUS_ordinal == P(s₁<s₂<s₃)` computed by brute force.

---

## 7. Known limitations (stated up front)

* The surface describes the **EEU-restricted** family of rules. A cost structure that
  distinguishes *which* wrong class was chosen (e.g. confusing 1↔3 is worse than 1↔2) is
  outside this family; those rules live on the 5-D object of A4. We expose the full
  confusion matrix at each operating point so the user can inspect off-diagonal structure,
  and document this restriction rather than hiding it.
* VUS is a *volume*, so it grows/shrinks faster than AUC with signal; a VUS of 0.5 is a
  fairly good 3-class model (chance is 1/6), not a coin flip. Always report `VUS_adj`
  alongside.
* Both VUS flavours are insensitive to **calibration** (they are max-over-`w` / rank
  quantities). They measure *ranking/separability*, not calibration — as with binary AUC.
* With `k` classes the plot stops at `k = 3`; the rank-based `HUM` generalises (chance
  `1/k!`) but is not visualisable. This is a hard geometric limit, not an implementation gap.
