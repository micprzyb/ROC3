# ROC analysis under the logical constraint `0 < p₁ < p₂ < p₃ < 1`

**Research note, written as a tutorial.** Every symbol is defined where it first appears
and every claim is worked on one small example you can check by hand. Regenerate all the
numbers with

```bash
.venv/bin/python experiments/06_constrained_numbers.py     # the worked example
.venv/bin/python experiments/04_constrained.py             # the propositions, tested
```

Code: [`roc3/constrained.py`](../roc3/constrained.py). Background on the surface and VUS
themselves: [`TUTORIAL.md`](TUTORIAL.md).

Contents

0. [The sentence is ambiguous — and the two readings disagree](#0-the-sentence-is-ambiguous--and-the-two-readings-disagree)
1. [A world you can check by hand](#1-a-world-you-can-check-by-hand)
2. [Reading B: the constraint on the output is a *gauge*](#2-reading-b-the-constraint-on-the-output-is-a-gauge)
3. [Reading A: the constraint on the world cuts ROC space](#3-reading-a-the-constraint-on-the-world-cuts-roc-space)
4. [Computing the ceiling](#4-computing-the-ceiling)
5. [What to report](#5-what-to-report)
6. [Attainability — the open question](#6-attainability--the-open-question)
7. [A caveat this exposed: coarse models](#7-a-caveat-this-exposed-coarse-models)
8. [Summary](#8-summary)
9. [FAQ / objections](#9-faq--objections)

---

## 0. The sentence is ambiguous — and the two readings disagree

> *"the probabilities must satisfy `0 < p₁ < p₂ < p₃ < 1` and this is a logical
> constraint. Moreover, the model was trained in such a way that it obeys the
> constraint."*

"The probabilities" can mean two different things, and they lead to **opposite**
conclusions:

| | **Reading B — the model's output** | **Reading A — the world** |
|---|---|---|
| what is ordered | `p̂(x)`, the score vector the model emits | `P(Y=k \| x)`, the true posterior |
| enforced by | architecture (an ordered softmax, a projection layer, a Semantic Probabilistic Layer) | nothing — it is a fact about the problem |
| effect on the ROC surface | **none at all** | **cuts it with three planes** |
| effect on VUS | **none** — it is a change of coordinates | **caps it strictly below 1** |
| what breaks | `argmax`, and the readability of the weight simplex | accuracy as a metric, and the whole 0–1 scale |

The prompt's second sentence ("the model was trained to obey it") is Reading B. Its first
("the probabilities **must** satisfy … a **logical** constraint") is Reading A. Both are in
play, so this note does both — separately, because mixing them is the trap. Reading B is a
non-event dressed as a constraint; Reading A is a hard limit that is easy to miss.

### How to tell which one you are in

Reading A forces `π₁ < π₂ < π₃` on the *observed prevalences* (§3.3). So:

```
observed prevalences ordered?      -> Reading A is possible
observed prevalences NOT ordered?  -> Reading A is impossible; you are in Reading B
```

Verified on both cases:

```
balanced priors + ordered outputs   prevalences (0.333, 0.333, 0.333)  ordered? No  -> B
the world of §1                     prevalences (0.142, 0.271, 0.588)  ordered? Yes -> A possible
```

`roc3.constrained.constrained_report` runs this check and refuses to print Reading-A
ceilings when they do not apply.

---

## 1. A world you can check by hand

### 1.1 Notation

| symbol | meaning |
|---|---|
| `x` | one input case |
| `y` | its true class, in `{1, 2, 3}` |
| `p_k(x)` | the **posterior** `P(Y = k \| x)` — the probability the case really is class `k` |
| `π_k` | the **prevalence** of class `k`, i.e. `P(Y = k)` overall |
| `f_k(x)` | the **class-conditional density**: how class-`k` cases are distributed over inputs |
| `R_m` | the set of inputs a given classifier labels `m` — its **decision region** |
| `C[k, m]` | `P(x ∈ R_m \| Y = k)` — the row-stochastic **confusion matrix** |
| `S_k` | `C[k, k]`, the sensitivity / true class fraction for class `k` |
| `L` | a **likelihood ratio** `f_i(x)/f_j(x)` |

The three are tied together by Bayes: `p_k(x) = π_k f_k(x) / g(x)`, where `g` is the
overall density of inputs. So the ordering constraint on the posterior,
`p₁(x) < p₂(x) < p₃(x)`, is the same statement as

```
π₁ f₁(x)  <  π₂ f₂(x)  <  π₃ f₃(x)        for every x.          (†)
```

Both forms are used below; `(†)` is the one that does the work.

### 1.2 The worked example

The simplest world with an ordered posterior: three possible inputs (call them atoms
`A`, `B`, `C`) with given posteriors. A case is drawn by picking an atom, then drawing its
label from that atom's posterior.

| atom | `P(x = atom)` | `p₁` | `p₂` | `p₃` | ordered? |
|---|---|---|---|---|---|
| A | 0.50 | 0.20 | 0.30 | 0.50 | ✅ |
| B | 0.30 | 0.10 | 0.30 | 0.60 | ✅ |
| C | 0.20 | 0.05 | 0.15 | 0.80 | ✅ |

Prevalences: `π_k = Σ_m P(x = m)·p_k(m)`, giving

```
π = (0.14, 0.27, 0.59)          and indeed π₁ < π₂ < π₃
```

### 1.3 The joint table — the object that explains everything

Write `J[k, m] = P(Y = k AND x = atom m) = P(x = m) · p_k(m)`:

|  | x = A | x = B | x = C | row sum = `π_k` |
|---|---|---|---|---|
| **class 1** | 0.100 | 0.030 | 0.010 | 0.140 |
| **class 2** | 0.150 | 0.090 | 0.030 | 0.270 |
| **class 3** | 0.250 | 0.180 | 0.160 | 0.590 |

**Every column increases as you go down.** That is exactly the constraint `(†)`, written as
joint probabilities instead of densities. Column A says: among cases at atom A, class 3
mass (0.250) exceeds class 2 mass (0.150) exceeds class 1 mass (0.100).

Keep that table in mind. The entire Reading-A analysis is the observation that *this
property survives adding columns together* — and a decision region is nothing but a set of
columns added together.

The class-conditional distributions are the rows divided by their sums:

```
f₁ = (0.7143, 0.2143, 0.0714)
f₂ = (0.5556, 0.3333, 0.1111)
f₃ = (0.4237, 0.3051, 0.2712)
```

---

## 2. Reading B: the constraint on the output is a *gauge*

### 2.1 What "gauge" means here

A **gauge** is a change of representation that leaves every observable quantity unchanged
— like measuring temperature in Celsius or Fahrenheit, or shifting the zero of potential
energy. The numbers on the dial move; nothing physical does.

The claim of this section is that forcing a model's outputs to be ordered is exactly that:
a relabelling of the coordinates, not a restriction on what the model can do.

### 2.2 The invariance

Every rule on the ROC surface is `ŷ(x) = argmax_k w_k p_k(x)`. Multiply the model's
columns by a fixed vector `c = (c₁, c₂, c₃)`:

```
argmax_k  w_k · (c_k p_k(x))   =   argmax_k  (w_k c_k) · p_k(x)
```

**Rescaling the model is the same as rescaling the weights.** The rule you get from the
rescaled model at weights `w` is the rule you got from the original model at weights
`w·c`. Since the surface is defined by sweeping `w` over *everything*, the set of rules —
and therefore the set of operating points, and therefore the volume — is untouched.

The rank-based `VUS₃AFC` is invariant for the same reason: each of its five conditions
([`TUTORIAL.md`](TUTORIAL.md) §8) shifts by the same constant on both sides.

### 2.3 Theorem — every model has an ordered twin

> For any score matrix there is a single vector `c` such that `p_k → c_k p_k` satisfies
> `p₁(x) < p₂(x) < p₃(x)` for **every** `x`, with the ROC surface and both VUS estimators
> unchanged.

*Proof.* Write `A(x) = log p₁ − log p₂` and `C(x) = log p₂ − log p₃`. A class rescaling
shifts `A` by `log(c₁/c₂)` and `C` by `log(c₂/c₃)`, by the same amount for every `x`.
Choose those shifts to be `−(max_x A + m)` and `−(max_x C + m)` for any margin `m > 0`.
Then `A(x) ≤ −m < 0` and `C(x) ≤ −m < 0` for all `x`, which is precisely
`p₁ < p₂ < p₃`. ∎

Worked on a Gaussian model:

```
original model:  ordering holds on  17.3% of rows
apply p_k -> c_k p_k with log c = (−17.619, −8.833, 0)
regauged model:  ordering holds on 100.0% of rows

  geometric VUS   0.7719359630  ->  0.7719359630
  3AFC VUS        0.7758680000  ->  0.7758680000
  identical set of operating points: True
```

Not "approximately" — the same numbers to ten decimal places, and the same set of points.

**Consequence.** An output-ordering constraint carries *no information about ranking
quality*. A constrained model can be a perfect ranker, a chance ranker, or actively
misleading, exactly like an unconstrained one.

### 2.4 Objection: "but the predictions did change!"

They did. If a case was called class 2 before and class 3 after, that is real. The
resolution is that **`w` is a coordinate, not an observable**: the same *rule* now sits at
a different `w`.

```
original model, w = (1, 1, 1)                      -> Se = (0.763, 0.727, 0.750)
regauged model, w = (1, 1, 1)                      -> Se = (0.000, 0.000, 1.000)   different rule
regauged model, w = 1/c = (0.9998, 0.0002, 0.0000) -> Se = (0.763, 0.727, 0.750)   SAME rule
```

The middle line is the one that looks alarming, and it is why §2.6 matters. The bottom
line is the point: the rule did not disappear, it moved. Sweeping `w` recovers it.

### 2.5 Objection: "it always predicts class 3, so it must be useless"

This is the most natural objection and it is wrong, in an instructive way. Take a model
that is a *perfect* ranker and put it in the ordered gauge:

```
argmax prediction for every single sample: class 3
  accuracy of that rule       = 0.3333  = the prevalence of class 3
  per-class Se at w = (1,1,1) = (0, 0, 1)
  smallest p₁ in the output   = 1.7e-11   (still strictly ordered)

  ...and yet   VUS = 1.000000    VUS₃AFC = 1.000000
  best rule on the surface: w = (1, 0, 0)  ->  Se = (1, 1, 1)
```

`argmax` is **one point** on the surface. VUS is the whole surface. A model whose default
operating point is degenerate can still rank perfectly — and this one does, because the
gauge only moved where the good rules live, not whether they exist.

This is the same distinction as binary AUC versus accuracy at a badly chosen threshold,
pushed to its extreme.

### 2.6 What genuinely breaks, and the free fix

**(a) `argmax` collapses.** As above. Anything downstream that calls `.predict()` — a
`classification_report`, a monitoring dashboard, a fairness audit — is measuring a
constant function. This is a real operational hazard, and the *only* one.

**(b) The weight simplex becomes unreadable.** Useful rules need `w₁ ≫ w₂ ≫ w₃`, so the
informative region is crushed into one corner of the ternary panel. On the demo the
optimum sits at a weight ratio of **9.2 × 10⁸**.

**(c) The fix costs nothing.** `gauge_normalize` divides column `k` by a reference `r_k` —
the per-column geometric mean, or the class prior (which makes plain `argmax` the
*maximum-likelihood* rule `argmax_k p_k(x)/π_k`, the classical prior-shift correction).
Being a class rescaling, it is covered by §2.2, so **no number changes**:

```
weight ratio at the optimum   9.2e8  ->  14.5
VUS                           0.6918339375  ->  0.6918339375
selected operating point      (0.798, 0.445, 0.830)  ->  (0.798, 0.445, 0.830)
```

Same rule, same confusion matrix, readable coordinates.

### 2.7 There is a floating-point cost

The gauge shift equals the *spread* of the model's log-odds, so a confident enough model
needs `p₁` below what float64 can hold:

| Gaussian separation | smallest log p needed | representable as a float64 probability? |
|---|---|---|
| 1.2 | −27 | yes |
| 2.5 | −68 | yes |
| 6.0 | −256 | yes |
| 20.0 | −2064 | **no** |

`to_ordered_gauge` raises with that diagnosis and offers `return_log=True`, where every
identity still holds exactly. This also exposed a real bug in the main library: a
probability floor of `1e-12` in `prepare_scores` was silently flattening confident models,
and it broke the *provable* invariance of §2.2 numerically — VUS moved 0.9971 → 0.8637
under a transformation that cannot change it. Floor is now `1e-300`. See
[`IDEAS_LOG.md`](IDEAS_LOG.md) §G9.

### 2.8 The chance level does not move

An uninformative constrained model still scores `1/6` (measured 0.172 at n = 400/class).
If the scores are independent of the label then all six label assignments are equally
likely to win, by exchangeability, no matter which region of the simplex the scores
occupy.

---

## 3. Reading A: the constraint on the world cuts ROC space

Now assume `(†)` holds of the **true posterior**. This is a statement about the problem,
and it binds every classifier — constrained or not, calibrated or not, however trained.

### 3.1 The order chamber — one sentence explains everything

The ordered part of the probability simplex, `T = {p : p₁ ≤ p₂ ≤ p₃}`, is a triangle with
corners

```
v₀ = (1/3, 1/3, 1/3)      v₁ = (0, 1/2, 1/2)      v₂ = (0, 0, 1)
```

Every ordered probability vector is a blend of those three, so they are the most extreme —
most informative — posteriors the logic permits. Read them off:

> **You can never be more than 1/3 sure of class 1, or more than 1/2 sure of class 2.**

The best possible evidence for class 1 is the *uniform* posterior. The label noise on
classes 1 and 2 is irreducible; no data and no model capacity touches it. Everything below
is that sentence made quantitative.

### 3.2 The master lemma

Take **any** classifier at all — any partition of the input space into decision regions
`R₁, R₂, R₃`. Its confusion matrix is `C[k, m] = P(x ∈ R_m | Y = k)`.

Now note that `π_k · C[k, m]` is just the **joint probability** `P(Y = k AND x ∈ R_m)` —
in the table of §1.3, it is the sum of column entries over the atoms in `R_m`. Since every
column of that table increases down the class index, so does any sum of columns:

> **Lemma.**  `π_i · C[i, m]  ≤  π_j · C[j, m]`   for all `i < j` and all `m`.  (★)
>
> *Proof.* `π_i C[i,m] = ∫_{R_m} π_i f_i ≤ ∫_{R_m} π_j f_j = π_j C[j,m]`, using `(†)`
> pointwise. ∎

Nine linear inequalities (3 class pairs × 3 decision regions). **That is the whole of
Reading A**; everything that follows is a corollary.

Checked on three rules in the worked world:

| rule | `(S₁, S₂, S₃)` | accuracy | lemma holds? |
|---|---|---|---|
| A→1, B→2, C→3 | (0.7143, 0.3333, 0.2712) | 0.350 | ✅ tightest slack +0.0200 |
| always predict 3 | (0.0000, 0.0000, 1.0000) | **0.590** | ✅ tightest slack 0.0000 |
| A→1, B→1, C→3 | (0.9286, 0.0000, 0.2712) | 0.290 | ✅ tightest slack 0.0000 |

### 3.3 Corollary 1 — the prevalences are ordered

Sum (★) over `m`: `π_i · 1 ≤ π_j · 1`. This is what makes the §0 diagnostic work.

The degenerate case is instructive: if `π_i = π_j` then (★) forces `C[i,·] = C[j,·]`, so
classes `i` and `j` are **indistinguishable to every classifier** and the ceiling collapses
to chance. An ordered posterior with balanced priors is not a hard problem — it is an
impossible one.

### 3.4 Corollary 2 — the perfect corner is unreachable

`S₁ = S₂ = 1` requires `C[1,1] = 1` and `C[2,2] = 1`, hence `C[2,1] = 0`. Then (★) with
`i=1, j=2, m=1` says `π₁ · 1 ≤ π₂ · 0`, i.e. `0.14 ≤ 0`. False.

More usefully, relaxing `C[j,m] ≤ 1 − S_j` turns (★) into three **cutting planes** through
the ROC cube:

```
π₁S₁ + π₂S₂ ≤ π₂        π₂S₂ + π₃S₃ ≤ π₃        π₁S₁ + π₃S₃ ≤ π₃
```

With `π = (0.14, 0.27, 0.59)`, evaluated at the perfect corner `(1,1,1)`:

| plane | requires | at (1,1,1) | satisfied? |
|---|---|---|---|
| `0.14·S₁ + 0.27·S₂ ≤ 0.27` | | 0.41 ≤ 0.27 | ❌ |
| `0.27·S₂ + 0.59·S₃ ≤ 0.59` | | 0.86 ≤ 0.59 | ❌ |
| `0.14·S₁ + 0.59·S₃ ≤ 0.59` | | 0.73 ≤ 0.59 | ❌ |

### 3.5 Corollary 3 — accuracy is a dead metric

Apply (★) with `i = k`, `j = 3`, `m = k`, for `k < 3`:

```
Σ_k π_k C[k,k]  ≤  Σ_k π₃ C[3,k]  =  π₃ · 1  =  π₃
```

So **maximum achievable accuracy is exactly `π₃`, and the constant rule "always predict 3"
attains it.** No classifier of any kind can beat guessing the majority class. Equivalently,
the Bayes error is `1 − E[max_k p_k] = 1 − E[p₃] = 1 − π₃`.

In the worked world: `π₃ = 0.59`, and brute force over all `3³ = 27` deterministic rules
gives a best accuracy of exactly **0.5900**, attained by the constant rule.

If you are in Reading A and someone reports accuracy, they are reporting the prevalence.

### 3.6 Corollary 4 — every pairwise AUC is capped

**Step 1: the constraint bounds a likelihood ratio.** Divide `(†)` through:

```
π_i f_i(x) ≤ π_j f_j(x)   ⟺   L(x) := f_i(x)/f_j(x)  ≤  R := π_j / π_i
```

**Step 2: a bounded likelihood ratio caps the AUC.** For the pair `(i, j)`, the optimal
score *is* `L`, and `AUC = P(L(X_i) > L(X_j))`. Let `F_j` be the CDF of `L` under class
`j`. Then:

* `AUC = E_i[F_j(L)]` — by definition.
* `= E_j[L · F_j(L)]` — change of measure, since `dP_i/dP_j = L`.
* Substitute `V = F_j(L)`, which is `Uniform(0,1)` under class `j`, and write `L = q(V)`
  with `q` non-decreasing. Then `AUC = ∫₀¹ v·q(v) dv`.
* Two constraints on `q`: `∫₀¹ q(v) dv = E_j[L] = ∫ (f_i/f_j) f_j = 1`, and `q ≤ R`.
* Maximise `∫ v·q(v) dv` subject to those. Since the weight `v` increases, push all of
  `q`'s mass to the top: `q = R` on the top `1/R` of the range, `0` below.
* `AUC ≤ ∫_{1−1/R}^{1} R·v dv = (R/2)[1 − (1−1/R)²] = **1 − 1/(2R)**`.

Substituting `R = π_j/π_i`:

```
AUC(i, j)  ≤  1 − π_i / (2 π_j)
```

Sanity: `R = 1` (equal priors) gives `AUC ≤ 1/2`, matching §3.3. `R → ∞` gives `AUC → 1`.

For the worked world:

| pair | `R = π_j/π_i` | cap `1 − 1/(2R)` |
|---|---|---|
| 1 vs 2 | 0.27/0.14 = 1.929 | **0.7407** |
| 2 vs 3 | 0.59/0.27 = 2.185 | **0.7712** |
| 1 vs 3 | 0.59/0.14 = 4.214 | **0.8814** |

**Step 3: the bound is tight.** The optimising `q` above is a two-point distribution, and
it can be built explicitly. Take `R = 4`, two inputs `u` and `v`:

* class `j` sits at `u` with probability `1 − 1/R = 0.75` and at `v` with probability
  `1/R = 0.25`;
* class `i` sits entirely at `v`.

Then `L(u) = 0` and `L(v) = R` exactly, and the constraint `π_i f_i ≤ π_j f_j` holds with
equality at `v`. Scoring by `L`:

```
class j scores: 300 cases at 0, 100 cases at 4
class i scores: 100 cases at 4
AUC (ties get 1/2) = 0.750 + 0.125 = 0.875000
bound 1 − 1/(2·4)                  = 0.875000        attained exactly
```

**Step 4: this also caps the VUS**, because correctly sorting a trio requires each pairwise
comparison to come out right (conditions 1–3 of [`TUTORIAL.md`](TUTORIAL.md) §8):

```
VUS ≤ min over pairs of AUC(i,j)  ≤  min over pairs of (1 − π_i/(2π_j))
```

§3.7 gives a better bound, but this one is closed-form.

### 3.7 The VUS ceiling, exactly

**What is being computed.** A confusion matrix is 9 numbers, but each row is a probability
vector, so **6 numbers determine it**. Take

```
x = (a₀, b₀, a₁, b₁, a₂, b₂)      with   C[k,1] = a_k ,  C[k,2] = b_k ,  C[k,3] = 1 − a_k − b_k
```

Every constraint is *linear* in `x`:

* 9 from "each row is a probability vector": `a_k ≥ 0`, `b_k ≥ 0`, `a_k + b_k ≤ 1`;
* 9 from the master lemma (★).

18 linear inequalities in 6 dimensions. A region carved out by flat walls like that is a
**polytope** — a higher-dimensional polyhedron. Its **vertices** are the points where 6
walls meet, and every feasible confusion matrix is a blend of them.

**The projection.** We only care about the diagonal `(S₁, S₂, S₃) = (C[1,1], C[2,2],
C[3,3])`, so map each vertex to its diagonal. The achievable set of operating points is
the convex hull of those images. For the worked world there are **18** distinct corner
operating points; the ones on the Pareto frontier are

```
(1.0000, 0.4815, 0.5424)     (0.0000, 1.0000, 0.5424)     (0.0000, 0.4815, 0.7797)
(1.0000, 0.0000, 0.7627)     (0.0000, 0.5185, 0.7627)     (0.0000, 0.0000, 1.0000)
```

Notice the first: `S₁ = 1` *is* reachable — but only by giving up most of `S₂` and `S₃`.
Nothing here comes close to `(1,1,1)`.

**The ceiling** is the volume of the **down-set** of that region — the set of requirement
triples `(u, v, z)` that some classifier could meet (same construction as
[`TUTORIAL.md`](TUTORIAL.md) §6, now applied to all classifiers at once rather than to one
model's weight sweep):

```
VUS_max(π) = 0.53243        for π = (0.14, 0.27, 0.59)
```

Against chance `0.1667` and an unconstrained maximum of `1.0000`.

The ceiling as a surface, `Zmax(u,v) = max{ S₃ : S₁ ≥ u, S₂ ≥ v }` — the roof over every
possible classifier:

```
           v=0.01   v=0.26   v=0.50   v=0.74   v=0.99
  u=0.01    0.992    0.880    0.768    0.657    0.545
  u=0.26    0.934    0.822    0.710    0.599    0.000
  u=0.50    0.876    0.764    0.653    0.000    0.000
  u=0.74    0.818    0.706    0.595    0.000    0.000
  u=0.99    0.760    0.648    0.000    0.000    0.000
```

The zeros are requirements that are simply infeasible. Mean height 0.533 ≈ the ceiling.

And the scalar ceilings for this world:

```
accuracy            0.5900     (= π₃; the LP agrees to 6 decimals)
balanced accuracy   0.6746
worst-class Se      0.5900
VUS                 0.5324
```

![ceiling](../figures/08_constrained_ceiling.png)

---

## 4. Computing the ceiling

Two independent implementations, which is how the result is trusted:

```python
from roc3.constrained import vus_ceiling, ceiling_surface, scalar_ceilings

vus_ceiling(priors, method="exact")   # vertex enumeration + hull volume
vus_ceiling(priors, method="lp")      # a grid of small linear programs, integrated
```

* **`exact`** enumerates every 6-subset of the 18 facets (`C(18,6) = 18 564`), solves the
  6×6 system, keeps the feasible solutions, projects onto the diagonal, and takes the
  down-set volume of the convex hull. No resolution parameter.
* **`lp`** solves, for each cell of a `(u,v)` grid, `max S₃ subject to S₁ ≥ u, S₂ ≥ v` and
  the 18 constraints, then averages. Slower, but it produces the ceiling as a *surface*,
  which is what the figure needs.

They agree to `1e-5` (0.53243 vs 0.53250 above). As a third check, specialising the whole
construction to two classes reproduces the closed form `1 − π₁/(2π₂)` from §3.6 to 9
decimal places, for every prior pair tested — two derivations that share no code arriving
at the same number.

---

## 5. What to report

**Renormalise.** The attainable range is `[1/6, VUS_max(π)]`, not `[1/6, 1]`:

```python
from roc3.constrained import normalized_vus
normalized_vus(vus, priors)        # (VUS − 1/6) / (VUS_max(π) − 1/6)
```

0 at chance, 1 at the best any classifier could do given the logic.

On the worked example in `experiments/04_constrained.py`, a model scores `VUS = 0.550`.
Against the usual scale that reads as mediocre. Against its ceiling of `0.610` it is
capturing **86.6%** of everything attainable, and the honest number is
`VUS_normalized = 0.866`. Reporting 0.550 out of 1.000 would understate the model by a
factor that has nothing to do with the model.

The ceiling is very sensitive to the priors, so report it alongside:

| priors `π` | `VUS_max` | max balanced acc. | max accuracy |
|---|---|---|---|
| (0.320, 0.330, 0.350) | **0.191** | 0.363 | 0.350 |
| (0.250, 0.330, 0.420) | **0.304** | 0.486 | 0.420 |
| (0.167, 0.333, 0.500) | **0.444** | 0.611 | 0.500 |
| (0.100, 0.250, 0.650) | **0.618** | 0.739 | 0.650 |
| (0.050, 0.200, 0.750) | **0.744** | 0.828 | 0.750 |
| (0.020, 0.080, 0.900) | **0.832** | 0.887 | 0.900 |
| (0.005, 0.045, 0.950) | **0.921** | 0.947 | 0.950 |

With near-equal priors the ceiling is **0.191** against a chance level of 0.167 — the
constraint leaves almost no room at all. It is survivable only when the priors are
strongly ordered, which makes sense: `π_j/π_i` is precisely the likelihood-ratio budget.

`constrained_report` / `format_constrained_report` print all of it: constraint compliance,
the Reading-A/B diagnostic, the argmax-degeneracy warning, raw and normalised VUS, the
ceiling, and the scalar ceilings.

---

## 6. Attainability — the open question

`VUS_max(π)` is the down-set volume of the polytope of **all** feasible confusion matrices.
A single fixed world exposes only a 2-parameter family of Bayes rules (the weighted-argmax
sweep), which traces the Pareto frontier of *that world's* achievable set — a subset. So

```
sup over worlds of VUS   ≤   VUS_max(π)
```

possibly strictly. Measured, by building worlds that place mass in the order chamber and
sample `y ~ Cat(p(x))` so the true posterior is ordered by construction:

| atoms in the world | fraction of the attainable volume captured |
|---|---|
| 3 (the chamber vertices) | 37.3% |
| 40 | 65.6% |
| 120 | 85.6% |
| 400 | 87.7% |
| 1200 | **89.3%** |
| uniform lattice over the chamber | 53.5% |

Richness helps, and mass concentrated near the chamber's extreme points helps — Dirichlet
concentration 0.12 beats a uniform lattice by 36 points, which is what the LP predicts,
since the optimum sits at vertices. **Whether the supremum equals `VUS_max(π)` is
unresolved.** Every one of 80 randomly generated constrained worlds respected the ceiling,
with a tightest observed slack of `+0.118`.

Also open:

1. **A direct VUS ceiling** not routed through the polytope — a 3-class analogue of the
   `1 − 1/(2R)` change-of-measure argument in §3.6.
2. **A world-aware ceiling.** Given an estimate of the posterior *distribution* rather than
   just the priors, the achievable set shrinks. That would turn `VUS_normalized` from
   "fraction of what is logically possible" into "fraction of what is possible on *this*
   problem".
3. **Partial orders.** If the logic only says `p₁ < p₃`, the polytope has fewer facets and
   the ceiling rises. The machinery handles any set of constraints of the form (★); only
   the list changes.
4. **Uncertainty in `π`.** `VUS_max` is sensitive to the priors, so a confidence interval
   on the ceiling belongs next to the point estimate.

---

## 7. A caveat this exposed: coarse models

Running the worked world end to end produces something that looks alarming:

```
3-atom world, ideal observer (the true posterior):
  geometric VUS (staircase)   0.0642     <- BELOW the chance level of 0.1667
  convex-hull VUS (randomised) 0.2652
  3AFC VUS (rank-based)        0.2751
  ceiling                      0.5293
```

The model is not worse than chance. The **staircase estimator is starved**. With only three
distinct score vectors, the weight sweep can reach just 8 distinct operating points, and
**seven of them have a zero coordinate**, so their boxes are flat and contribute no volume:

```
(0.000, 0.000, 1.000)   box 0.0000        (0.716, 0.000, 0.579)   box 0.0000
(0.000, 0.557, 0.579)   box 0.0000        (0.716, 0.443, 0.000)   box 0.0000
(0.000, 1.000, 0.000)   box 0.0000        (0.930, 0.000, 0.271)   box 0.0000
(1.000, 0.000, 0.000)   box 0.0000        (0.716, 0.331, 0.271)   box 0.0642  <- the only one
```

The whole VUS is that single box. The chance level of 1/6 assumes a *continuum* of
operating points filling the chance plane; a three-atom model cannot get near it.

This is the coarseness caveat from [`IDEAS_LOG.md`](IDEAS_LOG.md) §G4 in its sharpest form.
It converges away as the score distribution gets richer — same chamber, more atoms:

| atoms | operating points | geometric VUS | hull VUS | 3AFC VUS |
|---|---|---|---|---|
| 3 | 9 | 0.323 | 0.582 | 0.581 |
| 12 | 54 | 0.446 | 0.566 | 0.574 |
| 60 | 1 073 | 0.478 | 0.503 | 0.503 |
| 300 | 14 167 | 0.475 | 0.484 | 0.484 |

**Practical rule.** If your model emits few distinct score vectors — a shallow tree, a
quantised or rule-based scorer, a small ensemble vote — read `VUS₃AFC` or
`vus_convex_hull`, not the staircase. When they disagree by a lot, that gap is telling you
the model is coarse, which is itself worth knowing.

---

## 8. Summary

* An ordering constraint on the **model's output** is a **gauge choice**. It changes no ROC
  quantity whatsoever — same surface, same VUS, same operating points, to the last decimal.
  It breaks `argmax` and makes the weight simplex unreadable; both are fixed for free by
  `gauge_normalize`, which also changes nothing. A constrained model can still be a perfect
  ranker, and §2.5 exhibits one.
* An ordering constraint on the **world** is a hard limit on every classifier. It caps each
  pairwise AUC at `1 − π_i/(2π_j)`, caps accuracy at `π₃` (so the constant rule is
  Bayes-optimal and unbeatable), makes the perfect corner unreachable, and caps the VUS at
  a computable `VUS_max(π) < 1`.
* Therefore: **stop normalising by 1.** Normalise by the ceiling.
* The mechanism, in one line: *the constraint says you can never be more than 1/3 sure of
  class 1, or more than 1/2 sure of class 2.*

---

## 9. FAQ / objections

**"You said the constraint changes nothing, then you said it caps everything. Which is
it?"** Both, for different constraints that the same English sentence describes. Constrain
the *model's output* → nothing changes (§2). Constrain the *truth* → everything is capped
(§3). §0 has the test for telling them apart.

**"The model always predicts class 3. Surely VUS should be low?"** No — §2.5 builds one
with `VUS = 1.0`. `argmax` is one point on the surface; VUS is the whole surface.

**"Rescaling the columns changed my predictions, so how can nothing have changed?"**
The *rule* did not change, its *coordinate* did. §2.4 recovers the identical operating
point at `w = 1/c`.

**"Why can't a really good model beat the ceiling?"** Because the master lemma (★) is a
statement about decision *regions*, not about models. Any classifier partitions the input
space somehow, and the inequality follows from the posterior ordering alone. There is no
model that escapes it.

**"Why is the ceiling so low — 0.44 with `π = (1/6, 1/3, 1/2)`?"** Because
`π_j/π_i` is the entire likelihood-ratio budget. With `π₁/π₂ = 1/2`, class 1 can never be
more than twice as likely as class 2 at any input, so the two are barely separable. §3.3
shows the limit: equal priors make them *inseparable*.

**"My VUS came out below 1/6."** Probably §7 — a coarse model starving the staircase
estimator. Check `vus_convex_hull` and `VUS₃AFC`.

**"Can I use this if only some of the ordering is known?"** Yes. The machinery takes any
set of constraints of the form `π_i C[i,m] ≤ π_j C[j,m]`; a partial order just means fewer
of them and a higher ceiling. See §6, item 3.

---

## References for this note

* Boyd, K., Davis, J., Page, D. (2012). *Unachievable region in precision-recall space and
  its effect on empirical evaluation.* ICML. <https://arxiv.org/pdf/1206.4667> — the
  precedent: a region of evaluation space the problem's structure, not the model, puts out
  of reach.
* Ahmed, K., Teso, S., Chang, K.-W., Van den Broeck, G., Vergari, A. (2022). *Semantic
  Probabilistic Layers for Neuro-Symbolic Learning.* NeurIPS.
  <https://arxiv.org/abs/2206.00426> — how a model is "trained to obey" a hard logical
  constraint with guaranteed test-time satisfaction. The Reading-B mechanism.
* Standard ROC/VUS background: [`REFERENCES.md`](REFERENCES.md), and
  [`TUTORIAL.md`](TUTORIAL.md) for the surface and VUS themselves.
