# `roc3` tutorial — the 3-class ROC surface and VUS, from scratch

This is the long-form explanation. Nothing is assumed except that you have seen an ROC
curve once. Every symbol is defined where it first appears, and **every number below is
real** — regenerate all of them with

```bash
.venv/bin/python experiments/05_tutorial_numbers.py
```

Contents

1. [The setup, and all the notation](#1-the-setup-and-all-the-notation)
2. [Warm-up: what the binary ROC curve *is*](#2-warm-up-what-the-binary-roc-curve-is)
3. [The 3-class decision rule](#3-the-3-class-decision-rule-weighted-argmax)
4. [The operating point](#4-the-operating-point)
5. [The surface](#5-the-surface)
6. [VUS, definition 1: a volume](#6-vus-definition-1-a-volume)
7. [VUS, definition 2: sorting a trio](#7-vus-definition-2-sorting-a-trio)
8. [The algebraic shortcut](#8-the-algebraic-shortcut)
9. [Why the two definitions agree](#9-why-the-two-definitions-agree)
10. [Reading the dashboard](#10-reading-the-dashboard)
11. [Choosing an operating point](#11-choosing-an-operating-point)
12. [Uncertainty](#12-uncertainty)
13. [Ordered classes](#13-ordered-classes-the-ordinal-mode)
14. [Glossary of every symbol](#14-glossary-of-every-symbol)
15. [Traps and FAQ](#15-traps-and-faq)

---

## 1. The setup, and all the notation

### 1.1 What you have

A **classifier** already trained, and a **held-out sample** to evaluate it on. Concretely:

| symbol | meaning |
|---|---|
| `n` | number of evaluation samples |
| `x` | one sample (an input — an image, a row of features, whatever) |
| `y` | the **true class** of a sample, one of `{1, 2, 3}` |
| `n₁, n₂, n₃` | how many samples belong to each class; `n₁+n₂+n₃ = n` |
| `p_k(x)` | the **score the model gives to class `k`** for input `x`. Usually a probability, so `p₁(x)+p₂(x)+p₃(x) = 1`. |
| `P` | the whole `n × 3` matrix of scores. Row `i` is `(p₁(xᵢ), p₂(xᵢ), p₃(xᵢ))`. |
| `π_k` | the **prevalence** of class `k`, i.e. `n_k / n` — how common that class is |

A note on indices: the tutorial numbers classes **1, 2, 3**; NumPy numbers the columns of
`P` **0, 1, 2**. Column `0` of `P` is class 1. That is the only off-by-one in the whole
library and it is worth keeping in mind while reading code.

### 1.2 The toy dataset used throughout

Nine samples, three per class. Small enough that everything can be checked by hand.

| sample | true class `y` | `p₁` | `p₂` | `p₃` | `argmax` |
|---|---|---|---|---|---|
| x1 | 1 | 0.70 | 0.20 | 0.10 | 1 |
| x2 | 1 | 0.55 | 0.30 | 0.15 | 1 |
| x3 | 1 | 0.30 | 0.45 | 0.25 | 2 ← wrong |
| x4 | 2 | 0.20 | 0.60 | 0.20 | 2 |
| x5 | 2 | 0.35 | 0.40 | 0.25 | 2 |
| x6 | 2 | 0.15 | 0.35 | 0.50 | 3 ← wrong |
| x7 | 3 | 0.10 | 0.20 | 0.70 | 3 |
| x8 | 3 | 0.25 | 0.25 | 0.50 | 3 |
| x9 | 3 | 0.30 | 0.40 | 0.30 | 2 ← wrong |

Every row sums to 1. `n₁ = n₂ = n₃ = 3`. The plain-`argmax` classifier gets **6 of 9**
right, so accuracy is 0.667.

```python
import numpy as np
from roc3 import roc_surface

y = np.array([1,1,1, 2,2,2, 3,3,3])
P = np.array([[0.70,0.20,0.10],[0.55,0.30,0.15],[0.30,0.45,0.25],
              [0.20,0.60,0.20],[0.35,0.40,0.25],[0.15,0.35,0.50],
              [0.10,0.20,0.70],[0.25,0.25,0.50],[0.30,0.40,0.30]])
surf = roc_surface(y, P, classes=[1,2,3])
```

### 1.3 Log scores and log-ratios

Almost everything is easier in logs, because the rules only care about *ratios* of scores.
Define, for each sample:

```
D₁₂(x) = log p₁(x) − log p₂(x)     "how much more class-1-ish than class-2-ish"
D₁₃(x) = log p₁(x) − log p₃(x)
D₂₃(x) = log p₂(x) − log p₃(x)
```

These satisfy `D₁₃ = D₁₂ + D₂₃` identically — only two of the three are independent. For
the toy data:

| sample | `log p₁` | `log p₂` | `log p₃` | `D₁₂` | `D₁₃` | `D₂₃` |
|---|---|---|---|---|---|---|
| x1 | −0.3567 | −1.6094 | −2.3026 | **1.2528** | 1.9459 | 0.6931 |
| x2 | −0.5978 | −1.2040 | −1.8971 | 0.6061 | 1.2993 | 0.6931 |
| x3 | −1.2040 | −0.7985 | −1.3863 | −0.4055 | 0.1823 | 0.5878 |
| x4 | −1.6094 | −0.5108 | −1.6094 | −1.0986 | 0.0000 | 1.0986 |
| x5 | −1.0498 | −0.9163 | −1.3863 | −0.1335 | 0.3365 | 0.4700 |
| x6 | −1.8971 | −1.0498 | −0.6931 | −0.8473 | −1.2040 | −0.3567 |
| x7 | −2.3026 | −1.6094 | −0.3567 | −0.6931 | −1.9459 | −1.2528 |
| x8 | −1.3863 | −1.3863 | −0.6931 | 0.0000 | −0.6931 | −0.6931 |
| x9 | −1.2040 | −0.9163 | −1.2040 | −0.2877 | 0.0000 | 0.2877 |

Read `D₁₂(x1) = 1.2528` as: *for sample x1 the model thinks class 1 is `e^1.2528 ≈ 3.5`
times as likely as class 2* — which matches `0.70 / 0.20 = 3.5`.

---

## 2. Warm-up: what the binary ROC curve *is*

Skip if this is second nature — but the 3-class construction is built by lifting each of
these pieces one dimension, so it pays to have them named.

Two classes, `0` (negative) and `1` (positive). The model emits a single score `s(x)`
(higher = more positive-looking).

### 2.1 The rule is a threshold sweep

Pick a cut-off `t`. Predict positive iff `s(x) > t`. **Every choice of `t` is a different
classifier.** That one-parameter family is what the ROC curve draws.

### 2.2 The two coordinates

For a given `t`:

* **Sensitivity** `Se₁ = P(predict 1 | truly 1)` — of the true positives, what fraction do
  we catch? Also called the true positive rate, TPR.
* **Specificity** `Se₀ = P(predict 0 | truly 0)` — of the true negatives, what fraction do
  we correctly leave alone? Equal to `1 − FPR`.

The textbook ROC plots `TPR` against `FPR`. `roc3` uses `(Se₀, Se₁)` instead — the same
information, but now **both axes mean "fraction correct for this class"**, which is the
form that generalises. Perfect is the corner `(1, 1)`; a coin flip lies on the line
`Se₀ + Se₁ = 1`.

### 2.3 AUC, two ways, on a four-sample example

Labels `[0, 0, 1, 1]`, scores `[0.10, 0.60, 0.40, 0.80]`. So the negatives score
`{0.10, 0.60}` and the positives score `{0.40, 0.80}`.

**Way one — the pair-ranking definition.** `AUC = P(s(X⁺) > s(X⁻))`. Read that as:
*draw one random positive sample `X⁺` and one random negative sample `X⁻`; what is the
chance the model scores the positive higher?* With 2 × 2 = 4 pairs:

| positive | negative | positive scores higher? |
|---|---|---|
| 0.40 | 0.10 | yes |
| 0.40 | 0.60 | **no** |
| 0.80 | 0.10 | yes |
| 0.80 | 0.60 | yes |

3 of 4 → **AUC = 0.75**.

**Way two — the area.** Sweep `t` through every value and collect the operating points:

| `t` | `Se₀` | `Se₁` |
|---|---|---|
| −∞ | 0.00 | 1.00 |
| 0.10 | 0.50 | 1.00 |
| 0.40 | 0.50 | 0.50 |
| 0.60 | 1.00 | 0.50 |
| 0.80 | 1.00 | 0.00 |

The **region under the curve** is everything you can *guarantee*: the set of pairs
`(u, v)` such that some threshold achieves `Se₀ ≥ u` **and** `Se₁ ≥ v`. Its area is
**0.75** — the same number.

Both readings generalise to three classes, and they become §6 and §7 respectively.

---

## 3. The 3-class decision rule: weighted argmax

### 3.1 Why plain `argmax` is not enough

`argmax_k p_k(x)` gives you exactly **one** classifier. An ROC analysis needs a *family*.
For two classes the family came from moving `t`. What is the analogue?

### 3.2 The rule

Introduce a **weight vector** `w = (w₁, w₂, w₃)` with `w_k > 0`, and define

```
ŷ(x; w) = argmax_k   w_k · p_k(x)
```

In words: **multiply each class's score by that class's weight, then take the biggest.**
`w_k` is "how much I care about catching class `k`". Raising `w₂` makes the classifier
readier to say "class 2".

Only the *ratios* of the `w_k` matter (scaling all three by 5 changes nothing), so the
family has **2 free parameters** — one more than binary's single `t`, which is exactly
right.

### 3.3 It really is the threshold sweep, one dimension up

With two classes, `w₁p₁ > w₀p₀` ⟺ `p₁ > w₀/(w₀+w₁)` (using `p₀ = 1−p₁`). Setting
`t = w₀/(w₀+w₁)` recovers the ordinary threshold exactly. So this is not a new idea, it is
*the* idea, written in a form that survives a third class.

### 3.4 Worked: three different weights on the toy data

**`w = (1, 1, 1)` — plain argmax.**

| sample | `w₁p₁` | `w₂p₂` | `w₃p₃` | predicted | true |
|---|---|---|---|---|---|
| x1 | **0.700** | 0.200 | 0.100 | 1 | 1 |
| x2 | **0.550** | 0.300 | 0.150 | 1 | 1 |
| x3 | 0.300 | **0.450** | 0.250 | 2 | 1 |
| x4 | 0.200 | **0.600** | 0.200 | 2 | 2 |
| x5 | 0.350 | **0.400** | 0.250 | 2 | 2 |
| x6 | 0.150 | 0.350 | **0.500** | 3 | 2 |
| x7 | 0.100 | 0.200 | **0.700** | 3 | 3 |
| x8 | 0.250 | 0.250 | **0.500** | 3 | 3 |
| x9 | 0.300 | **0.400** | 0.300 | 2 | 3 |

**`w = (2, 1, 1)` — double the weight on class 1.** Now x3 flips to class 1 (correct), but
x5, x8, x9 also flip to class 1 (two of them wrongly):

| sample | `w₁p₁` | `w₂p₂` | `w₃p₃` | predicted | true |
|---|---|---|---|---|---|
| x3 | **0.600** | 0.450 | 0.250 | 1 | 1 |
| x5 | **0.700** | 0.400 | 0.250 | 1 | 2 |
| x8 | **0.500** | 0.250 | 0.500 | 1 | 3 |
| x9 | **0.600** | 0.400 | 0.300 | 1 | 3 |

**`w = (5, 1, 0.5)` — an extreme.** Class 1 wins for *every* sample. That is one of the
three degenerate corners of the family ("always predict 1"); they are always reachable and
they always sit on the chance plane.

> **A note on ties.** With two-decimal toy data it is easy to land exactly on a decision
> boundary: at `w = (1, 1, 3)`, sample x4 scores `w₂p₂ = 1×0.60` and `w₃p₃ = 3×0.20`,
> which are *equal*. The library breaks such ties the way `np.argmax` does — the lowest
> class index wins — and the rule grid deliberately places its cut points strictly
> *between* distinct data values so the sweep never sits on one. Ties have probability
> zero for genuinely continuous scores; they matter only for hand-picked round numbers
> like these. (`3 × 0.20` is also not exactly `0.60` in binary floating point, which is a
> good reason not to build an example on a tie.)

### 3.5 Log-ratio coordinates

Taking logs, `ŷ = argmax_k (log p_k(x) + log w_k)`. Only differences of `log w` matter, so
the whole 2-parameter family is indexed by

```
a = log(w₂/w₁)        b = log(w₃/w₁)
```

`a = b = 0` is plain argmax. Positive `a` favours class 2 over class 1. The library
stores these on the surface object as `surf.A` and `surf.B`; they are what the grid
actually sweeps.

---

## 4. The operating point

### 4.1 The confusion matrix

Fix a `w`. Apply the rule to all `n` samples and count:

```
C[k, m] = number of samples whose TRUE class is k and whose PREDICTED class is m
```

Rows are truth, columns are prediction. For `w = (1,1,1)` on the toy data:

```
              →1   →2   →3
  true 1   [   2    1    0 ]
  true 2   [   0    2    1 ]
  true 3   [   0    1    2 ]
```

Row 1 reads: of the three true class-1 samples, 2 were called class 1 and 1 was called
class 2.

### 4.2 The three sensitivities

```
S_k  =  C[k, k] / n_k  =  P(predict k | truly k)
```

This is the **fraction of class `k` that the classifier gets right** — the direct
generalisation of sensitivity and specificity. Also called the *true class fraction* in
the medical literature. For `w = (1,1,1)`:

```
S₁ = 2/3 = 0.667      S₂ = 2/3 = 0.667      S₃ = 2/3 = 0.667
```

**The triple `(S₁, S₂, S₃)` is one point in the unit cube `[0,1]³`.** That point is the
*operating point* of this classifier. Three of them, from §3.4:

| `w` | `(S₁, S₂, S₃)` | `S₁+S₂+S₃` |
|---|---|---|
| (1, 1, 1) | (0.667, 0.667, 0.667) | 2.000 |
| (2, 1, 1) | (1.000, 0.333, 0.333) | 1.667 |
| (1, 1, 2) | (0.667, 0.333, 1.000) | 2.000 |
| (5, 1, 0.5) | (1.000, 0.000, 0.000) | 1.000 |

```python
surf.confusion_at([1, 1, 1])      # the 3x3 matrix
surf.sensitivities_at([1, 1, 1])  # the triple
```

### 4.3 Two landmarks in the cube

* **The perfect corner `(1, 1, 1)`** — every class fully caught.
* **The chance plane `S₁ + S₂ + S₃ = 1`.** Any rule that ignores `x` — "always say 2",
  "flip a weighted coin" — assigns class `k` with some fixed probability `q_k`, so
  `S_k = q_k` and `Σ S_k = Σ q_k = 1`. Landing on this plane means the classifier is
  contributing nothing. Notice the last row of the table above: `w = (5, 1, 0.5)`
  degenerated into "always 1" and duly sums to 1.

---

## 5. The surface

Now sweep `w` over all its possibilities and collect every operating point. Because `w`
has 2 free parameters, the points form a **2-dimensional surface inside the 3-dimensional
cube**. That surface is the 3-class ROC curve.

With only 9 samples the surface is a finite set of points — which makes it printable. The
library's grid contains 90 weight vectors, producing **29 distinct operating points**:

```
    S1     S2     S3     sum    example w (normalised)
 1.000  0.667  0.333    2.000   (0.599, 0.277, 0.124)
 1.000  0.333  1.000    2.333   (0.404, 0.153, 0.443)
 1.000  0.000  0.000    1.000   (1.000, 0.000, 0.000)   <- "always 1"
 0.667  1.000  0.667    2.333   (0.327, 0.442, 0.231)
 0.667  0.667  1.000    2.333   (0.309, 0.289, 0.401)
 0.667  0.667  0.667    2.000   (0.378, 0.354, 0.268)   <- plain argmax
 0.000  1.000  0.000    1.000   (0.000, 1.000, 0.000)   <- "always 2"
 0.000  0.000  1.000    1.000   (0.000, 0.000, 1.000)   <- "always 3"
 ...  (29 in total)
```

```python
surf.S            # (n_rules, 3) the operating points
surf.W            # (n_rules, 3) the weight vector behind each one
surf.CM           # (n_rules, 3, 3) the full confusion matrix behind each one
np.unique(surf.S, axis=0)
```

Two sanity checks visible in that table: the three "always predict k" corners are present
and all sum to 1 (on the chance plane, as they must); and no point reaches `(1,1,1)`,
because this model is not perfect.

With a realistic sample size the points become dense and the picture is a genuine surface:

![surface inflating with signal](../figures/06_signal_sweep.png)

---

## 6. VUS, definition 1: a volume

### 6.1 What region are we measuring?

Same idea as §2.3 "way two". Define the **achievable region**

```
R = { (u, v, z) ∈ [0,1]³  :  some weight w achieves S₁ ≥ u AND S₂ ≥ v AND S₃ ≥ z }
```

In words: *the set of sensitivity requirements the model can actually meet.* If you demand
"catch 80% of class 1, 60% of class 2 and 90% of class 3", that demand is the point
`(0.8, 0.6, 0.9)`; it is inside `R` exactly when some `w` delivers it.

`R` is a **down-set**: if a requirement is met, every weaker requirement is met too. So
`R` is the union of boxes `[0,S₁]×[0,S₂]×[0,S₃]`, one box per operating point.

> **VUS = volume of `R`.**

### 6.2 Doing it by hand on the toy data

Most of the 29 points are *dominated* — some other point is at least as good on all three
axes, so its box is already inside another box. Only 4 points survive (the **Pareto
front**):

| point | box volume `S₁·S₂·S₃` |
|---|---|
| A = (1, 2/3, 1/3) | 2/9 = 0.2222 |
| B = (1, 1/3, 1) | 1/3 = 0.3333 |
| C = (2/3, 1, 2/3) | 4/9 = 0.4444 |
| D = (2/3, 2/3, 1) | 4/9 = 0.4444 |

Adding those gives 1.4444, which is wrong — the boxes overlap. Inclusion–exclusion, using
`vol(box(p) ∩ box(q)) = min(p₁,q₁)·min(p₂,q₂)·min(p₃,q₃)`:

```
singles   A+B+C+D                              = 6/27 + 9/27 + 12/27 + 12/27 = 39/27
pairs     AB 3/27, AC 4/27, AD 4/27,
          BC 4/27, BD 6/27, CD 8/27            =                              29/27
triples   ABC 2/27, ABD 2/27, ACD 4/27,
          BCD 4/27                             =                              12/27
quadruple ABCD                                 =                               2/27

volume = 39/27 − 29/27 + 12/27 − 2/27 = 20/27 = 0.7407
```

```python
surf.vus            # 0.7407
surf.vus_adjusted   # 0.6889  -- see below
```

(The library does *not* use inclusion–exclusion — it sweeps a plane down the `S₃` axis in
`O(N log N)`. But on four points the two agree and the hand calculation shows what is
being measured.)

### 6.3 What the number means

**VUS is the probability that a randomly chosen requirement is satisfiable.** Draw
`(u, v, z)` uniformly at random from the cube — a random demand on the three
sensitivities. VUS is the chance the model can meet it with *some* setting of `w`.

### 6.4 The scale

| model | VUS |
|---|---|
| perfect | **1** |
| useless (scores unrelated to the label) | **1/6 ≈ 0.167** |
| perfectly, systematically wrong | **0** |

Why 1/6 and not 1/2? An uninformative classifier's operating points fill the chance plane
`S₁+S₂+S₃ = 1`, and the region under that plane is a corner simplex of the cube, whose
volume is `1/3! = 1/6`. In general the chance level for `c` classes is `1/c!`.

**This trips people up constantly**, so the library also reports

```
VUS_adjusted = (VUS − 1/6) / (1 − 1/6)
```

which is 0 at chance and 1 at perfect — the 3-class analogue of Gini `= 2·AUC − 1`. Our
toy model: VUS 0.7407, VUS_adjusted 0.6889.

### 6.5 Two variants

```python
surf.partial_vus((0.5, 0.5, 0.5))   # 0.1852 - VUS restricted to the box [0.5,1]^3,
                                    #   normalised. Use when only the high-sensitivity
                                    #   corner matters (analogue of partial AUC).
surf.vus_convex_hull                # 0.8827 - what becomes achievable if you are allowed
                                    #   to RANDOMISE between rules. The gap
                                    #   0.8827 - 0.7407 = +0.1420 is head-room available
                                    #   from better thresholding alone, model untouched.
```

---

## 7. VUS, definition 2: sorting a trio

This is the reading you will see quoted as *"the probability of correctly sorting a trio
of cases, one from each class"* (Mossman 1999). That sentence is unhelpfully compressed.
Here is the whole thing.

### 7.1 First, the binary version spelled out

For two classes the analogous game is **2-alternative forced choice (2AFC)**:

1. Draw one sample from the positive class and one from the negative class.
2. Shuffle them, and hide the labels. You are told *"one of these two is the positive
   one"* — but not which.
3. Your only tool is the model's score. **Guess that the higher-scoring one is the
   positive.**
4. `AUC` = the probability you are right.

That is exactly the table in §2.3. Nothing about thresholds — just ranking.

### 7.2 The 3-class version

Now the same game with three:

**Step 1 — form a trio.** Draw one sample from class 1, one from class 2, one from
class 3. Call them `x⁽ᵃ⁾`, `x⁽ᵇ⁾`, `x⁽ᶜ⁾`. In the toy data there are `3 × 3 × 3 = 27`
possible trios.

**Step 2 — hide the labels.** You are told: *"these three cases are one of each class"*.
You are not told which is which.

**Step 3 — you must hand out the three labels, one each.** Not "score each case
independently" — a **bijection**: label 1 goes to exactly one case, label 2 to another,
label 3 to the last. There are `3! = 6` ways to do that. Writing an assignment as
`σ = (σₐ, σ_b, σ_c)` meaning "give case `a` label `σₐ`, case `b` label `σ_b`, case `c`
label `σ_c`", the six are

```
(1,2,3)   (1,3,2)   (2,1,3)   (2,3,1)   (3,1,2)   (3,2,1)
```

### 7.2.1 "Surely you just read off each case's best class?"

This is the natural objection, and it deserves a straight answer, because **it is a
theorem — and it is usually not applicable.**

> **Proposition.** Let `m_j = argmax_k p_k(x_j)` be each case's own favourite class. *If*
> `(m_a, m_b, m_c)` happens to be a permutation (all three different), then it **is** the
> best assignment, and there is nothing to compute.
>
> *Proof.* For any assignment `σ`, `Π_j p_{σ(j)}(x_j) ≤ Π_j max_k p_k(x_j) = Π_j p_{m_j}(x_j)`.
> The right-hand side is an upper bound over all `σ`; if `m` is itself a permutation then
> that bound is attained by `σ = m`. ∎

So your instinct is right whenever it applies. The catch is the word *if*. Of the 27 trios
in the toy dataset, the three per-case favourites are all different in only **10**. In the
other **17** they are not a bijection at all:

```
(x1, x4, x9) -> each wants (1, 2, 2)      (x1, x6, x7) -> each wants (1, 3, 3)
(x1, x5, x9) -> each wants (1, 2, 2)      (x1, x6, x8) -> each wants (1, 3, 3)
(x2, x4, x9) -> each wants (1, 2, 2)      (x3, x6, x8) -> each wants (2, 3, 3)   ... and 11 more
```

Two cases both want class 2; nobody wants class 1. **The rule is not wrong here — it is
undefined.** You are forced to break the tie, and *how you break it* is precisely the
assignment problem. (Where it is defined, it agreed with the full computation 10 times out
of 10, exactly as the proposition promises.)

**Would a greedy tie-break do?** "Take the largest probability anywhere, lock that pair in,
delete its row and column, repeat." Reasonable, and it is right 24 times out of 27 here.
It fails on:

| trio | greedy says | the actual optimum | products |
|---|---|---|---|
| (x1, x6, x8) | (1,3,2) | **(1,2,3)** | 0.0875 vs **0.1225** |
| (x2, x6, x8) | (1,3,2) | **(1,2,3)** | 0.0688 vs **0.0963** |
| (x3, x6, x7) | (2,1,3) | **(1,2,3)** | 0.0473 vs **0.0735** |

This is the standard failure of greedy on an assignment problem: locking in one locally
attractive pair can force two bad ones. Note that in all three cases greedy is *pessimistic*
— it declares the model wrong when it was actually right. With only `3! = 6` candidates
there is no reason to approximate: just check them all.

**Step 4 — the key you sort on.** Score each candidate assignment by how plausible the
model finds it *as a whole*. Under independence, the likelihood of assignment `σ` is the
**product**

```
p_{σₐ}(x⁽ᵃ⁾) · p_{σ_b}(x⁽ᵇ⁾) · p_{σ_c}(x⁽ᶜ⁾)
```

or, in logs (which is what the code uses, and what keeps the arithmetic stable),

```
score(σ) = log p_{σₐ}(x⁽ᵃ⁾) + log p_{σ_b}(x⁽ᵇ⁾) + log p_{σ_c}(x⁽ᶜ⁾)
```

Pick the `σ` with the largest score.

**Those numbers are not arbitrary — they are posterior probabilities.** Given that the
three cases are one of each class, Bayes gives
`P(σ | the three cases) ∝ Π_j f_{σ(j)}(x_j)`, and writing the class-conditional density as
`f_k(x) = p_k(x)·g(x)/π_k` turns that into

```
P(σ | the three cases)  ∝  Π_j p_{σ(j)}(x_j) · [ Π_j g(x_j) ] / [ Π_j π_{σ(j)} ]
```

The marginal term `Π_j g(x_j)` does not depend on `σ`, and neither does `Π_j π_{σ(j)}` —
because `σ` is a *permutation*, that product is always `π₁π₂π₃`. Both cancel, leaving

```
P(σ | the three cases)  =  Π_j p_{σ(j)}(x_j)  /  Σ_{σ'} Π_j p_{σ'(j)}(x_j)
```

So dividing the six products by their sum gives an honest probability for each
arrangement, and picking the largest is the Bayes-optimal play. The scores are the whole
answer to "how sure is the model about this trio", not bookkeeping.

**Step 5 — did you get it right?** The true assignment is `(1,2,3)` (case `a` really is
class 1, etc.). `VUS₃AFC` is the fraction of trios where the winning `σ` is `(1,2,3)`.

Summary of the notation: *what is being sorted* = three cases into three labels;
*the key* = the total log-score of the whole assignment, equivalently its posterior
probability. It is an assignment problem, not three independent decisions — and §7.5
shows a case where that distinction changes the answer.

### 7.3 Worked example — a trio where your instinct already settles it

Trio `(x1, x4, x7)`. The relevant slice of the log-score table:

```
              log p₁      log p₂      log p₃
   x1        −0.3567     −1.6094     −2.3026
   x4        −1.6094     −0.5108     −1.6094
   x7        −2.3026     −1.6094     −0.3567
```

Each case's own favourite: x1 wants class 1 (`p₁ = 0.70`), x4 wants class 2 (`0.60`),
x7 wants class 3 (`0.70`). All different — so by the proposition in §7.2.1 **the answer is
already determined and no search is needed.** Here is the search anyway, to show it agrees
and to see what the numbers look like:

| assignment | how to read it | total log-score | product | posterior |
|---|---|---|---|---|
| **(1,2,3)** | x1→1, x4→2, x7→3 | **−1.2242** | **0.2940** | **0.808** ← winner, and true |
| (1,3,2) | x1→1, x4→3, x7→2 | −3.5756 | 0.0280 | 0.077 |
| (2,1,3) | x1→2, x4→1, x7→3 | −3.5756 | 0.0280 | 0.077 |
| (3,2,1) | x1→3, x4→2, x7→1 | −5.1160 | 0.0060 | 0.017 |
| (2,3,1) | x1→2, x4→3, x7→1 | −5.5215 | 0.0040 | 0.011 |
| (3,1,2) | x1→3, x4→1, x7→2 | −5.5215 | 0.0040 | 0.011 |

Check the winner by hand: `0.70 × 0.60 × 0.70 = 0.294`. ✓ **Correct.** The posterior
column is the product divided by the column total (0.364), so: *given these three cases,
the model is 80.8% confident of this arrangement.*

**This example on its own is misleading** — it is the easy case, where you could have read
the answer straight off. The next two are the ones that matter.

### 7.4 Worked example — a trio where the shortcut is undefined

Trio `(x3, x5, x9)` — the three hardest samples.

```
              log p₁      log p₂      log p₃
   x3        −1.2040     −0.7985     −1.3863
   x5        −1.0498     −0.9163     −1.3863
   x9        −1.2040     −0.9163     −1.2040
```

All three cases have the same favourite: x3 wants class 2 (`0.45`), x5 wants class 2
(`0.40`), x9 wants class 2 (`0.40`). **`(2,2,2)` is not a permutation**, so "read off each
case's best class" produces nothing at all here. You must choose *which* case gets to keep
class 2 and what the other two get — and that is a genuine optimisation.

| assignment | total log-score | product | posterior |
|---|---|---|---|
| **(2,1,3)** | **−3.0523** | **0.04725** | **0.223** ← winner |
| (1,2,3) — the truth | −3.3242 | 0.03600 | 0.170 |
| (3,1,2) | −3.3524 | 0.03500 | 0.165 |
| (2,3,1) | −3.3888 | 0.03375 | 0.159 |
| (1,3,2) | −3.5066 | 0.03000 | 0.142 |
| (3,2,1) | −3.5066 | 0.03000 | 0.142 |

The model prefers to call x3 "class 2" and x5 "class 1" — swapping them — because
`0.45 × 0.35 × 0.30 = 0.04725` beats the truth's `0.30 × 0.40 × 0.30 = 0.036`. **Wrong.**
The posteriors are all near 1/6, which is the model saying, correctly, that it has almost
no idea.

### 7.5 Worked example — the trio that proves pairwise checks are not enough

This is the one that justifies the whole apparatus. Trio `(x3, x6, x8)`:

```
        p₁      p₂      p₃
  x3   0.30    0.45    0.25       favourite: class 2
  x6   0.15    0.35    0.50       favourite: class 3
  x8   0.25    0.25    0.50       favourite: class 3
```

Favourites are `(2, 3, 3)` — again not a permutation. Now compare the truth against each
rival **one at a time**:

| rival | what it does to the truth | product | truth wins? |
|---|---|---|---|
| (1,2,3) — the truth | — | 0.052500 | — |
| (2,1,3) | swaps labels **1 ↔ 2** | 0.033750 | ✅ yes |
| (1,3,2) | swaps labels **2 ↔ 3** | 0.037500 | ✅ yes |
| (3,2,1) | swaps labels **1 ↔ 3** | 0.021875 | ✅ yes |
| (3,1,2) | 3-cycle | 0.009375 | ✅ yes |
| **(2,3,1)** | **3-cycle** | **0.056250** | ❌ **no** |

Read that carefully. **Every pairwise comparison says the model got this trio right.**
Class 1 beats class 2, class 2 beats class 3, class 1 beats class 3 — all three checks
pass. And yet the trio is sorted **wrong**, because rotating all three labels at once
(x3→2, x6→3, x8→1) scores higher: `0.45 × 0.50 × 0.25 = 0.05625` against the truth's
`0.30 × 0.35 × 0.50 = 0.0525`.

That gap is invisible to any amount of pairwise analysis. On the whole toy dataset:

```
trios passing all three PAIRWISE checks : 21 / 27
trios that are actually sorted correctly: 20 / 27
```

The missing one is exactly `(x3, x6, x8)`. **This is why VUS is not a function of the three
pairwise AUCs, and why conditions 4 and 5 in §8 exist** — they are the two 3-cycles, and
they have no two-class analogue. It is also the concrete reason a genuinely 3-class figure
of merit is not the same thing as an average of binary ones (§15, "why not just average
the three pairwise AUCs?").

### 7.6 Averaging over all trios

Doing that for all 27 trios: the correct assignment wins **20** times.

```
VUS₃AFC = 20/27 = 0.7407
```

The seven failures, and what the model said instead:

```
(x1, x6, x9) -> (1,3,2)      (x3, x5, x7) -> (2,1,3)      (x3, x6, x8) -> (2,3,1)
(x2, x6, x9) -> (1,3,2)      (x3, x5, x8) -> (2,1,3)      (x3, x6, x9) -> (2,3,1)
                             (x3, x5, x9) -> (2,1,3)
```

```python
from roc3 import vus_forced_choice
vus_forced_choice(y, P, classes=[1,2,3])     # 0.7407
```

### 7.7 The 3AFC profile — a free diagnostic

The six assignments' win-rates sum to 1, so they form a distribution. That distribution
tells you *which confusion* is costing you:

| assignment | win rate |
|---|---|
| (1,2,3) — correct | **0.7407** |
| (2,1,3) — classes 1 and 2 swapped | 0.1111 |
| (1,3,2) — classes 2 and 3 swapped | 0.0741 |
| (2,3,1) — a 3-cycle | 0.0741 |
| (3,1,2) | 0.0000 |
| (3,2,1) — classes 1 and 3 swapped | 0.0000 |

Read: the model's dominant error is confusing classes 1 and 2; it essentially never
confuses 1 with 3. That is actionable in a way a single scalar is not.

```python
from roc3 import forced_choice_profile
fc = forced_choice_profile(y, P, classes=[1,2,3], method="exact")
print(fc.profile_table(classes=[1,2,3]))
```

### 7.8 Why chance is 1/6 and perfect is 1

* **Uninformative model**: the scores carry no information about the labels, so all six
  assignments are equally likely to win → 1/6.
* **Perfect model**: `p(x) = (1,0,0)` for a class-1 case, etc. Only the true assignment
  has a nonzero product → 1.

### 7.9 Any number of classes

The same game with `c` classes is `HUM` (hypervolume under the manifold): draw one case
per class, assign the `c` labels bijectively to maximise the total log-score (a linear
assignment problem), and ask how often the identity wins. Chance is `1/c!`. At `c = 2` it
is **exactly** the AUC; at `c = 3` it is the VUS.

```python
from roc3 import hum
hum(y, P)      # works for any number of classes; no surface beyond 3
```

---

## 8. The algebraic shortcut

Enumerating six permutations per trio is `O(6·n₁n₂n₃)`. There is a closed form. Write

```
A(x) = log p₁(x) − log p₂(x)          (= D₁₂)
B(x) = log p₁(x) − log p₃(x)          (= D₁₃)
C(x) = log p₂(x) − log p₃(x)          (= D₂₃),      note B = A + C
```

Comparing the true assignment against each of the five rivals and cancelling common
terms, the truth wins **iff all five of these hold**:

| # | condition | which rival it beats |
|---|---|---|
| 1 | `A(x⁽ᵃ⁾) > A(x⁽ᵇ⁾)` | swap labels 1↔2 |
| 2 | `C(x⁽ᵇ⁾) > C(x⁽ᶜ⁾)` | swap labels 2↔3 |
| 3 | `B(x⁽ᵃ⁾) > B(x⁽ᶜ⁾)` | swap labels 1↔3 |
| 4 | `A(x⁽ᵃ⁾) + C(x⁽ᵇ⁾) > B(x⁽ᶜ⁾)` | the 3-cycle 1→2→3 |
| 5 | `B(x⁽ᵃ⁾) > A(x⁽ᵇ⁾) + C(x⁽ᶜ⁾)` | the 3-cycle 1→3→2 |

Conditions 1–3 are pure rank comparisons (the pairwise swaps of §7.5) — sample `a` must look more "1-vs-2" than sample
`b` does, and so on. Conditions 4 and 5 are the genuinely three-way ones — the two 3-cycles. They are
the reason VUS is not just an average of pairwise AUCs; §7.5 exhibits a trio that passes
1–3 and fails 4.

The `A, B, C` table for the toy data:

| sample | `A` | `B` | `C` |
|---|---|---|---|
| x1 | 1.2528 | 1.9459 | 0.6931 |
| x2 | 0.6061 | 1.2993 | 0.6931 |
| x3 | −0.4055 | 0.1823 | 0.5878 |
| x4 | −1.0986 | 0.0000 | 1.0986 |
| x5 | −0.1335 | 0.3365 | 0.4700 |
| x6 | −0.8473 | −1.2040 | −0.3567 |
| x7 | −0.6931 | −1.9459 | −1.2528 |
| x8 | 0.0000 | −0.6931 | −0.6931 |
| x9 | −0.2877 | 0.0000 | 0.2877 |

Counting trios that satisfy all five: **20/27 = 0.7407** — identical to the brute force.
Conditions 1 and 3 involve only two of the three indices, which lets the implementation
prune aggressively (`method="exact_fast"`, typically 5–20× faster).

Two useful invariances fall straight out of these conditions, because every one of them is
homogeneous and shifts equally on both sides:

* **Temperature scaling** `p ↦ p^α` does not change VUS₃AFC.
* **Class re-weighting** `p_k ↦ c_k p_k` does not change it either.

So VUS measures **ranking / separability, not calibration** — exactly like binary AUC.

---

## 9. Why the two definitions agree

On the toy data the volume (§6) and the trio game (§7) both give **20/27**. That is not a
coincidence: it is the 3-class version of the fact that binary AUC is simultaneously "the
area" and "the pair-ranking probability" (Mossman 1999; He, Gallas & Frey 2010).

The agreement is exact in the population for the ideal observer, and empirically it is
very tight in general: across ideal-observer and deliberately mis-calibrated models the
largest discrepancy measured was **0.0016**
([`IDEAS_LOG.md`](IDEAS_LOG.md) §F5). Both are computed and reported, and if they ever
disagree materially that is a signal worth investigating.

Practical differences worth knowing:

| | geometric VUS | VUS₃AFC |
|---|---|---|
| needs a rule grid? | yes (converges from below as resolution grows) | no |
| unbiased? | slightly low at coarse grids | yes (a U-statistic) |
| standard error | bootstrap | **closed form** |
| tied / constant scores | gives 0 for a constant scorer | gives 1/6 |
| gives you the surface? | **yes** | no |

Use the geometric one to draw and to choose thresholds; use the rank one to report a
number with an interval.

---

## 10. Reading the dashboard

```python
from roc3 import dashboard
dashboard(y, P, classes=[1,2,3], criterion="youden", path="roc3.png")
```

![clinical dashboard](../figures/01_clinical_youden.png)

| panel | what it shows | the question it answers |
|---|---|---|
| **A** | the surface in 3-D, drawn as the graph of `Z(u,v) = max{S₃ : S₁≥u, S₂≥v}`, with the chance triangle and the perfect corner | How good is the model, and what shape is the trade-off? |
| **B** | the **weight simplex** — a triangle where each point is a `w`, coloured by your objective | **Where do I set the weights?** |
| **C** | the same triangle with iso-sensitivity contours for all three classes | At weight `w`, what does each class get? |
| **D** | panel A as a contour map | Given requirements on classes 1 and 2, what is the best class-3 sensitivity? |
| **E** | iso-slices: the achievable `(S₁,S₂)` frontier subject to `S₃ ≥ z` for several `z` | If I must guarantee the dangerous class, what do I give up? |
| **F** | one-vs-rest ROC curves | Familiar reference — **but three thresholds here do *not* define a classifier** |
| **G** | the operating-point card: confusion matrix and the deployable rule | What exactly do I deploy, and what will it do? |

**On the weight simplex (panels B and C).** A triangle is the natural picture for `w`
because only ratios matter, so `w` can be normalised to sum to 1 — and points with three
non-negative coordinates summing to 1 *are* a triangle (barycentric coordinates). The
corner labelled "1, w=1" means `w = (1,0,0)`: all weight on class 1, i.e. "always predict
1". The centre is plain argmax. Moving toward a corner makes the classifier readier to
call that class.

`Z(u,v)` in panels A and D is the **monotone envelope**: for each pair of requirements on
classes 1 and 2, the best class-3 sensitivity still achievable. Its average height *is*
the VUS, so the picture and the headline number are literally the same object.

---

## 11. Choosing an operating point

This is the whole point. Reading a point off the surface must hand you a deployable rule.

### 11.1 The criteria

```python
op = surf.select("youden")
```

| criterion | maximises |
|---|---|
| `youden` | `S₁+S₂+S₃−1` (the generalised Youden index) |
| `balanced_accuracy` | `(S₁+S₂+S₃)/3` — *the same argmax as `youden`*; both names exist because both are in common use, and the equivalence is documented rather than hidden |
| `closest_to_perfection` | minimises the distance from `(S₁,S₂,S₃)` to `(1,1,1)` |
| `maximin` | `min_k S_k` — the worst-class guarantee |
| `accuracy` | `Σ_k π_k S_k` (prevalence-weighted) |
| `max_volume` | `S₁·S₂·S₃` |
| `expected_cost` | minimises `Σ_k π_k Σ_m cost[k,m]·C[k,m]/n_k` for your own 3×3 cost matrix — the only criterion that can see *which* wrong class was chosen |

On the toy data:

```
                youden  S = [0.667 0.667 1.000]   w = [0.309 0.289 0.401]
               maximin  S = [0.667 0.667 0.667]   w = [0.378 0.354 0.268]
 closest_to_perfection  S = [0.667 0.667 1.000]   w = [0.309 0.289 0.401]
              accuracy  S = [0.667 0.667 1.000]   w = [0.309 0.289 0.401]
            max_volume  S = [0.667 0.667 1.000]   w = [0.309 0.289 0.401]
```

(Several coincide here only because 9 samples give a very coarse surface. On real data
they separate — see [`README.md`](../README.md).)

### 11.2 Hard requirements

```python
op = surf.select("maximin", constraints={3: 0.99})   # Se for class 3 must be >= 0.99
print(op.describe())
```

```
------------------------------------------------------------------
OPERATING POINT  (maximin, constraints={3: 0.99})
------------------------------------------------------------------
Decision rule:  predict argmax_k  w_k * p_k(x)
      weights w = 1:0.3094, 2:0.2895, 3:0.4011
   (relative)   = 1:0.772, 2:0.722, 3:1.000
   log-ratios   a=log(w[2]/w[1])=-0.067   b=log(w[3]/w[1])=+0.259

Confusion (rows = true class):
                   ->1       ->2       ->3     Se
  true 1              2         1         0   0.6667
  true 2              0         2         1   0.6667
  true 3              0         0         3   1.0000

  accuracy 0.7778   balanced accuracy 0.7778   macro-F1 0.7746
  sensitivity [0.6667 0.6667 1.    ]   precision [1.     0.6667 0.75  ]
------------------------------------------------------------------
```

That block is the deliverable: multiply the model's three probabilities by
`(0.309, 0.289, 0.401)`, take the argmax, and you will see the confusion matrix shown.
If a requirement is impossible, `select` raises and tells you the best value each
constrained class could reach on its own.

### 11.3 The price of a guarantee

Sweeping a floor and re-optimising shows what a guarantee costs. On the clinical demo,
insisting on `Se[malignant] ≥ z` is **free up to z ≈ 0.85** and then bites hard:

![price of a guarantee](../figures/07_price_of_a_guarantee.png)

---

## 12. Uncertainty

```python
from roc3.inference import (vus_3afc_inference, bootstrap_vus,
                            permutation_test_vus, bootstrap_operating_point)

vus_3afc_inference(y, P, classes=[1,2,3])
#  {'vus': ..., 'stderr': ..., 'ci': (lo, hi), 'p_value': ...}
```

* **`vus_3afc_inference`** — a closed-form standard error. `VUS₃AFC` is a three-sample
  U-statistic, so `Var ≈ ζ₁/n₁ + ζ₂/n₂ + ζ₃/n₃`, where `ζ_k` is the variance of the
  conditional mean given one of the three arguments. Those conditional means fall out of
  the same triple sweep, so nothing is resampled. Agrees with the bootstrap to within 6%.
* **`bootstrap_vus`** — stratified bootstrap (resample within each class). The
  general-purpose fallback, and the only practical route for the geometric VUS.
* **`permutation_test_vus`** — tests `H₀: the scores carry no information` by shuffling
  labels. The null distribution centres on 1/6, as it should.
* **`bootstrap_operating_point`** — a confidence region for `(S₁,S₂,S₃)` at a **fixed**
  rule. This is the honest uncertainty of what you deploy; it deliberately does *not*
  include the optimism from having chosen `w` on the same data.

---

## 13. Ordered classes: the ordinal mode

If the three classes are naturally ordered — mild < moderate < severe, stage I < II < III
— and a **single** number `s(x)` drives the decision, there is a better rule family: two
cut-points.

```
ŷ = 1  if  s < t₁
  = 2  if  t₁ ≤ s < t₂
  = 3  if  s ≥ t₂
```

Two free parameters again, so again a surface — but now the parameters are *literal
cut-off values on the marker*, which you can type straight into production.

```python
from roc3 import ordinal_roc_surface, vus_ordinal
from roc3.plots import ordinal_dashboard

osurf = ordinal_roc_surface(y, marker)   # classes in increasing order
vus_ordinal(y, marker)                   # P(s(X₁) < s(X₂) < s(X₃))
```

Here "sorting the trio" is **literal**: draw one case from each class, look at their three
marker values, and ask whether they come out in the order `s₁ < s₂ < s₃`. That is the
classical Nakas–Yiannoutsos VUS, and it is where the phrase in §7 comes from.

![ordinal dashboard](../figures/05_ordinal.png)

---

## 14. Glossary of every symbol

| symbol | read as | defined in |
|---|---|---|
| `x` | one input sample | §1.1 |
| `y` | its true class, in `{1,2,3}` | §1.1 |
| `n`, `n_k` | total sample count; count in class `k` | §1.1 |
| `π_k` | prevalence of class `k`, `n_k/n` | §1.1 |
| `p_k(x)` | the model's score for class `k` on input `x` | §1.1 |
| `P` | the `n × 3` score matrix | §1.1 |
| `D₁₂, D₁₃, D₂₃` | pairwise log-score ratios, e.g. `log p₁ − log p₂` | §1.3 |
| `A, B, C` | the same three quantities, named for the 3AFC conditions | §8 |
| `w = (w₁,w₂,w₃)` | class weights defining a rule | §3.2 |
| `ŷ(x; w)` | the prediction of the rule with weights `w` | §3.2 |
| `a, b` | log weight ratios `log(w₂/w₁)`, `log(w₃/w₁)` | §3.5 |
| `Δ²` | the 2-simplex — all `w ≥ 0` summing to 1; drawn as a triangle | §3.2, §10 |
| `C[k,m]` | confusion count: true class `k`, predicted `m` | §4.1 |
| `S_k` | sensitivity / true class fraction for class `k` | §4.2 |
| operating point | the triple `(S₁,S₂,S₃)` for one rule | §4.2 |
| chance plane | `S₁+S₂+S₃ = 1` | §4.3 |
| down-set / achievable region `R` | all requirement triples the model can meet | §6.1 |
| Pareto front | operating points nothing else dominates on all three axes | §6.2 |
| VUS | volume of `R`; equivalently the trio-sorting probability | §6, §7 |
| `VUS_adjusted` | `(VUS − 1/6)/(1 − 1/6)`; chance 0, perfect 1 | §6.4 |
| `Z(u,v)` | the monotone envelope, `max{S₃ : S₁≥u, S₂≥v}` | §10 |
| σ | an assignment of the three labels to the three cases | §7.2 |
| 2AFC / 3AFC | the two- / three-alternative forced-choice game | §7.1, §7.2 |
| HUM | the `c`-class version of VUS | §7.8 |

---

## 15. Traps and FAQ

**"VUS = 0.5 sounds terrible."** It is not. Chance is **1/6 ≈ 0.167**, not 1/2. A VUS of
0.5 is a decent 3-class model. Always read `VUS_adjusted` alongside — 0.5 raw is 0.40
adjusted.

**"Can't I just pick three thresholds, one per class?"** No — and this is the single most
common mistake. Three independent thresholds do not define a classifier: a sample can
clear two of them, or none. That is why the one-vs-rest ROC panel is labelled *reference
only*. The weights `w` couple the three decisions, which is exactly what a 3-class
trade-off is.

**"Why not just average the three pairwise AUCs?"** You can (`hand_till_m`), and the
library reports it. But it is a scalar with no surface, its chance level is 1/2 so it is
not the volume of anything, and it is blind to genuinely three-way confusions. §7.5 has an
explicit trio that passes all three pairwise checks and is still sorted wrong, because
rotating all three labels at once beats the truth. Conditions 4 and 5 in §8 are exactly
those two rotations, and they have no two-class analogue.

**"Surely the winning assignment is just each case's most likely class?"** When those
three favourites happen to be all different, yes — provably (§7.2.1). But on the toy data
that happens in only 10 of 27 trios; in the other 17 two cases want the same class, so the
rule returns something that is not a permutation and is therefore not an answer at all.
Greedy tie-breaking is a reasonable repair and is right 24 times out of 27 — but with only
six candidates there is no reason to approximate.

**"My VUS went up but accuracy went down."** Expected. VUS measures *ranking*, invariant
to temperature and to class re-weighting (§8). Accuracy measures one specific operating
point. A model can rank perfectly and still be badly calibrated; the surface tells you
what re-weighting would buy.

**"The VUS changed when I increased `resolution`."** By design. The geometric VUS
converges **from below** as the rule grid is refined, roughly as `O(1/R)`. The default
`resolution=160` is about 0.005 low; use 240–320 for anything you report. `VUS₃AFC` has no
grid and does not move.

**"Two thresholds/criteria gave me the same answer."** `youden` and `balanced_accuracy`
are affine transforms of each other and *always* pick the same rule. On small or coarse
surfaces others can coincide too.

**"My scores aren't probabilities."** Pass `score_type="log"` for logits or log-scores.
Only differences of log-scores matter, so an arbitrary per-row constant is irrelevant and
un-normalised logits are fine.

**"I have more than three classes."** The plot stops at three — that is geometry, not a
missing feature. `hum()` gives the scalar for any `c`, with chance `1/c!`.

**"My domain logic forces `p₁ < p₂ < p₃`."** Then read
[`CONSTRAINED.md`](CONSTRAINED.md) — the answer depends on whether the constraint is on
the model's output (it changes nothing) or on the truth (it caps the VUS strictly below 1).

---

## Where to go next

* [`PLAN.md`](PLAN.md) — the design: every approach considered and why this one won.
* [`IDEAS_LOG.md`](IDEAS_LOG.md) — every idea tried, the dead ends, the bugs, the measurements.
* [`REFERENCES.md`](REFERENCES.md) — annotated bibliography.
* [`CONSTRAINED.md`](CONSTRAINED.md) — ROC under a logical ordering constraint.
* `experiments/01_validation.py` — the correctness suite.
* `experiments/03_threshold_case_study.py` — a full threshold-selection walkthrough.
* `experiments/05_tutorial_numbers.py` — regenerates every number in this file.
