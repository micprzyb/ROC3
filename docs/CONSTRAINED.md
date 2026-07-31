# ROC analysis under the logical constraint `0 < p₁ < p₂ < p₃ < 1`

**Research note.** Code: [`roc3/constrained.py`](../roc3/constrained.py).
Tests: [`experiments/04_constrained.py`](../experiments/04_constrained.py) — every
proposition below is stated, then attacked, then reported.

---

## 0. The sentence is ambiguous, and the two readings point opposite ways

> *"the probabilities must satisfy `0 < p₁ < p₂ < p₃ < 1` and this is a logical
> constraint. Moreover, the model was trained in such a way that it obeys the
> constraint."*

There are two things "the probabilities" can mean, and it matters enormously which:

| | **Reading B — the model's output** | **Reading A — the world** |
|---|---|---|
| what is constrained | `p̂(x)`, the score vector the model emits | `P(Y=k \| x)`, the true posterior |
| how it is enforced | architecture (ordered softmax, a Semantic Probabilistic Layer, a projection) | it is a fact; nothing enforces it |
| effect on the ROC surface | **none whatsoever** | **cuts ROC space with three planes** |
| effect on VUS | **none** — pure gauge | **caps it strictly below 1** |
| what breaks | `argmax`, and the readability of the weight simplex | accuracy as a metric; the whole 0–1 scale |

The second sentence of the prompt ("the model was trained to obey it") is Reading B.
The first ("the probabilities **must** satisfy … this is a **logical** constraint") is
Reading A. Both are in play, so this note does both. They are treated separately because
mixing them is the trap: Reading B is a non-event dressed up as a constraint, and Reading
A is a hard limit that is easy to miss.

**How to tell which you are in.** Reading A implies `π₁ < π₂ < π₃` (§3.1). If the
observed prevalences are *not* ordered, the constraint cannot be a fact about the world —
either it is a model-side gauge, or the model is misspecified.
`roc3.constrained.constrained_report` checks this and says so.

---

## 1. Reading B: the output-ordering constraint is a **gauge**

### 1.1 The invariance

Every rule in the family is `ŷ(x) = argmax_k w_k p_k(x)`. Rescale the model's columns by
a constant vector `c`:

```
argmax_k  w_k · (c_k p_k(x))  =  argmax_k  (w_k c_k) · p_k(x)
```

Rescaling the **model** is the same as rescaling the **rule**. So the *set* of achievable
operating points is identical, hence the ROC surface is identical, hence the VUS is
identical. The rank-based `VUS₃AFC` is invariant for the same reason (proved
independently in [`PLAN.md`](PLAN.md) §3.4: each of its five conditions shifts by the same
constant on both sides).

### 1.2 Theorem — every model has an ordered representative

> For any score matrix there is a single vector `c` such that `p_k → c_k p_k` satisfies
> `p₁(x) < p₂(x) < p₃(x)` for **every** `x`, with the ROC surface and both VUS estimators
> unchanged.

*Proof.* Write `A(x) = log p₁ − log p₂` and `C(x) = log p₂ − log p₃`. A class rescaling
shifts `A` by `log(c₁/c₂)` and `C` by `log(c₂/c₃)`, uniformly over `x`. Choose the shifts
to be `−(max_x A + m)` and `−(max_x C + m)` for any `m > 0`. Then `A(x) ≤ −m < 0` and
`C(x) ≤ −m < 0` for all `x`, which is exactly `p₁ < p₂ < p₃`. ∎

**Consequence.** The output-ordering constraint carries *no information about ranking
quality*. It costs nothing and buys nothing. A constrained model can be a perfect
ranker (`VUS = 1`), a chance ranker, or anti-correlated.

*Verified* (P1): for three models the regauged version satisfies the constraint on 100%
of rows and reproduces the geometric VUS and `VUS₃AFC` **to the last printed digit**, with
a bit-identical set of operating points. And a constrained model with `VUS = 1.000000`
is constructed explicitly.

### 1.3 What the constraint *does* break

**(a) `argmax` collapses.** If `p₃` is always the largest, `predict()` returns class 3 for
every input. Its accuracy is exactly the prevalence `π₃` and its balanced accuracy is
`1/3`. Anything downstream that calls `.predict()` — a sklearn `classification_report`,
a monitoring dashboard, a fairness audit — is measuring a constant function.
*Verified* (P2): argmax accuracy `0.1667` = prevalence, per-class sensitivity `[0, 0, 1]`,
while the surface is untouched at `VUS = 0.6918`.

**(b) The weight simplex becomes unreadable.** The rules that do anything useful need
`w₁ ≫ w₂ ≫ w₃`, so the informative region is crushed into one corner of the ternary
panel. On the demo the optimum sits at a weight ratio of **9.2 × 10⁸**.

**(c) The fix is free: re-gauge before plotting.** `gauge_normalize` divides column `k`
by a reference `r_k` — the per-column geometric mean, or the class prior (which makes
plain `argmax` the *maximum-likelihood* rule `argmax_k p_k/π_k`). Since this is a class
rescaling, **no number changes**: same surface, same VUS, same selected operating point,
same confusion matrix. Only the coordinates move.
*Verified* (P2): weight ratio drops from `9.2 × 10⁸` to `14.5`; VUS identical to 10
decimal places; the selected operating point is literally the same triple.

**(d) There is a floating-point cost.** The gauge shift equals the *spread* of the model's
log-odds, so a confident model needs `p₁` far below float64's floor:

| separation | min log p needed | representable? |
|---|---|---|
| 1.2 | −27 | yes |
| 2.5 | −68 | yes |
| 6.0 | −256 | yes |
| 20.0 | −2064 | **no** |

`to_ordered_gauge` raises with that diagnosis and offers `return_log=True`, where the
identity still holds exactly. This also flushed out a real bug in the main library —
see [`IDEAS_LOG.md`](IDEAS_LOG.md) §G9.

**(e) The chance level does not move.** An uninformative constrained model still scores
`1/6`: if the scores are independent of the label, all six permutations are equally likely
by exchangeability, whatever region of the simplex the scores live in. *Verified* (P3).

---

## 2. Reading A: the constraint on the world cuts ROC space

Now assume the *true posterior* is ordered: `π₁f₁(x) < π₂f₂(x) < π₃f₃(x)` for all `x`,
where `f_k` is the class-conditional density. This is a statement about the problem, and
it binds **every classifier**, constrained or not, calibrated or not.

### 2.1 The order chamber, and why there must be a cap

The ordered part of the probability simplex,
`T = {p : p₁ ≤ p₂ ≤ p₃}`, is a triangle with vertices

```
v₀ = (1/3, 1/3, 1/3)      v₁ = (0, 1/2, 1/2)      v₂ = (0, 0, 1)
```

Read those off. **The most favourable evidence the logic permits for class 1 is the
uniform posterior** — you can never be more than 1/3 sure of class 1. For class 2 the
ceiling is 1/2. Only class 3 can ever be identified with certainty. The label noise on
classes 1 and 2 is *irreducible*, and no amount of data or model capacity touches it.
Everything below is that observation made quantitative.

### 2.2 The master lemma

Let `R₁, R₂, R₃` be **any** partition of the input space — any classifier at all — and let
`C[k, m] = P(x ∈ R_m | Y = k)` be its row-stochastic confusion matrix. Integrating
`π_i f_i < π_j f_j` over `R_m`:

> **Lemma.**  `π_i · C[i, m]  ≤  π_j · C[j, m]`  for all `i < j` and all `m`.  (★)

Nine linear inequalities. That is the whole of Reading A; everything else is a corollary.

### 2.3 Corollaries

**(1) The priors are ordered.** Sum (★) over `m`: `π_i ≤ π_j` for `i < j`.
The degenerate case is instructive: if `π_i = π_j` then (★) forces `C[i,·] = C[j,·]`,
i.e. **classes `i` and `j` are indistinguishable to every classifier**, and the ceiling
collapses to chance. An ordered posterior with balanced priors is not a hard problem — it
is an impossible one. This is why the diagnostic in §0 works: balanced prevalences plus
ordered outputs can only be Reading B.

**(2) The perfect corner is unreachable.** `S_k = C[k,k] = 1` for all `k` forces
`C[2,1] = 0`, and (★) with `i=1, j=2, m=1` then gives `π₁ · 1 ≤ 0`. Contradiction.
More usefully, relaxing `C[j,m] ≤ 1 − S_j` gives three explicit cutting planes:

```
π₁S₁ + π₂S₂ ≤ π₂        π₂S₂ + π₃S₃ ≤ π₃        π₁S₁ + π₃S₃ ≤ π₃
```

**(3) Accuracy is a dead metric.** For `k < 3`, apply (★) with `i=k, j=3, m=k`:

```
Σ_k π_k C[k,k]  ≤  Σ_k π₃ C[3,k]  =  π₃
```

so **`max accuracy = π₃`, attained by the constant rule "always predict 3"**. No
classifier can beat guessing the majority class. (Equivalently: the Bayes error is
`1 − E[max_k p_k] = 1 − E[p₃] = 1 − π₃`.) *Verified* (P4): the LP reproduces the closed
form to 6 decimals for every prior tested.

**(4) Every pairwise AUC is capped.** Two independent derivations, which agree:

*Via bounded likelihood ratios.* (★) pointwise says `L = f_i/f_j ≤ R := π_j/π_i`. Now
`AUC = E_j[L·F_j(L)]` by change of measure, where `F_j` is the CDF of `L` under `f_j`.
Substituting `V = F_j(L) ~ U(0,1)` and `L = q(V)` with `q` non-decreasing, we must
maximise `∫₀¹ v·q(v) dv` subject to `∫₀¹ q = 1`, `0 ≤ q ≤ R`. The extremal `q` puts all
its mass as high as possible: `q = R·1[v ≥ 1 − 1/R]`, giving

```
AUC(i, j)  ≤  1 − 1/(2R)  =  1 − π_i / (2 π_j)
```

The bound is tight (take `L ∈ {0, R}` with `P_j(L = R) = 1/R`). Sanity: `R = 1` ⟹
AUC ≤ 1/2; `R → ∞` ⟹ AUC → 1.

*Via the polytope.* Specialising §2.4 to two classes gives the down-set area
`∫₀¹ (1 − (π_i/π_j)·s) ds = 1 − π_i/(2π_j)` — **the same number**. *Verified* (P5): the
polytope construction reproduces `1 − π₁/(2π₂)` to 9 decimals for four prior pairs.

**(5) `VUS ≤ min over pairs of AUC(i,j)`**, since correct 3-AFC assignment requires each
pairwise comparison to come out right (conditions 1–3 of [`PLAN.md`](PLAN.md) §3.3).

### 2.4 The ceiling, exactly

The feasible confusion matrices form a polytope: 3 simplex rows + the 9 inequalities (★).
The achievable sensitivity triples are its projection onto the diagonal — a convex
polytope, computable by enumerating vertices. The down-set volume of that projection is

> **`VUS_max(π)` — the largest VUS any classifier can attain on any world with priors `π`.**

Two implementations, cross-checked: exact vertex enumeration + hull volume
(`method='exact'`), and integrating a per-cell LP `max S₃ s.t. S₁ ≥ u, S₂ ≥ v`
(`method='lp'`). They agree to `1e-5` (P5).

**The bound is not vacuous — it is brutal:**

| priors `π` | `VUS_max` | max balanced acc. | max accuracy |
|---|---|---|---|
| (0.320, 0.330, 0.350) | **0.191** | 0.363 | 0.350 |
| (0.250, 0.330, 0.420) | **0.304** | 0.486 | 0.420 |
| (0.167, 0.333, 0.500) | **0.444** | 0.611 | 0.500 |
| (0.100, 0.250, 0.650) | **0.618** | 0.739 | 0.650 |
| (0.050, 0.200, 0.750) | **0.744** | 0.828 | 0.750 |
| (0.020, 0.080, 0.900) | **0.832** | 0.887 | 0.900 |
| (0.005, 0.045, 0.950) | **0.921** | 0.947 | 0.950 |

Chance is 0.167. With near-equal priors the ceiling is **0.191** — the constraint leaves
almost no room above chance at all. The constraint is only survivable when the priors are
strongly ordered, which makes sense: `π_j/π_i` is exactly the likelihood-ratio budget.

---

## 3. So how should we report?

**Renormalise.** The attainable range is `[1/6, VUS_max(π)]`, not `[1/6, 1]`:

```
VUS_normalized = (VUS − 1/6) / (VUS_max(π) − 1/6)
```

0 at chance, 1 at the best any classifier could do given the logic.

On the worked example (P7): a model scores `VUS = 0.550`. Against the usual scale that
reads as mediocre. Against the ceiling of `0.610` it is capturing **86.6%** of everything
attainable, and the honest report is `VUS_normalized = 0.866`. Reporting 0.550 against a
maximum of 1.000 would have understated the model by a factor that has nothing to do with
the model.

`constrained_report` / `format_constrained_report` print the whole picture: constraint
compliance, the argmax degeneracy warning, raw and normalised VUS, the ceiling, and the
scalar ceilings on accuracy, balanced accuracy and worst-class sensitivity.

![ceiling](../figures/08_constrained_ceiling.png)

---

## 4. What is *not* settled

**The ceiling is a bound over all classifiers, and a single world cannot saturate it.**
`VUS_max(π)` is the down-set volume of the polytope of *all* feasible confusion matrices.
A fixed world exposes only a 2-parameter family of Bayes rules (the weighted-argmax
sweep), which traces the Pareto frontier of *that world's* achievable set — a subset. So
`sup over worlds of VUS ≤ VUS_max(π)`, possibly strictly.

Measured (P6), building worlds by placing mass in the order chamber and sampling
`y ~ Cat(p(x))` so the true posterior is ordered by construction:

| atoms | fraction of the attainable volume captured |
|---|---|
| 3 (the chamber vertices) | 37.3% |
| 40 | 65.6% |
| 120 | 85.6% |
| 400 | 87.7% |
| 1200 | **89.3%** |
| uniform lattice over the chamber | 53.5% |

Richness helps and mass concentrated near the chamber's extreme points helps (Dirichlet
concentration 0.12 beats the uniform lattice by 36 points, which is what the LP predicts —
the optimum is at vertices). Whether the supremum equals `VUS_max(π)` is **open**. Every
one of 80 randomly generated constrained worlds respected the ceiling, with a tightest
observed slack of `+0.118`.

Also open:

1. **A direct VUS ceiling** that does not go through the polytope — a 3-class analogue of
   the `1 − 1/(2R)` change-of-measure argument.
2. **A tighter, world-aware ceiling.** Given an estimate of the posterior distribution
   (not just the priors), the achievable set shrinks. That would turn `VUS_normalized`
   from "fraction of what is logically possible" into "fraction of what is possible on
   *this* problem".
3. **Partial orders.** If the logic only says `p₁ < p₃` (not the full chain), the polytope
   has fewer facets and the ceiling rises. The machinery here handles any set of
   inequalities of the form (★); only the constraint list changes.
4. **Estimating the ceiling when the priors are uncertain.** `VUS_max` is quite sensitive
   to `π` (§2.4), so a confidence interval on the ceiling belongs next to the point
   estimate.

---

## 5. Summary

* An ordering constraint on the **model's output** is a **gauge choice**. It changes no
  ROC quantity at all. It breaks `argmax` and makes the weight simplex unreadable; both
  are fixed for free by `gauge_normalize`, which also changes nothing.
* An ordering constraint on the **world** is a hard limit. It caps every pairwise AUC at
  `1 − π_i/(2π_j)`, caps accuracy at `π₃` (so the constant rule is Bayes-optimal), makes
  the perfect corner unreachable, and caps the VUS at a computable `VUS_max(π) < 1`.
* Therefore: **stop normalising by 1.** Normalise by the ceiling.
* The mechanism, in one line: *the constraint says you can never be more than 1/3 sure of
  class 1, or more than 1/2 sure of class 2.*

## References for this note

* Boyd, K., Davis, J., Page, D. (2012). *Unachievable region in precision-recall space and
  its effect on empirical evaluation.* ICML. <https://arxiv.org/pdf/1206.4667> — the
  precedent: a region of evaluation space that the problem's structure, not the model,
  puts out of reach.
* Ahmed, K., Teso, S., Chang, K.-W., Van den Broeck, G., Vergari, A. (2022). *Semantic
  Probabilistic Layers for Neuro-Symbolic Learning.* NeurIPS.
  <https://arxiv.org/abs/2206.00426> — how a model is "trained to obey" a hard logical
  constraint with guaranteed test-time satisfaction. This is the Reading-B mechanism.
* Standard ROC/VUS background is in [`REFERENCES.md`](REFERENCES.md).
