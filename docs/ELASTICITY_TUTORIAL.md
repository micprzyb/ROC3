# Tutorial: grading a price-elasticity model that has no labels

The long-form version of [`ELASTICITY.md`](ELASTICITY.md). Nothing is assumed beyond
[`PRICETEST.md`](PRICETEST.md) §1. Every symbol, convention and term is defined in
§1.3–§1.5 and repeated in the [glossary](#11-glossary), and **every number below is
real** — regenerate the small worked example with

```bash
.venv/bin/python experiments/09_elasticity_tutorial_numbers.py   # the 900-customer toy
.venv/bin/python experiments/08_elasticity.py                    # the 600k-customer run
```

The toy is deliberately *deterministic* — purchase counts are constructed to hit their
expected values exactly — so every quantity can be checked with a calculator. The
statistics are left to the big run.

Contents

1. [What "elasticity" means here](#1-what-elasticity-means-here)
2. [What the data contains, and what it never will](#2-what-the-data-contains-and-what-it-never-will)
3. [The toy world](#3-the-toy-world)
4. [Fact 1 — your arm classifier already *is* an elasticity model](#4-fact-1--your-arm-classifier-already-is-an-elasticity-model)
5. [Fact 2 — the best price is a point on the ROC surface](#5-fact-2--the-best-price-is-a-point-on-the-roc-surface)
6. [Validation 1 — the group trick](#6-validation-1--the-group-trick)
7. [Validation 2 — what the policy is worth](#7-validation-2--what-the-policy-is-worth)
8. [Validation 3 — the doubly-robust pseudo-outcome, and two traps](#8-validation-3--the-doubly-robust-pseudo-outcome-and-two-traps)
9. [Comparing two models](#9-comparing-two-models)
10. [The recipe](#10-the-recipe)
11. [Glossary](#11-glossary)
12. [FAQ](#12-faq)

---

## 1. What "elasticity" means here

### 1.1 The definition

**Price elasticity of demand** measures how much demand responds to price, as a ratio of
*proportional* changes:

```
ε  =  − (proportional change in quantity) / (proportional change in price)
```

The minus sign is a convention that makes `ε` positive for an ordinary product — one whose
demand falls when its price rises. `ε = 1` means a 1% price rise costs you 1% of demand;
`ε = 3` means it costs you 3%.

Proportional changes are logs, so the clean version is

```
ε  =  − d log(demand) / d log(price)
```

Write **`β(p)`** for demand at price `p`. We only have three prices, so instead of a
derivative we use the **arc elasticity** — the same thing as a difference quotient between
two of them:

```
ε  =  − [ log β(p_high) − log β(p_low) ] / [ log p_high − log p_low ]
```

### 1.2 It is a property of a *customer*

`β(p)` here is not market demand. It is **one person's** probability of buying at price
`p`. So elasticity is that person's own sensitivity: a bargain-hunter has a large `ε`, a
must-have-it buyer has a small one. That per-customer quantity is what we want to model,
and what we can never observe.

### 1.3 Symbols

| symbol | meaning |
|---|---|
| `x` | a customer's **covariates** — whatever features you have on them (age, history, channel, …) |
| `k` | the **arm**, i.e. one price cell of the experiment. `1` = cheapest, `2` = middle, `3` = dearest |
| `m_k` | the **price multiplier** of arm `k`; here `(0.9, 1.0, 1.1)` for −10% / 0% / +10% |
| `q_k` | the **randomisation probability** (also called the **propensity**) of arm `k` — the chance a customer is put in it. Here `1/3` each, and known by design because you chose it |
| `β_k(x)` | `P(buy \| customer x, shown arm k)` — the customer's **demand curve**, three numbers. This is `β(p)` of §1.1 evaluated at that customer and at price `m_k`. **Never observed** |
| `Y` | 1 if the customer bought, 0 otherwise. **Observed** |
| `K` | which arm the customer was actually shown. **Observed** |
| `p_k(x)` | `P(arm = k \| x, bought)` — what your arm classifier predicts |
| `ε(x)` | customer `x`'s arc elasticity |
| `c(x)` | a per-customer normalising constant that appears in §4 and cancels everywhere |
| `d` | a **policy** — a rule that assigns one price to each customer. `d(x)` is the arm it picks for `x`. (The causal-inference literature usually writes `π` here; this repo reserves `π` for class prevalence, so the policy is `d`, for *decision*.) |
| `V(d)` | the **policy value**: average revenue per customer if you priced according to `d` |
| `τ(x)` | the **uplift** `β₁(x) − β₃(x)` (§8) — an absolute difference, *not* the elasticity |
| `ψ_i` | the doubly-robust **pseudo-outcome** for customer `i` (§8) |
| `D_k` | a **purchase rate**: the fraction of the customers shown arm `k` who bought. Computed within a group when we bin (§6) |
| `n_k` | how many customers were shown arm `k` (within whatever group is under discussion) |
| `w` | the weight vector of `roc3`'s decision rule `argmax_k w_k p_k(x)` — see [`TUTORIAL.md`](TUTORIAL.md) §3 |

### 1.4 Conventions

| notation | meaning |
|---|---|
| `β̂`, `p̂`, `ε̂` | a **hat** means "estimated from data", as opposed to the unknown true value |
| `1{A}` | the **indicator**: 1 if statement `A` is true, 0 if not. Used to pick out a subset inside an average |
| `argmax_k f(k)` | the **value of `k`** that makes `f(k)` largest — not the largest value itself. `argmax` of `(0.4, 0.9, 0.2)` is 2 |
| `a ∝ b` | "`a` is **proportional to** `b`": they differ by a positive factor that does not depend on the index being varied |
| `E[Z \| x]` | the **conditional expectation** of `Z` — its average over all customers having covariates `x` |
| `Σ_j` | sum over all arms `j` |
| `sd(Z)` | the **standard deviation** of `Z` — the typical size of its fluctuation |
| **standard error** | the standard deviation of an *estimate*, i.e. how much it would move if you reran the experiment. Written `± …` throughout |
| **p-value** | the probability of seeing a result at least this extreme if the stated null hypothesis were true. Small = the null is hard to sustain |

### 1.5 Words that carry weight

* **Identified** — a quantity is *identified* if it is pinned down exactly by the
  distribution you can observe, given infinite data. Unidentified means no estimator can
  ever recover it, however much data you collect: the information simply is not there.
  §2 turns on the fact that elasticity is unidentified per customer and identified per
  group.
* **Oracle** — a hypothetical model handed the true answer. Used to grade the *method*: if
  a check cannot recognise the oracle, the check is broken.
* **Out-of-fold** — a prediction for a customer made by a model that was fitted *without*
  that customer, e.g. via cross-validation. Needed whenever predictions are used to form
  groups (§6.5).
* **Nuisance model** — a model you do not care about in itself, fitted only because some
  estimator needs it as an input. `β̂` inside `ψ` (§8) is one.
* **Precision** — the reciprocal of the variance. "Weighting by precision" means trusting
  the better-measured bins more.

**Notation is shared across this repo.** `π_k` always means the *prevalence* of class or
arm `k` ([`TUTORIAL.md`](TUTORIAL.md), [`CONSTRAINED.md`](CONSTRAINED.md),
[`PRICETEST.md`](PRICETEST.md)), never a policy — which is why the policy above is `d`.
`m_k`, `q_k`, `β_k`, `p_k` and `D_k` mean the same thing here as in
[`PRICETEST.md`](PRICETEST.md).

One constant recurs everywhere: `log(m₃/m₁) = log(1.1/0.9) = **0.20067**`. It is the
denominator of every elasticity in this document.

---

## 2. What the data contains, and what it never will

Here is what a price test actually hands you — one row per customer:

| customer | features `x` | arm shown `K` | bought `Y` |
|---|---|---|---|
| 1 | … | 0.9 | 1 |
| 2 | … | 1.1 | 0 |
| 3 | … | 1.0 | 1 |
| 4 | … | 0.9 | 0 |
| … | … | … | … |

Look at customer 1. They saw the cheap price and bought. That is **one point** on their
demand curve. What would they have done at 1.0? At 1.1? Those events did not happen and
never will. Their elasticity is a contrast between one thing that happened and two that
did not.

> **There is no elasticity column, and no amount of cleverness will produce one.** Every
> supervised-learning instinct — hold out a test set, compare predictions to labels — is
> unavailable, because there are no labels.

This is not a data-collection failure you could fix with a better pipeline. It is the
fundamental problem of causal inference, and it applies to every individual treatment
effect ever estimated.

**The way out is to stop asking about individuals.** Elasticity is not identified for one
person, but it is *exactly* identified for any **group** — because the arms were
randomised, any group you can name contains all three arms in known proportion, so its
demand curve is read straight off raw purchase counts. And a model's predictions are a
way of naming groups. §6 turns that into a validation procedure.

---

## 3. The toy world

Three kinds of customer, 100 of each in each arm, so 900 customers in total. The demand
curves are the thing we are pretending not to know:

| type | `β` at 0.9 | `β` at 1.0 | `β` at 1.1 | true arc elasticity |
|---|---|---|---|---|
| **L** (loyal) | 0.90 | 0.85 | 0.80 | **0.587** |
| **M** (middling) | 0.60 | 0.45 | 0.30 | **3.454** |
| **S** (sensitive) | 0.40 | 0.20 | 0.10 | **6.908** |

Worked, for type S: `−log(0.10 / 0.40) / 0.20067 = −(−1.38629) / 0.20067 = 6.9083`.

And here is what the experiment records — purchase counts, nothing else:

| type | arm 0.9 | arm 1.0 | arm 1.1 |
|---|---|---|---|
| L | 90/100 | 85/100 | 80/100 |
| M | 60/100 | 45/100 | 30/100 |
| S | 40/100 | 20/100 | 10/100 |

(The `β` table is invisible in reality; it is shown so we can grade the methods against a
truth they cannot see.)

---

## 4. Fact 1 — your arm classifier already *is* an elasticity model

### 4.1 The derivation

You already fitted `p_k(x) = P(arm = k | x, bought)`. Apply Bayes, conditioning on
purchase:

```
p_k(x)  =  q_k β_k(x)  /  Σ_j q_j β_j(x)
```

Call the denominator `c(x)`. It does not depend on `k` — it is the same number for all
three arms of the same customer. So with equal randomisation,

```
p_k(x)  =  β_k(x) / (3 c(x))      i.e.   p_k(x) ∝ β_k(x)
```

The posterior over arms **is the demand curve**, rescaled by an unknown positive factor.

Now take any *ratio* of two arms and that unknown factor cancels:

```
p₃(x) / p₁(x)  =  β₃(x) / β₁(x)
```

The elasticity is built from exactly such a ratio, so:

> ```
> ε(x)  =  − [ log p₃(x) − log p₁(x) − log(q₃/q₁) ] / [ log m₃ − log m₁ ]
> ```
>
> **exactly** — no approximation, no unidentified constant, no extra model.

(The `log(q₃/q₁)` term is zero under equal randomisation; keep it if your arms were not
allocated equally.)

### 4.2 Worked on the toy world

Posteriors are just the demand curves divided by their own row sums (`p_cheap`, `p_mid`
and `p_exp` are just readable names for `p₁`, `p₂`, `p₃`; `Σβ` is the row sum
`β₁ + β₂ + β₃`):

| type | Σβ | `p_cheap` | `p_mid` | `p_exp` |
|---|---|---|---|---|
| L | 2.55 | 0.3529 | 0.3333 | 0.3137 |
| M | 1.35 | 0.4444 | 0.3333 | 0.2222 |
| S | 0.70 | 0.5714 | 0.2857 | 0.1429 |

Recovering type S's elasticity from the posterior alone:

```
−log(0.1429 / 0.5714) / 0.20067  =  −log(0.25) / 0.20067  =  1.38629 / 0.20067  =  6.9083
```

which is the true value. Across all three types the discrepancy is at most **2.7 × 10⁻¹⁵**
— machine noise. On the 600,000-customer run the maximum error is **4.4 × 10⁻¹⁵**.

```python
from roc3.elasticity import elasticity_from_posterior
eps = elasticity_from_posterior(proba, price_multipliers=(0.9, 1.0, 1.1))
```

### 4.3 Two things to get right

* **Column order.** Cheapest arm first, throughout. Reversing it flips the sign of every
  elasticity.
* **Unequal randomisation.** If `q` is not uniform, pass it; the constant no longer
  cancels on its own.

### 4.4 What is *not* recovered

The level. `c(x)` is genuinely unknown from the arm classifier, so you know the *shape*
of each customer's demand curve but not its height. If you need absolute purchase
probabilities — to forecast revenue rather than to choose a price — fit an ordinary
`P(buy | x)` model on **all** customers and combine:

```
β_k(x)  =  p_k(x) · P(buy | x) / q_k
```

(`demand_curve_from_posterior`; verified to 4.4 × 10⁻¹⁶.)

---

## 5. Fact 2 — the best price is a point on the ROC surface

If you offer price `m_k` to customer `x`, your expected revenue is

```
R_k(x)  =  m_k · β_k(x)  =  m_k · c(x) · p_k(x) · 3      (equal q)
```

To choose the best price you maximise over `k` — and `c(x)` is a positive number that does
not depend on `k`, so it **cannot change which `k` wins**:

> ```
> argmax_k  m_k β_k(x)   =   argmax_k  (m_k / q_k) · p_k(x)
> ```

The right-hand side is exactly `roc3`'s weighted-argmax rule with weights `w ∝ m/q`.
**Choosing an operating point on the 3-class ROC surface *is* choosing a pricing policy.**
The unknown level cancels, so you do not need it to price — only to forecast.

Worked on the toy world. Using the (invisible) demand curves:

| type | `m·β` at 0.9 | at 1.0 | at 1.1 | best |
|---|---|---|---|---|
| L | 0.8100 | 0.8500 | **0.8800** | 1.1 |
| M | **0.5400** | 0.4500 | 0.3300 | 0.9 |
| S | **0.3600** | 0.2000 | 0.1100 | 0.9 |

Using only the posterior:

| type | `m·p` at 0.9 | at 1.0 | at 1.1 | best |
|---|---|---|---|---|
| L | 0.3176 | 0.3333 | **0.3451** | 1.1 |
| M | **0.4000** | 0.3333 | 0.2444 | 0.9 |
| S | **0.5143** | 0.2857 | 0.1571 | 0.9 |

Different numbers, identical decisions. (On the 600k run: 100.000000% agreement.)

Note the economics the toy encodes: loyal customers should be charged *more*, not less.
Discounting them is pure margin given away. That is what a personalisation model is for.

```python
from roc3.elasticity import revenue_optimal_policy
pol = revenue_optimal_policy(proba, (0.9, 1.0, 1.1))          # revenue
unit_cost = 0.6          # what the item costs you; NB nothing to do with c(x) in section 4
pol = revenue_optimal_policy(proba, (0.9, 1.0, 1.1),
                             unit_margin=(0.9 - unit_cost,
                                          1.0 - unit_cost,
                                          1.1 - unit_cost))       # margin, not revenue
```

---

## 6. Validation 1 — the group trick

### 6.1 Why groups work when individuals don't

Take any set of customers you can define **without looking at their arm or their outcome**
— say, "everyone the model predicts is highly elastic". Because the arm was assigned by a
coin flip, that set contains all three arms in known proportion, and the three sub-groups
are otherwise comparable. So:

```
purchase rate in arm k, within the group   =   the group's average β_k
```

which is a demand curve. Read the elasticity off it. **No model, no assumptions beyond the
randomisation you already ran.**

The model supplies the groups; the data supplies the answer. This is the **GATES** idea —
*Group Average Treatment Effects Sorted* — from Chernozhukov, Demirer, Duflo &
Fernández-Val, *Econometrica* 2025. "Sorted" because the groups are formed by sorting
customers on the model's own prediction. The name is used throughout as shorthand for
"bin by prediction, measure the realised effect per bin".

### 6.2 Worked by hand

Bin the 900 customers by predicted elasticity. Here the model is perfect, so the bins are
exactly the three types:

| bin | n | predicted `ε` | rate @0.9 | rate @1.0 | rate @1.1 | **realised `ε`** |
|---|---|---|---|---|---|---|
| 0 | 300 | 0.5869 | 0.9000 | 0.8500 | 0.8000 | **0.5869** |
| 1 | 300 | 3.4542 | 0.6000 | 0.4500 | 0.3000 | **3.4542** |
| 2 | 300 | 6.9083 | 0.4000 | 0.2000 | 0.1000 | **6.9083** |

Top bin, by hand: `−log(0.10 / 0.40) / 0.20067 = 6.9083`. That is the whole calculation.
The realised column was computed from purchase counts alone; the predicted column came
from the model; they match, so the model is telling the truth.

```python
from roc3.elasticity import gates_elasticity, blp_slope
g = gates_elasticity(arm, bought, eps, n_bins=5)   # arm/bought for ALL customers
blp_slope(g)                                        # -> slope, intercept, p-values
```

(`blp` is for **Best Linear Predictor** — the straight line that best predicts the realised
elasticity from the predicted one. Its slope is the calibration number of §6.4.)

### 6.3 A model with no signal

Attach the same predictions to the wrong customers and the structure collapses:

| bin | predicted | realised |
|---|---|---|
| 0 | 0.5869 | 1.7130 |
| 1 | 3.4542 | 3.0706 |
| 2 | 6.9083 | 2.0075 |

spread **+0.29** (against **+6.32** for the honest model), calibration slope **+0.02**.
The test has no trouble telling them apart.

### 6.4 The two numbers to read

**Spread** — top-bin minus bottom-bin realised elasticity. *Does the model separate
customers at all?* This is the ranking question.

**Calibration slope** — regress realised on predicted across bins, weighting each bin by
its precision.

| slope | meaning | what to do |
|---|---|---|
| ≈ 0 | no signal | the model knows nothing about elasticity |
| > 0 but ≠ 1 | ranks correctly, magnitudes wrong | usable for targeting; recalibrate before using the numbers |
| ≈ 1 | calibrated | the predicted elasticities mean what they say |

On the 600,000-customer run, with three real models:

where `p(no signal)` is the p-value of the test that the calibration slope is 0 — small
means the model demonstrably knows *something* about elasticity:

| model | corr(ε̂, ε_true) | spread | slope | p(no signal) |
|---|---|---|---|---|
| A: logistic, both features | +0.987 | +2.425 ± 0.079 | **+0.956 ± 0.017** | ~0 |
| B: gradient boosting | +0.953 | +2.296 ± 0.075 | +0.915 ± 0.016 | ~0 |
| C: logistic, irrelevant feature only | −0.002 | +0.037 ± 0.051 | +0.439 ± 0.414 | **0.29** |

and the **oracle** — scoring by the true elasticity — comes out at slope **0.997 ± 0.018**
with p(slope = 1) = 0.88. Exactly calibrated, as it must be. That is the method
validating itself, which is the reassurance you want before trusting it on a real model.

### 6.5 Three ways to get this wrong

* **In-sample predictions.** Bins formed with a model that saw those customers are
  self-fulfilling: the model can manufacture a spread from noise. Use out-of-fold
  predictions, and prefer several splits (this is exactly why Chernozhukov et al. use
  repeated data splitting; see also Imai's 2025 comment for how much the split matters).
* **Scoring only buyers.** The model is *fitted* on buyers, but it is a function of `x`,
  so evaluate it on **everyone**. The bin purchase rates need the non-buyers in the
  denominator.
* **Bins too thin.** The per-bin standard error goes as `1/√(n_k · D_k)`. Five bins of
  100,000 is comfortable; five bins of 500 will show you noise and call it heterogeneity.

---

## 7. Validation 2 — what the policy is worth

Ranking well is not the goal. Making money is.

### 7.1 Inverse-probability weighting, from scratch

You want to know what a policy `d` *would have earned*, but you only ran the randomised
experiment. The trick: for the customers who happened to be shown the arm the policy would
have chosen, you observe exactly what the policy would have got. Those customers are a
random `q_k` fraction of the relevant group, so scale them up by `1/q_k`:

```
V(d)  =  mean over ALL customers of    1{K_i = d(x_i)} / q_{K_i} · m_{K_i} · Y_i
```

With `q = 1/3`, every matching customer counts triple, and non-matching customers
contribute zero. It is unbiased because the arm was genuinely random — no modelling
assumption anywhere.

### 7.2 Worked by hand

"Always offer 0.9": 300 customers were shown that arm, of whom 90 + 60 + 40 = **190**
bought.

```
V  =  3 × 0.9 × 190 / 900  =  513 / 900  =  0.57000
```

and the true value, computable only because this is a simulation, is
`0.9 × (0.90 + 0.60 + 0.40)/3 = 0.57000`. Exact.

| policy | buyers among the matched | IPW value | truth |
|---|---|---|---|
| always 0.9 | 190 / 300 | **0.57000** | 0.57000 |
| always 1.0 | 150 / 300 | **0.50000** | 0.50000 |
| always 1.1 | 120 / 300 | **0.44000** | 0.44000 |
| personalised (L→1.1, M→0.9, S→0.9) | 180 / 300 | **0.59333** | 0.59333 |

```python
from roc3.elasticity import policy_value
policy_value(arm, bought, pol)                    # IPW: unbiased, assumption-free
policy_value(arm, bought, pol, beta_hat=beta)     # AIPW: same guarantee, far less noise
```

The augmented (AIPW) version adds a model-based term and subtracts its error on the
matched customers. Because the propensities `q` are *known exactly* here, AIPW stays
unbiased no matter how poor the outcome model is — it only reduces variance. On the 600k
run, IPW recovers the oracle policy's value it cannot observe: `0.53694 ± 0.00145` against
a true `0.53529`.

### 7.3 The trap: benchmark against the best *flat* price

Personalisation looks worth **+18.67%** against flat-mid on the toy — and **+4.09%**
against the best flat price. The first number is mostly not personalisation at all.

To see why, consider a model with **no usable features**. It learns the marginal
`P(arm | bought) = (0.4130, 0.3261, 0.2609)`, the same for every customer. Then
`m · p = (0.3717, 0.3261, 0.2870)`, so it offers **0.9 to everyone**. Its "personalised"
policy is a constant rule worth `0.57000` — *exactly* flat-cheap.

Against flat-mid that is **+14.00%**. Against the best flat price it is **+0.00%**. The
apparent skill was entirely the discovery that the cheap price beats the middle one — which
you would have found by reading the experiment's **topline**, the three overall per-arm
purchase rates, without fitting anything at all.

The same thing happens at scale. On the 600k run:

| model | value | vs flat mid | **vs best flat** |
|---|---|---|---|
| C (no signal) | 0.52176 | +4.49% | **+0.00%** |
| B | 0.53365 | +6.87% | **+2.28%** |
| A | 0.53667 | +7.48% | **+2.86%** |
| oracle | 0.53694 | +7.53% | **+2.91%** |

Read against the best flat price, A captures 98% of the available personalisation value
and B captures 78%, and the useless model is correctly worth nothing. **Always benchmark
against `max_k V(flat_k)`.**

---

## 8. Validation 3 — the doubly-robust pseudo-outcome, and two traps

### 8.1 What it is

The standard device in the causal-inference literature is a per-customer quantity whose
*conditional mean* is the effect you want, even though it is wildly noisy for any one
person. For the uplift `τ(x) = β₁(x) − β₃(x)`:

```
ψ_i  =  β̂₁(x_i) − β̂₃(x_i)  +  1{K=1}/q₁ · (Y_i − β̂₁)  −  1{K=3}/q₃ · (Y_i − β̂₃)
```

Read it as: start from a model's guess, then correct it using the observed outcome,
scaled up by `1/q` because only a fraction of customers landed in the relevant arm. The
correction has mean zero when the model is right, and the `1/q` weighting fixes it when
the model is wrong — hence "doubly robust". `E[ψ | x] = τ(x)` exactly.

*Verified:* mean `ψ = +0.1513` against a true mean uplift of `+0.1548`.

It is a genuinely useful tool, and it is why DR-based criteria dominate **CATE**
model-selection benchmarks (Mahajan et al. 2023). *CATE* is the **Conditional Average
Treatment Effect**, `E[effect | x]` — the general name for "the effect for customers who
look like this". Elasticity and uplift are both CATEs, of different effects.
But in this setting the pseudo-outcome has two traps.

### 8.2 Trap 1 — the target is a *difference*; elasticity is a *ratio*; neither is the decision

`ψ` targets `β₁ − β₃`, an **absolute** uplift. Elasticity is a **log ratio**. Those pick
out different customers:

* a **ratio** is largest where `β` is small — you can halve a 4% purchase rate easily;
* a **difference** is largest where `β ≈ ½` and the demand curve is steepest.

Measured on the 600k run:

```
Spearman(true elasticity, true uplift)          +0.024      <- essentially unrelated!
Spearman(true elasticity, true revenue gain)    +0.780
Spearman(true uplift,     true revenue gain)    +0.444
```

Elasticity and uplift rank customers **almost independently**. So:

> **The elasticity model is not automatically the right targeting score.** Neither is
> uplift. For a pricing decision the score to rank on is the predicted **revenue gain** —
> how much more you expect to earn from `x` by giving them their best price instead of the
> price everyone would otherwise get:
> `max_k m_k β̂_k(x) − m_base β̂_base(x)`, where `base` is whichever arm is your default
> (the middle one here).

`elasticity_report` computes that and ranks on it for the **targeting curve** — a plot of
policy value against the fraction of customers you personalise, taking the most promising
first (the right-hand panel of the figure in §9). A model with real signal earns most of
its gain in the first slice; one with none traces a straight line. Decide what you are
ranking *for* before you choose a metric.

### 8.3 Trap 2 — never rank-correlate against ψ

`ψ` is a near-three-point variable: its value is dominated by which arm the customer
landed in and whether they bought. Those atoms depend on `x`, so **the ranks of ψ encode
`β(x)`**, not the uplift. The result is a rank correlation that swings with the nuisance
model even though AIPW is unbiased throughout.

Two correlations are compared below. **Pearson** is the ordinary linear correlation, which
uses the values. **Spearman** is Pearson applied to the *ranks* — it asks only about
ordering, which is normally a virtue with noisy data and is exactly the wrong choice here.

| `β̂` used inside ψ | Pearson(ε_true, ψ) | Spearman(ε_true, ψ) |
|---|---|---|
| the truth | +0.0009 | +0.005 |
| zero (pure IPW form) | −0.0004 | −0.008 |
| from a constant `P(buy\|x)` | +0.0004 | **−0.105** |

The decisive case is the **oracle**: scoring by the *true* elasticity, the main run gives
Pearson `+0.0026` and Spearman `−0.0738`. Since the oracle is the truth by construction,
that −0.074 is pure artefact — and every model in the run shows the same artefact at the
same size (A −0.074, B −0.077), which would have ranked them identically and told you
nothing.

**Use regression against ψ. Never rank correlation.**

And keep the scale in mind: `sd(ψ) = 1.22` against a true uplift spread of `0.043` —
**28× more noise than signal**. Nothing per-customer will ever be visible. Only averages.
Which is exactly why §6 bins.

---

## 9. Comparing two models

Use a **paired** bootstrap: score both models on the *same* resampled customers, so the
shared sampling noise cancels and the interval on the difference is far tighter than two
separate intervals would suggest.

```python
from roc3.elasticity import compare_elasticity_models
compare_elasticity_models(arm, bought, score_a, score_b, policy_a, policy_b, n_boot=400)
```

**A vs C (real signal vs none)** — everything separates, as it should:

| criterion | A | C | difference | 95% CI | |
|---|---|---|---|---|---|
| policy value | +0.53809 | +0.52593 | +0.01216 | (+0.0078, +0.0162) | significant |
| GATES spread | +2.277 | −0.039 | +2.316 | (+2.015, +2.599) | significant |
| calibration slope | +0.952 | −0.520 | +1.473 | (+0.153, +2.748) | significant |

**A vs B (two genuinely good models)** — and here it gets interesting:

| criterion | A | B | difference | 95% CI | |
|---|---|---|---|---|---|
| policy value | +0.53809 | +0.53558 | +0.00251 | (−0.0011, +0.0061) | **not** significant |
| GATES spread | +2.277 | +2.333 | −0.056 | (−0.255, +0.085) | **not** significant |
| calibration slope | +0.952 | +0.921 | +0.031 | (+0.006, +0.054) | **significant** |

A and B are indistinguishable in what they are *worth* and in how well they *rank*, and
differ only in **calibration**. If you need a ranking — whom to discount — they are
interchangeable. If you need the elasticity number itself — to set a price, or feed an
optimiser — prefer A.

Three criteria, three different answers, and which one is right depends on your use. That
is why the suite reports all of them, and why the literature is clear that no single
criterion for CATE model selection is best in general (Curth & van der Schaar 2023), with
losses built from a particular learner's pseudo-outcome biased toward that learner
(congeniality bias, Mahajan et al. 2023).

![elasticity diagnostics](../figures/10_elasticity_diagnostics.png)

---

## 10. The recipe

```python
from roc3.elasticity import elasticity_report, format_elasticity_report

rep = elasticity_report(
    arm, bought, proba,                  # ALL customers, buyers and not
    price_multipliers=(0.9, 1.0, 1.1),   # cheapest first
    arm_probs=None,                      # pass q if allocation was not equal
    n_bins=5,
    p_buy=p_buy_model,                   # optional: enables AIPW and absolute revenue
)
print(format_elasticity_report(rep))
```

In descending order of how much weight to put on them:

1. **Policy value against the best flat price** — decision-relevant, unbiased by design,
   and the number a business will act on.
2. **The GATES calibration curve** — assumption-light, interpretable, and it separates
   "ranks correctly" from "magnitudes correct".
3. **Targeting-curve area** — how concentrated the gains are. Matters when personalising
   has an operational cost.
4. **The VUS from [`PRICETEST.md`](PRICETEST.md)** — a pure ranking summary, with the
   elasticity-implied ceiling for context. It is measuring the same index, since the arm
   classifier *is* the elasticity model (§4).
5. **DR-based losses** — standard and useful, but read §8 first.

---

## 11. Glossary

| term | meaning | where |
|---|---|---|
| **arc elasticity** | `−[log β(p_hi) − log β(p_lo)] / [log p_hi − log p_lo]` | §1.1 |
| **arm** | one price cell of the experiment | §1.3 |
| `β_k(x)` | customer `x`'s purchase probability at arm `k`; never observed | §1.3 |
| `p_k(x)` | `P(arm = k \| x, bought)` — the arm classifier's output | §1.3 |
| `q_k` | the randomisation probability (propensity) of arm `k`; known by design | §1.3 |
| `m_k` | the price multiplier of arm `k` | §1.3 |
| `D_k`, `n_k` | purchase rate in arm `k`, and how many customers were shown it | §1.3 |
| `c(x)` | the per-customer normalising constant that cancels out of every ratio | §4.1 |
| `w` | the weight vector of the rule `argmax_k w_k p_k(x)` | §1.3, §5 |
| **hat** (`β̂`) | "estimated from data", as opposed to the true value | §1.4 |
| **indicator** `1{A}` | 1 if `A` holds, 0 otherwise | §1.4 |
| **argmax** | the *index* that maximises, not the maximum | §1.4 |
| **∝** | proportional to, up to a factor free of the index being varied | §1.4 |
| **identified** | pinned down exactly by the observable distribution, given infinite data | §1.5 |
| **oracle** | a hypothetical model handed the true answer; used to grade the method | §1.5 |
| **out-of-fold** | predicted by a model fitted without that customer | §1.5 |
| **nuisance model** | a model fitted only because an estimator needs it as input | §1.5 |
| **precision** | the reciprocal of the variance | §1.5 |
| **demand curve** | the vector `(β₁(x), β₂(x), β₃(x))` for one customer | §3 |
| **GATES** | *Group Average Treatment Effects Sorted*: bin by predicted effect, measure the realised effect per bin | §6.1 |
| **BLP** | *Best Linear Predictor*: the line fitted through the bins; its slope is the calibration | §6.2 |
| **calibration slope** | slope of realised on predicted across bins; 1 = calibrated | §6.4 |
| **spread** | top-bin minus bottom-bin realised elasticity | §6.4 |
| **policy** `d` | a rule assigning a price to each customer | §1.3, §5 |
| **policy value** `V(d)` | average revenue per customer under that rule | §7 |
| **IPW** | inverse-probability weighting — reweight matched customers by `1/q` | §7.1 |
| **AIPW / doubly robust** | IPW plus a model-based term; same guarantee, less noise | §7.2 |
| **flat policy** | a rule that offers the same price to everyone | §7.3 |
| **CATE** | *Conditional Average Treatment Effect*, `E[effect \| x]` | §8.1 |
| **pseudo-outcome** `ψ` | a noisy per-customer variable whose conditional mean is the effect | §8.1 |
| **uplift** `τ(x)` | `β_cheap − β_expensive`, an absolute difference (not elasticity) | §8.2 |
| **revenue gain** | `max_k m_k β̂_k(x) − m_base β̂_base(x)`; the right score for pricing | §8.2 |
| **Pearson / Spearman** | linear correlation / the same on ranks | §8.3 |
| **targeting curve** | policy value as a function of what fraction you personalise | §8.2, §10 |
| **congeniality bias** | a loss built from one learner's pseudo-outcome favours that learner | §9 |
| **paired bootstrap** | resample customers once and score both models on the same resample | §9 |

---

## 12. FAQ

**"Can I ever validate a single customer's predicted elasticity?"** No. Not with this
data, not with more of it, not with a better model. Every method here is group-level. A
model can pass every check and still be wrong about any individual.

**"Then how can the calibration slope be meaningful?"** Because a slope of 1 across bins
says the model's predictions are *right on average within each group it defines* — which
is exactly the property you need if you are going to act on them for groups. It says
nothing about any one person, and does not need to.

**"My slope is 0.5. Is the model broken?"** No — it ranks correctly but its magnitudes are
compressed by half. Fine for targeting; recalibrate (regress realised on predicted and
apply the inverse) before feeding the numbers to a pricing optimiser.

**"My spread is large but the policy value is not."** Common, and informative. The model
separates customers by elasticity, but elasticity is not the same as revenue gain (§8.2).
Rank on predicted revenue gain instead.

**"Why not just fit a model of `β_k(x)` directly and read the elasticity off it?"** You
can, and it is a perfectly good alternative: fit one model of `P(buy)` on the pooled data
with the arm included as a feature, then read off its prediction at each of the three
prices. (In the causal-inference literature that is called an **S-learner** — "S" for
*single*, one model covering all treatment arms, as against a **T-learner** that fits a
separate model per arm.) The point of §4 is only that if you already have the arm
classifier, you already have the elasticity, for free and exactly. Either way the *validation* in §6–§9 is the same, because the validation
does not care how the prediction was produced.

**"Do I need the `P(buy | x)` model?"** Only for absolute revenue forecasts and for AIPW.
Not for choosing prices (§5), not for the GATES check (§6), not for IPW policy value
(§7.1).

**"The randomisation was not exactly equal."** Pass `arm_probs`. Everything continues to
work; the `log(q₃/q₁)` term in §4.1 stops being zero, and IPW reweights by the true `q`.

**"The randomisation was broken."** Then stop. IPW, GATES and the whole framework rest on
it. No amount of modelling recovers what a broken assignment destroys.

**"Can I extrapolate to a −40% discount?"** No. Everything here is local to the ±10% range
you tested, and no diagnostic in this suite will warn you when you leave it.

---

## Where to go next

* [`ELASTICITY.md`](ELASTICITY.md) — the compact research note with the full results.
* [`PRICETEST.md`](PRICETEST.md) — where the ordering constraint comes from, and why the
  VUS ceiling is the measured demand response.
* [`TUTORIAL.md`](TUTORIAL.md) — the 3-class ROC surface and VUS from scratch.
* `experiments/09_elasticity_tutorial_numbers.py` — regenerates every toy number here.
* `experiments/08_elasticity.py` — the 600,000-customer run.
