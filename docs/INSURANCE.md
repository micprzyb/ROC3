# Mandatory motor insurance: price tests where everyone buys

A different problem from the retail panel, and different in ways that change the models
rather than just the data. Notation extends [`MODELS.md §0`](MODELS.md#0-notation).

---

## 1. What makes this problem its own thing

| | retail panel | motor insurance |
|---|---|---|
| products | 3,218 SKUs | **one** — a policy |
| the decision | buy / don't buy / buy later | **buy from us, or from a competitor** |
| can the customer wait? | yes | **no** — the old policy expires on a date |
| outcome | units sold (a count) | **converted** (binary) |
| price variation | the seller repriced, observationally | a **randomised** price test |
| arm allocation | — | **0.1 / 0.8 / 0.1** at −10% / 0% / +10% |
| the objective | revenue | **profit**: conversion × (premium − expected claims) |

Four consequences follow, and each one invalidates something that was true in the retail
setting.

### 1.1 Category demand is fixed; only share is elastic

Everyone buys. What the price moves is **which insurer**, not whether. So this is a
share/conversion problem, and the elasticity being estimated is an elasticity **of
conversion**, not of category demand. That is the quantity that matters commercially, but it
must not be called "the price elasticity of demand for motor insurance" — that number is
close to zero and is a different thing.

### 1.2 Demand is bounded, so elasticity is not a constant even with one coefficient

Let conversion follow a logit in log price:

$$s(x, p) \;=\; \sigma\big(a(x) \;-\; \beta(x)\,\log p\big), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

Then

$$\varepsilon(x, p) \;=\; -\frac{\partial \log s}{\partial \log p} \;=\; \beta(x)\,\bigl(1 - s(x,p)\bigr).$$

**The elasticity is mechanically tied to the conversion rate.** A customer we already
convert at 90% has elasticity $0.1\beta$; one we convert at 5% has $0.95\beta$. So:

* elasticity is **heterogeneous even if $\beta$ is completely homogeneous** — all it takes is
  variation in the baseline $a(x)$, which is exactly what a technical pricing model creates;
* elasticity **varies along the price ladder for the same customer**, so a single number per
  customer is already an approximation, and the −10% and +10% arms have genuinely different
  elasticities;
* it is **not** the multiplicative world of the retail panel. There, a homogeneous elasticity
  made the arm posterior constant in $x$ and VUS collapsed to chance
  ([`PLAN_PRICETEST_MODELS.md`](PLAN_PRICETEST_MODELS.md) §4.2, Proposition 2). Here it does
  not: this is Proposition 3, and **VUS above chance is not evidence of elasticity
  heterogeneity** — it can come entirely from variation in the baseline conversion rate.

### 1.3 Unequal allocation breaks the posterior ordering

This is the part that changes the constraint, and it is worth being precise about because it
modifies the premise the constraint was originally stated under.

With arms assigned at rates $\pi = (0.1,\,0.8,\,0.1)$, the arm posterior **among converters**
is

$$b_k(x) \;=\; \frac{\pi_k\, s_k(x)}{\sum_j \pi_j\, s_j(x)}.$$

Monotone demand still gives $s_1 \ge s_2 \ge s_3$ (cheaper converts better). But the middle
arm carries **eight times** the traffic, so

$$b_2 \;\gg\; b_1 \qquad \text{even though} \qquad s_1 > s_2 .$$

So the stated constraint $0 < p_1 < p_2 < p_3 < 1$ is a consequence of monotone demand **only
under equal allocation**. At 0.1/0.8/0.1 the posterior is not ordered at all and enforcing
that ordering would be enforcing something false.

**The constraint that is actually true, and that generalises to any allocation and any number
of arms:**

$$\frac{b_1(x)}{\pi_1} \;\ge\; \frac{b_2(x)}{\pi_2} \;\ge\; \dots \;\ge\; \frac{b_K(x)}{\pi_K}$$

Since $\pi$ is **known by design** — this is the thing randomisation buys — this is still a
hard, checkable constraint on the model's output, exactly as strong as the original. It
reduces to $b_1 \ge b_2 \ge b_3$ when $\pi$ is uniform. Equivalently, and most usefully:
**every arc elasticity is non-negative.**

### 1.4 The objective is profit, and it needs the elasticity in a specific way

With expected claims cost $c(x)$, profit per quote is $s(x,p)\,(p - c(x))$. Setting the
derivative to zero gives the Lerner condition

$$\frac{p^\* - c}{p^\*} \;=\; \frac{1}{\varepsilon(x, p^\*)} \qquad\Longleftrightarrow\qquad p^\* \;=\; \frac{c(x)}{1 - 1/\varepsilon(x, p^\*)}$$

which requires $\varepsilon > 1$ for a finite optimum. Two things to note:

* it is an **implicit** equation, because $\varepsilon$ depends on $p$ through §1.2 — solved
  by a few fixed-point iterations, not in closed form;
* the elasticity enters as $1/\varepsilon$, so accuracy matters most where $\varepsilon$ is
  near 1, and barely at all where it is large. **That is a different loss function from
  squared error in $\varepsilon$**, and it is the one to report.

### 1.5 Only 20% of the traffic carries price information

At 0.1/0.8/0.1 the identifying variation lives in the side arms. A test on 1,000,000 quotes
gives 100,000 at −10% and 100,000 at +10%. That is a lot in absolute terms and very little
per covariate cell, and it means:

* per-arm (T-learner) models are starved in exactly the arms that matter;
* pooling across arms with the price as a feature (S-learner, GLM) uses the 80% too, and is
  the reason the GLM does well here despite being the crudest model;
* the effective sample size for *heterogeneity* is smaller still, so shrinkage and constraints
  earn their keep.

---

## 2. Measured on the simulator (200,000 quotes)

`elasticity_lab/insurance.py`. Every structural claim in §1 is confirmed:

**§1.2 — elasticity is mechanically tied to conversion.** By quintile of true conversion:

| conversion | 7.5% | 19% | 33% | 51% | 76% |
|---|---|---|---|---|---|
| mean β | 3.83 | 3.47 | 3.21 | 2.96 | 2.55 |
| mean ε | **3.54** | 2.80 | 2.15 | 1.46 | **0.63** |

β moves by 1.5×; the elasticity moves by **5.6×**. Most of the heterogeneity is induced by
the conversion rate, not by taste. Realised arc elasticity across the arms: 1.40; mean true
point elasticity 2.12 (sd 1.23); 78% of quotes have ε > 1, so 22% have no interior profit
optimum at all.

**§1.3 — the posterior ordering fails, completely.** Demand $s_k$ is decreasing for
**100.0%** of quotes. The posterior $b_k$ is decreasing for **0.0%** of them: the mean
posterior is $(0.124,\ 0.795,\ 0.081)$, dominated by the 80% control allocation. Enforcing
$p_1 < p_2 < p_3$ here would be enforcing something false in every single row;
$b_k/\pi_k$ decreasing holds in all of them.

**§1.4 — profit.** Mean profit per quote is 75.4 at control and 123.8 at the oracle optimum
(+64%), with a median optimal price 1.22× the technical premium.

### 2.1 A trap: the elasticity must be evaluated at a common reference price

Because $\varepsilon = \beta(1-s)$ depends on the price, the *same customer* has a
different elasticity in each arm. Binning on the elasticity at the price they happened to be
quoted therefore selects a **different population into each arm within a bin** — for a given
ε, a customer in the dear arm has a lower β than one in the cheap arm — and the realised arc
elasticity comes out badly biased:

| | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 |
|---|---|---|---|---|---|
| predicted | 0.52 | 1.30 | 2.02 | 2.78 | 3.95 |
| realised, binned on the **quoted-price** ε | **0.31** | 0.53 | 0.81 | 0.92 | 2.37 |
| realised, binned on the **control-price** ε | **0.53** | 1.14 | 1.96 | 2.91 | 4.11 |

The bias is not subtle and it is not a simulator artefact: any model whose `elasticity()`
is evaluated at the customer's own quoted price has it. `true_elasticity_control` exists for
this reason, and every grouped diagnostic must use a common reference price.

## 3. Restoring the familiar constraint — two routes, and how they relate

Under unequal allocation the admissible set is still a **triangle**, just not the usual one.
`{b : b_1/\pi_1 \ge b_2/\pi_2 \ge b_3/\pi_3}` is cut from the simplex by two homogeneous
linear inequalities, so its vertices are $e_1$, $(\pi_1,\pi_2,0)/(\pi_1+\pi_2)$ and $\pi$
itself — measured at 0.1/0.8/0.1:

| standard chamber $\{v_1 \ge v_2 \ge v_3\}$ | admissible region |
|---|---|
| (1, 0, 0) | (1, 0, 0) |
| (0.5, 0.5, 0) | (0.111, 0.889, 0) |
| (⅓, ⅓, ⅓) | (0.1, 0.8, 0.1) |
| area 0.1667 | area **0.0889** |

The map $b \mapsto (b_k/\pi_k)/\sum_j(b_j/\pi_j)$ carries the second onto the first, exactly
(`max err 0.00e+00`). It is the **gauge map** of [`CONSTRAINED.md`](CONSTRAINED.md) with
$c_k = 1/\pi_k$, so by the gauge theorem **VUS is bit-for-bit invariant under it** — verified,
`max diff 0.00e+00`.

So there are two routes to the ordinary $p_1 \ge p_2 \ge p_3$ machinery, and they are not the
same thing:

| | what it changes | mean ε | effective sample |
|---|---|---|---|
| `balance="weights"` — weight converters by $1/\pi_k$ | the training **objective** | 1.894 | **42.5%** |
| `balance="oversample"` — resample instead | the training objective | 1.846 | 100%* |
| `balance="none"` + `to_balanced_gauge` | only the **coordinates** | 1.842 | 100% |

Same VUS for all three, necessarily. **Different fitted models**, because a weighted
log-loss is a different loss. Weighting spends 57% of the effective sample to buy
coordinates the transformation gives for free — so on this book the transformation is the
better default, though the difference in accuracy is small.

*(oversampling replaces weight variance with resampling variance rather than removing it.)*

Worth noting: even in balanced coordinates the *unconstrained* fit violates the ordering for
about 43% of quotes, so the constraint is doing real work rather than rubber-stamping.

## 4. The model set, graded

`experiments/18_insurance_models.py`, 300,000 quotes, 70/30 split, graded against
`true_elasticity_control` (mean 2.115, sd 1.233).

| model | mean ε | bias | RMSE | corr | % negative |
|---|---|---|---|---|---|
| `conversion_glm` | 1.720 | −0.395 | **0.856** | **0.837** | 0 |
| `glm_then_gbm` (price hidden) | 1.720 | −0.395 | 0.858 | 0.835 | 0 |
| `glm_then_gbm` (price shown) | 1.729 | −0.386 | 0.869 | 0.812 | 0 |
| `conversion_gbm` | 1.790 | −0.325 | 1.164 | 0.502 | 0.1% |
| `arm_posterior` (none, isotonic) | 2.061 | **−0.055** | 1.473 | 0.439 | 0 |
| `tlearner` | 1.895 | −0.221 | 1.524 | 0.480 | **8.6%** |
| `arm_posterior` (weights, isotonic) | 2.170 | +0.055 | 1.584 | 0.436 | 0 |
| `arm_posterior` (weights, free) | 1.894 | −0.221 | 1.773 | 0.443 | **13.7%** |

Five things worth reading off it.

**The GLM wins, and partly for a rigged reason.** The simulator's demand *is* a logit, so
`ConversionGLM` is correctly specified and $\varepsilon = \beta(1-s)$ is exact rather than
approximate. That advantage would not survive a misspecified world. What *would* survive is
the structural one from §1.5: it pools all three arms, so it uses the 80% control traffic
that the arm-based methods cannot.

**Two-stage with the price hidden ties the GLM to three decimals** — as it must, since stage
two cannot touch the price derivative. That is a correctness check, not a finding. **With the
price shown it is slightly worse** (0.869 vs 0.856): the GBM re-imports the flattening
problem onto the correction term, which is the predicted failure of the naive residual
approach.

**The arm-posterior model has the best level and the worst spread.** Bias −0.055 against the
GLM's −0.395 — it is nearly unbiased, because it assumes nothing about the shape of the
demand curve — but its RMSE is 1.47 and its GATES is badly over-dispersed (predicted 0.24 →
4.76 across bins where the realised range is only 0.69 → 1.81). It is the unbiased-but-noisy
member of the set, and the right way to use it is as a check on the GLM's level rather than
as a per-customer scorer.

**The constraint earns its place.** Unconstrained, the arm-posterior model reports
upward-sloping demand for **13.7%** of quotes; with the isotonic constraint, **0%**, by
construction. The T-learner, which has no constraint available to it, sits at 8.6%.

**Only the GLM's GATES is monotone.** Realised elasticity by predicted quintile: 0.50, 1.29,
1.95, 2.68, 3.07 against predicted 0.66, 1.35, 1.83, 2.22, 2.54 — the ranking is right and
the spread is if anything *understated*. Neither the GBM nor the arm-posterior model is
monotone.

## 5. What it is all worth — the sobering part

Every policy evaluated by inverse-propensity weighting on held-out quotes, which is unbiased
because the allocation is known.

| policy | profit/quote | se | vs best flat | in se |
|---|---|---|---|---|
| Lerner via `conversion_glm` | 86.36 | 2.88 | +0.16 | **+0.06** |
| **all dear (+10% for everyone)** | **86.20** | 2.98 | — | — |
| Lerner via `glm_then_gbm` (price shown) | 86.12 | 2.87 | −0.08 | −0.03 |
| Lerner via `conversion_gbm` | 85.47 | 2.84 | −0.73 | −0.26 |
| Lerner via `tlearner` | 82.97 | 2.32 | −3.22 | −1.39 |
| Lerner via `arm_posterior` | 78.9–79.9 | ~2.1 | −6.3 to −7.2 | **−3.0 to −3.5** |
| all control | 75.41 | 0.82 | −10.78 | −13.1 |
| all cheap (−10%) | 55.96 | 1.77 | −30.24 | −17.1 |

**No targeted policy beats charging everyone +10%.** The best is 0.06 standard errors above
it — indistinguishable — and three of them are *significantly worse*. Meanwhile the flat
move itself is worth **+14.3%** on profit per quote (75.41 → 86.20) at 13 standard errors.

The honest conclusion for this book: **the money is in the level, not the targeting.** That
is the same `lift_over_uniform` lesson as the retail work ([`ELASTICITY_MODELS.md`](ELASTICITY_MODELS.md)
§5.4), arriving independently in a completely different setting, and it is worth taking
seriously before anyone builds a personalisation pipeline.

Two caveats in the other direction. With 10% of traffic per side arm, a targeted policy is
evaluated on ~9,600 effective quotes, so **anything smaller than about 4.6 in this table is
invisible** — the experiment cannot detect a personalisation gain below roughly 5%. And the
Lerner rule was discretised onto the three prices the test actually ran; a real deployment
would price on a continuum, where the GLM's ranking (corr 0.84) has more room to pay.

## 6. Widening the arms, and grading on profit

`experiments/19_arm_width_and_profit.py`. 300,000 quotes per design.

### 6.1 Squared error is the wrong loss, in principle

The optimal price moves with the elasticity at rate $-c/(\varepsilon-1)^2$ and profit is
locally quadratic at its optimum, so the regret from an error scales as
$(\Delta\varepsilon)^2/(\varepsilon-1)^4$. Normalising at $\varepsilon = 3$:

| ε | 1.05 | 1.10 | 1.20 | 1.50 | 2.00 | 3.00 | 6.00 |
|---|---|---|---|---|---|---|---|
| relative cost of a unit error | **2,560,000** | 160,000 | 4,096 | 256 | 16 | 1 | 0.04 |

Squared error treats every column alike; the decision does not. And 22% of this book sits
*below* $\varepsilon = 1$, where the Lerner price is undefined — those rows need a cap, not
a better estimate.

### 6.2 Arm width buys a lot — for the nonparametric models only

RMSE against the true elasticity at control:

| model | ±5% | ±10% | ±20% | ±30% | ±50% |
|---|---|---|---|---|---|
| `conversion_glm` | 0.899 | 0.856 | 0.748 | 0.741 | 0.751 |
| `glm_then_gbm` | 0.901 | 0.858 | 0.751 | 0.745 | 0.755 |
| `conversion_gbm` | 1.228 | 1.164 | 1.073 | 1.042 | 1.023 |
| `arm_posterior` | 2.518 | 1.473 | 0.912 | **0.752** | 0.828 |
| `tlearner` | 2.903 | 1.524 | 0.867 | **0.714** | 0.825 |

**The GLM's rank correlation is flat at 0.835 across the entire range** (0.838 at ±5%, 0.833
at ±50%). Being correctly specified and pooling all three arms, it has already extracted
everything at ±5% and wider arms tell it nothing new. The nonparametric models go from
useless to best: `tlearner` climbs 0.236 → 0.890 and **overtakes the GLM at about ±30%**.

**±50% is worse than ±30%** for both arm-based methods — at that width the dear arm converts
so rarely that the choice-based sample thins out. There is an interior optimum, and on this
book it is near ±30%.

### 6.3 …but personalisation still is not detectable, and the flat gain grows faster

Lift over the best flat policy, in standard errors:

| model | ±5% | ±10% | ±20% | ±30% | ±50% |
|---|---|---|---|---|---|
| `conversion_glm` | −0.23 | +0.06 | +0.74 | **+1.17** | +1.14 |
| `tlearner` | −3.30 | −1.39 | −0.47 | +1.15 | +1.28 |

Widening moves it from invisible to *suggestive*, but even at ±30% no model clears two
standard errors. Meanwhile:

| | ±5% | ±10% | ±20% | ±30% | ±50% |
|---|---|---|---|---|---|
| best **flat** profit/quote | 81.64 | 86.20 | 91.68 | 94.18 | **98.20** |
| test cost/quote | 0.10 | 0.55 | 2.16 | 4.89 | 14.18 |

Against control at 75.41, a wider test is worth far more for **finding the right level** —
+30% by ±50% — than for enabling targeting. The same conclusion as §5, now with the
experiment design as the lever.

### 6.4 Fix the allocation before widening the arms

The cheapest finding here. At ±10%:

| allocation | eps RMSE | regret | lift (se) | effective quotes | test cost |
|---|---|---|---|---|---|
| 0.1 / 0.8 / 0.1 | 0.856 | 6.73 | +0.06 | 9,626 | 0.56 |
| **0.2 / 0.6 / 0.2** | **0.730** | **3.54** | −0.07 | 19,118 | 1.25 |
| balanced | 0.758 | 4.28 | +0.50 | 30,131 | 2.02 |

**0.2/0.6/0.2 at ±10% beats 0.1/0.8/0.1 at ±30% on regret (3.54 vs 3.83) at a quarter of the
cost (1.25 vs 4.89).** Rebalancing is a far cheaper way to buy the same information than
widening, because the loss from a mispriced quote grows with the square of the mispricing
while the information grows roughly linearly.

### 6.5 Does the profit loss change the answer?

Mostly not, and this is a negative result about my own proposal. Comparing the model
*ranking* by squared error against the ranking by regret:

| arm width | best by ε-RMSE | best by regret | ρ | agree |
|---|---|---|---|---|
| ±5% / ±10% / ±20% | `conversion_glm` | `conversion_glm` | 0.90 | ✓ |
| ±30% | `tlearner` | `tlearner` | 1.00 | ✓ |
| ±50% | `conversion_glm` | `glm_then_gbm` | 0.60 | ✗ |

The two losses pick the same model at four widths out of five. So the
$(\varepsilon-1)^{-4}$ weighting is **not** worth the complexity for *model selection* here.
Where it should still matter is *within* a model — tuning, and deciding which customers to
spend capacity on — because that is where the 2,560,000:1 ratio in §6.1 actually bites. That
has not been tested and should not be claimed.

## 7. Status

The simulator, the model set, the decision layer and the diagnostics are in
`elasticity_lab/insurance.py`. This document records
the structure of the problem, which is what determines the models; results follow once the
benchmark has run.

The design deliberately reuses what already exists: the arm-posterior construction and its
constraint machinery come from [`pricelevels.py`](../elasticity_lab/pricelevels.py) with
$\pi$ **known** instead of estimated, and the ceiling/leakage audit comes from
`roc3.pricetest`, which was written for exactly this experiment.
