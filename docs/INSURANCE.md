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

## 3. Status

The simulator, the decision layer and the diagnostics are in `elasticity_lab/insurance.py`. This document records
the structure of the problem, which is what determines the models; results follow once the
benchmark has run.

The design deliberately reuses what already exists: the arm-posterior construction and its
constraint machinery come from [`pricelevels.py`](../elasticity_lab/pricelevels.py) with
$\pi$ **known** instead of estimated, and the ceiling/leakage audit comes from
`roc3.pricetest`, which was written for exactly this experiment.
