# The models, in detail

A complete reference for every estimator, objective and metric in `elasticity_lab`, with
every symbol defined before it is used.

This is the *what and why* document. [`ELASTICITY_MODELS.md`](ELASTICITY_MODELS.md) is the
results document; the notebooks in [`../notebooks/`](../notebooks/) are the tutorial. New
model families are planned in
[`PLAN_PRICETEST_MODELS.md`](PLAN_PRICETEST_MODELS.md) and are **not** yet described here.

**Contents**

* [0. Notation](#0-notation) — the symbol table; read this first
* [1. What an elasticity model must produce](#1-what-an-elasticity-model-must-produce)
* [2. The estimand, stated precisely](#2-the-estimand-stated-precisely)
* [3. The seven models](#3-the-seven-models)
* [4. Tuning objectives](#4-tuning-objectives)
* [5. Evaluation metrics](#5-evaluation-metrics)
* [6. Where each symbol lives in the code](#6-where-each-symbol-lives-in-the-code)

---

## 0. Notation

Every symbol used anywhere in the package. Nothing below this table is introduced without
appearing here.

### 0.1 Indices and raw data

| symbol | meaning |
|---|---|
| $i$ | a **product** (SKU). $i = 1,\dots,I$ with $I = 3{,}218$. |
| $t$ | a **week** (the Monday of an ISO week). $t = 1,\dots,T$ with $T = 103$. |
| $(i,t)$ | a **product-week** — one row of the panel. There are $n = 180{,}233$ of them. |
| $q_{it} \in \mathbb{N}$ | **units sold** of product $i$ in week $t$. Always $\ge 1$: a product-week with no sales is not a row. |
| $p_{it} > 0$ | the **price** of product $i$ in week $t$ — specifically the *modal posted line price*, not `revenue / units` (see [ELASTICITY_MODELS §1.1](ELASTICITY_MODELS.md#11-the-price-measurement-trap)). |
| $X_{it} \in \mathbb{R}^{19}$ | the **control vector**: 19 engineered features listed in `elasticity_lab.features.CONTROL_FEATURES`. Every entry uses only information from weeks strictly before $t$. |

### 0.2 Transformed variables

| symbol | definition | code |
|---|---|---|
| $y_{it}$ | $\log q_{it}$ — the **outcome** | `log_qty`, `OUTCOME` |
| $d_{it}$ | $\log p_{it}$ — the **treatment** ($d$ for *dose*) | `log_price`, `TREATMENT` |
| $r_{it}$ | $d_{it} - \bar d_{i,t}^{(12)}$ — the **relative price**: log price minus the product's own trailing 12-week mean log price. "How deep is this week's discount." | `rel_price` |
| $\bar d_{i,t}^{(12)}$ | the trailing 12-week mean of $d_{i,\cdot}$, computed with `shift(1)` so it excludes week $t$ | — |

> **A notation collision to be aware of.** In the code the residualised treatment is called
> `rt`. That is **"residual of the treatment"** and has nothing to do with the week index
> $t$. Mathematically it is $\tilde d$, below.

### 0.3 The quantity of interest

| symbol | meaning |
|---|---|
| $\varepsilon_{it}$ | the **price elasticity of demand** at $(i,t)$, defined in §2. Reported throughout as a **positive** number: $\varepsilon = 1.5$ means "a 1% price rise costs 1.5% of demand". |
| $\theta_{it}$ | $-\varepsilon_{it}$. The *regression coefficient* on $d$. Negative when demand slopes down. Both appear because the econometrics is natural in $\theta$ and the reporting is natural in $\varepsilon$; **every sign confusion in this codebase came from mixing them**, so both are named. |
| $\hat\varepsilon(x)$ | a model's **prediction** of the elasticity at covariate value $x$. |
| $\bar\varepsilon$ | the population mean elasticity, $\mathbb{E}[\varepsilon]$. |

### 0.4 Nuisance functions and residuals

| symbol | definition | code |
|---|---|---|
| $g(x)$ | $\mathbb{E}[\,y \mid X = x\,]$ — the **outcome nuisance** | `my`, `_model()` in `CausalScorer` |
| $m(x)$ | $\mathbb{E}[\,d \mid X = x\,]$ — the **treatment nuisance** (the continuous analogue of a propensity score) | `mt` |
| $\tilde y_{it}$ | $y_{it} - \hat g(X_{it})$ — the **residualised outcome** | `ry` |
| $\tilde d_{it}$ | $d_{it} - \hat m(X_{it})$ — the **residualised treatment** | `rt` |
| $\eta_{it}$ | the structural error: $y = \theta(x)\,d + g_0(x) + \eta$ with $\mathbb{E}[\eta \mid x,d] = 0$ | — |

### 0.5 Panel effects

| symbol | meaning |
|---|---|
| $\alpha_i$ | **product fixed effect** — everything time-invariant about product $i$ (brand, quality, shelf position). |
| $\gamma_t$ | **week fixed effect** — everything common to week $t$ (season, the Christmas peak, a macro shock). |
| $\mu$ | the grand mean. |

### 0.6 Cross-validation

| symbol | meaning |
|---|---|
| $F$ | the number of folds. |
| $\mathcal{T}_f, \mathcal{V}_f$ | the **training** and **validation** index sets of fold $f$. |
| — | Folds are *rolling-origin*: $\mathcal{T}_f$ is every row with week $< t_f$, $\mathcal{V}_f$ is weeks $[t_f, t_f + h)$ for a horizon $h = 8$. |

### 0.7 Greek letters used for hyperparameters

Kept separate so they are never confused with the structural parameters above.

| symbol | meaning |
|---|---|
| $\delta$ | the **finite-difference step** used to read an elasticity off a fitted surface (§3.4). |
| $\lambda$ | an L2 penalty (`reg_lambda`, `ridge`, `alpha`). |
| $\tau^2$ | the across-product variance of true slopes, in the empirical-Bayes model (§3.7). |
| $\mathrm{se}_i$ | the standard error of product $i$'s raw slope (§3.7). |
| $w_i$ | product $i$'s **shrinkage weight** (§3.7). Not to be confused with the ROC weight vector below. |

### 0.8 Symbols borrowed from `roc3`

Used only where this package touches the 3-class ROC surface.

| symbol | meaning |
|---|---|
| $c$ | the number of classes. |
| $w \in \Delta^{c-1}$ | the **ROC rule weight vector**; the rule is $\hat y = \arg\max_k w_k \hat p_k(x)$. |
| $S_k$ | the **sensitivity** for class $k$ (the diagonal of the row-normalised confusion matrix). |
| $\pi_k$ | the **prevalence** of class $k$. |
| VUS | **volume under the surface** — chance $1/6$, perfect $1$ for $c = 3$. |
| HUM | **hypervolume under the manifold** — the VUS for general $c$; chance $1/c!$. |

---

## 1. What an elasticity model must produce

Every model in `elasticity_lab.models` implements the same four things.

```python
m = build_model("dml_partialling", n_folds=5).fit(train_df)

m.predict_log_qty(valid_df)   # array of length len(valid_df): an estimate of y
m.elasticity(valid_df)        # array of length len(valid_df): an estimate of eps
m.average_elasticity          # float: the model's own mean elasticity
m.params                      # dict: the hyperparameters it was built with
```

Two design decisions are worth stating.

**Elasticity is returned per row, not per model.** A constant-elasticity model returns a
constant array. This costs nothing and means every downstream metric — GATES, BLP, policy
value — takes exactly one code path.

**Elasticity is positive.** Internally the regressions estimate $\theta = -\varepsilon$;
every model negates before returning. The `average_elasticity` property and the
`elasticity()` array are always on the reporting convention of §0.3.

---

## 2. The estimand, stated precisely

The word "elasticity" is used loosely in practice. Three distinct quantities travel under
the name, and the models below do not all estimate the same one.

**(a) The structural elasticity at a point.**

$$\varepsilon(x) \;=\; -\,\frac{\partial \,\log \mathbb{E}[\,q \mid X = x,\; \log p = d\,]}{\partial d}$$

the percentage fall in expected demand from a marginal percentage price rise, holding $X$
fixed. This is what the heterogeneous models (§3.4, §3.6, §3.7) target.

**(b) The average structural elasticity.** $\bar\varepsilon = \mathbb{E}[\varepsilon(X)]$,
averaging over the population of product-weeks. This is what you want for a single
across-the-board pricing decision.

**(c) The variance-weighted average.** What a pooled regression actually returns:

$$\varepsilon^{\text{pooled}} \;=\; \frac{\operatorname{Cov}\!\big(d,\ \varepsilon(X)\,d\big)}{\operatorname{Var}(d)}$$

a weighted average in which product-weeks whose price moved most get the most say. **(b)
and (c) differ whenever elasticity correlates with how much the price moved**, which it does
here — the gap is $-0.295$ on the benchmark even with the confounding switched off entirely
(`experiments/13_elasticity_models_numbers.py`).

Any model reporting a single number is reporting (c) unless it has done something specific
to report (b). This is not a technicality; it is the second of the two independent reasons
the naive regression is wrong.

**Identification.** Every model here assumes **conditional unconfoundedness**: given $X$,
the price is as good as randomly assigned,

$$q(d') \;\perp\!\!\!\perp\; d \;\mid\; X \qquad \text{for every counterfactual price } d',$$

where $q(d')$ is the quantity that *would* have sold at price $d'$. This is an assumption
about the world and cannot be tested from the data. Everything in this package is about
what to do *given* that it holds; nothing in it rescues you if it does not.

---

## 3. The seven models

Ordered by how much structure they impose, most first.

### 3.1 `PooledLogLog` — pooled log-log OLS

$$y_{it} \;=\; a \;-\; \varepsilon\, d_{it} \;+\; c^{\top} X_{it} \;+\; u_{it}$$

Estimated by ordinary least squares; with `ridge` $> 0$, by ridge regression with penalty
$\lambda \lVert (a, \varepsilon, c) \rVert_2^2$.

* **$a$** — intercept. **$c \in \mathbb{R}^{19}$** — control coefficients. **$u$** — error.
* **Elasticity:** the single number $\varepsilon$, returned for every row.
* **Assumes:** one elasticity for every product-week; controls enter *linearly*.
* **Fails when:** the confounder is a nonlinear function of the controls (it is, by
  construction, on the benchmark), or elasticity varies (it does).
* **Hyperparameters:** `use_controls` (bool), `ridge` ($\lambda \ge 0$).
* **Why it is here:** the reference point. Every other model has to beat it.

### 3.2 `TwoWayFixedEffects` — product and week fixed effects

$$y_{it} \;=\; \mu \;+\; \alpha_i \;+\; \gamma_t \;-\; \varepsilon\, d_{it} \;+\; c^{\top} X_{it} \;+\; u_{it}$$

$\alpha_i$ absorbs everything permanent about the product, $\gamma_t$ everything common to
the week. What identifies $\varepsilon$ is only **within-product, within-week** price
movement.

**How it is fitted.** Not by adding 3,300 dummy columns. By the Frisch–Waugh–Lovell theorem
the slope is unchanged if every variable is first *demeaned* by both effects, so the
estimator alternates

$$v \leftarrow v - \operatorname{mean}_i(v), \qquad v \leftarrow v - \operatorname{mean}_t(v)$$

for `n_iter` passes and then runs OLS on the demeaned data. In an unbalanced panel the two
projections do not commute, so this is a Gauss–Seidel iteration rather than a single step;
four passes suffice here.

**Collinear controls must be dropped, not merely tolerated.** A control that is constant
within a week (`week_idx`, `woy_sin`, `is_q4`) is *exactly* collinear with $\gamma_t$; one
constant within a product (`prod_log_price_mean`) with $\alpha_i$. After demeaning those
columns are zero up to floating-point residue, and OLS by pseudo-inverse assigns them
enormous coefficients fitted to that residue. `collinear_tol` drops any column whose
demeaned standard deviation falls below 2% of its raw one. Leaving them in produced a
held-out RMSE of $3.6\times10^{7}$.

* **Elasticity:** the single number $\varepsilon$.
* **Assumes:** one elasticity; the two effects enter *additively*; controls linear.
* **Forecasting caveat:** $\gamma_t$ for a future week is not identified — there is no data
  for it. `predict_log_qty` uses the mean of the last four estimated week effects. Treat
  this as an inference tool that can also predict, not as a forecaster.
* **Hyperparameters:** `add_controls`, `n_iter`, `collinear_tol`.

### 3.3 `PoissonGLM` — a count model with a log link

$$\mathbb{E}[\,q_{it} \mid d_{it}, X_{it}\,] \;=\; \exp\!\big(a - \varepsilon\, d_{it} + c^{\top} X_{it}\big)$$

Every model above takes $\log q$ as the outcome, which assumes additive errors *on the log
scale* and cannot handle a zero. Quantities are counts; a log-link Poisson keeps the
multiplicative reading — $-b_1$ is still the elasticity — while modelling the outcome on its
natural scale, and stays consistent under *multiplicative* demand shocks, which is what a
demand system actually has.

**Fitted by L-BFGS, not IRLS.** IRLS weights equal the fitted mean; with weekly quantities
in the thousands the working Hessian is badly conditioned and the solver does not finish in
useful time. `sklearn.linear_model.PoissonRegressor` on the standardised design converges in
about 1.6 s. Features are standardised internally and the coefficients transformed back, so
`alpha` penalises comparable quantities and the reported elasticity is on the original scale.

* **`predict_log_qty` returns $\log \mathbb{E}[q]$**, not $\mathbb{E}[\log q]$. These differ
  by a Jensen gap, so this model's RMSE is not directly comparable with the others'. Its
  *elasticity* is.
* **Hyperparameters:** `use_controls`, `alpha` ($\lambda$), `max_iter`.

### 3.4 `LightGBMDemand` — the S-learner

Fit one flexible model $\hat f$ of $y$ on **everything** — controls and price together —
then read the derivative numerically:

$$\hat\varepsilon(x) \;=\; -\,\frac{\hat f(x,\, d + \delta) \;-\; \hat f(x,\, d - \delta)}{2\,\delta}$$

"S" is for *single*: one model covers all price levels, as opposed to a T-learner (one model
per treatment level) or an X-learner.

**Two things about this model matter more than its hyperparameters.**

**(i) The perturbation must be consistent.** `rel_price`, `price_change` and `is_discounted`
are all *functions of* `log_price`. Moving `log_price` alone holds the discount depth fixed
while changing the price, which is not a price change, and the derivative it measures is not
the elasticity. `perturb_price(df, delta)` moves all four together. Measured on the real
panel, the shortcut is off by **87% of the answer**.

**(ii) $\delta$ is part of the estimator, not a nuisance knob.** A boosted tree is
*piecewise constant* in $d$, so a step smaller than the typical distance to a price split
divides an almost-always-zero numerator by an almost-zero denominator. Holding the fitted
model fixed and varying only $\delta$ moves the elasticity error from **4.24 to 0.74**. The
damage is **variance, not bias** — the mean elasticity is flat to within 0.02 across a 60×
range of $\delta$. See `experiments/15_why_delta_matters.py`.

* **Elasticity:** per row, heterogeneous.
* **Assumes:** nothing about functional form. That is the appeal, and the problem: nothing
  in the squared-error objective asks it to get the price derivative right.
* **Hyperparameters:** the usual LightGBM set, plus `delta`.

### 3.5 `DoubleMLPartialling` — Robinson's partialling-out with learned nuisances

The **partially linear model**:

$$y \;=\; \theta\, d \;+\; g_0(X) \;+\; \eta, \qquad d \;=\; m(X) \;+\; \nu$$

Learn $\hat g$ and $\hat m$, residualise, and regress residual on residual:

$$\hat\theta \;=\; \frac{\langle \tilde d,\ \tilde y\rangle}{\langle \tilde d,\ \tilde d\rangle}, \qquad \hat\varepsilon = -\hat\theta$$

Two properties make this different in kind from §3.4.

**Neyman orthogonality.** The moment condition
$\psi(\theta; g, m) = (y - g(X) - \theta(d - m(X)))\,(d - m(X))$
has zero Gateaux derivative with respect to $(g, m)$ at the truth. A *first-order* error in
either nuisance costs only a *second-order* error in $\hat\theta$. Nothing similar is true of
reading a derivative off a fitted surface.

**Cross-fitting.** Nuisances are predicted out of fold, so the nuisance model's own
overfitting cannot leak into the residuals. Here the folds are formed **by product** — a
nuisance model never sees the SKU it is predicting for — which is the version of sample
splitting appropriate to a panel.

* **Standard error.** From the orthogonal score:
  $\widehat{\mathrm{se}} = \sqrt{\textstyle\sum_i \psi_i^2}\,/\,\lvert\langle \tilde d, \tilde d\rangle\rvert$
  with $\psi_i = \tilde d_i(\tilde y_i - \hat\theta \tilde d_i)$.
* **Elasticity:** one number. Constant by construction.
* **Hyperparameters:** the LightGBM set for the nuisances, plus `n_folds`.
* Chernozhukov et al. (2018); Robinson (1988).

### 3.6 `RLearnerHeterogeneous` — the R-learner

The same residuals, a different final stage. Instead of one slope, minimise over functions

$$\hat\theta(\cdot) \;=\; \arg\min_{\theta} \; \sum_i \big(\tilde y_i - \theta(x_i)\,\tilde d_i\big)^2$$

Algebraically this is a **weighted regression** of the pseudo-outcome
$\tilde y_i / \tilde d_i$ on $x_i$ with weights $\tilde d_i^{\,2}$, and the weights are the
point: a product-week whose price barely moved has $\tilde d \approx 0$, an essentially
undefined pseudo-outcome, and near-zero weight. The method ignores the rows that contain no
information rather than being misled by them.

`min_weight_quantile` drops the lowest-weight tail outright, because $\tilde y/\tilde d$
overflows before the weight reaches zero.

* **Elasticity:** per row, heterogeneous.
* **Costs data.** It is the same orthogonal score as §3.5 *plus* a second-stage model of
  $\varepsilon(x)$, and that second stage fits noise at small sample sizes. At 800 products it
  is the worst model in the zoo; at 3,218 it lands at bias $-0.133$.
* **Hyperparameters:** the nuisance set, plus `final_*` for the second stage.
* Nie & Wager (2021).

### 3.7 `ShrunkPerProduct` — per-product slopes with empirical-Bayes shrinkage

Run one within-product regression per SKU. On its own this is hopeless: a product with 20
weeks and little price movement produces a slope with an enormous standard error, sometimes
of the wrong sign. Empirical Bayes pulls each slope toward the pooled mean in proportion to
its own noise:

$$\hat\varepsilon_i^{\text{shrunk}} \;=\; w_i\,\hat\varepsilon_i^{\text{raw}} \;+\; (1 - w_i)\,\bar\varepsilon^{\text{pool}}, \qquad w_i \;=\; \frac{\tau^2}{\tau^2 + \mathrm{se}_i^2}$$

* **$\hat\varepsilon_i^{\text{raw}}$** — the OLS slope from product $i$'s own weeks, after
  optionally removing week means (`demean_week`).
* **$\mathrm{se}_i$** — its standard error.
* **$\bar\varepsilon^{\text{pool}}$** — the precision-weighted pooled mean,
  $\sum_i \hat\varepsilon_i / \mathrm{se}_i^2 \big/ \sum_i 1/\mathrm{se}_i^2$.
* **$\tau^2$** — the across-product variance of the *true* slopes, estimated by method of
  moments as $\operatorname{Var}_i(\hat\varepsilon_i^{\text{raw}}) - \operatorname{mean}_i(\mathrm{se}_i^2)$
  (observed spread minus sampling noise), or fixed by the `tau2` hyperparameter.

A product whose price never moved gets $w_i \approx 0$ and lands on the pooled mean, which
is the honest answer for it.

* **Elasticity:** one number per product.
* **Best at ranking, worst at level.** On the benchmark it has the highest rank correlation
  with the truth ($+0.734$) and the largest bias ($-1.093$). Within-product slopes are
  informative about *which* products are elastic and biased about *how* elastic, because
  nothing here removes confounding.
* **Hyperparameters:** `min_obs`, `min_price_sd`, `tau2`, `demean_week`.

### 3.8 Summary

| model | assumes | elasticity | orthogonal | heterogeneous |
|---|---|---|---|---|
| `PooledLogLog` | constant $\varepsilon$; linear controls | one number | ✗ | ✗ |
| `TwoWayFixedEffects` | constant $\varepsilon$; additive $\alpha_i, \gamma_t$ | one number | ✗ | ✗ |
| `PoissonGLM` | constant $\varepsilon$; multiplicative demand | one number | ✗ | ✗ |
| `LightGBMDemand` | nothing about form | per row | ✗ | ✓ |
| `DoubleMLPartialling` | constant $\varepsilon$; learnable nuisances | one number | ✓ | ✗ |
| `RLearnerHeterogeneous` | learnable nuisances | per row | ✓ | ✓ |
| `ShrunkPerProduct` | $\varepsilon$ varies by product only | per product | ✗ | ✓ |

---

## 4. Tuning objectives

Defined in `elasticity_lab.tuning`. All three are computed for **every** trial of every
search, whichever one is being minimised — which is what makes the objective comparison of
[ELASTICITY_MODELS §4](ELASTICITY_MODELS.md#4-hyperparameter-optimisation--the-central-problem)
free.

### 4.1 `rmse` — held-out predictive loss

$$\mathrm{RMSE} \;=\; \sqrt{\frac{1}{\lvert\mathcal{V}\rvert}\sum_{(i,t)\in\mathcal{V}} \big(\hat y_{it} - y_{it}\big)^2}$$

What ordinary HPO minimises, and the wrong objective here: $y$ is dominated by product
identity, season and last week's sales, so a search buys RMSE by flattening the price
dimension.

### 4.2 `rloss` — the orthogonal R-loss

For a candidate whose elasticity function is $\hat\varepsilon(\cdot)$, and residuals
$\tilde y, \tilde d$ from nuisance models fitted on **that fold's training rows only**:

$$R(\hat\varepsilon) \;=\; \frac{1}{\lvert\mathcal{V}\rvert}\sum_{(i,t)\in\mathcal{V}} \Big(\tilde y_{it} \;+\; \hat\varepsilon(X_{it})\,\tilde d_{it}\Big)^{2}$$

The $+$ is not a typo: $\theta = -\varepsilon$, so $\tilde y - \theta\tilde d = \tilde y + \varepsilon \tilde d$.

**Why this is the right objective.** Its population minimiser over all functions is exactly
the true $\varepsilon(\cdot)$ — being good at it *is* being good at the thing we want. And it
is **model-agnostic**: it consumes only the numbers $\hat\varepsilon(x_i)$, so a
fixed-effects regression and a boosted tree land on the same axis using nothing but observed
data.

**One rule.** The nuisance models are fitted **once per fold and shared by every
candidate** (`CausalScorer` caches them). If each candidate brought its own residuals, a
candidate could win by being paired with a lucky nuisance fit.

Two normalisations, because a raw R-loss is on the scale of $\operatorname{Var}(\tilde y)$
and means nothing alone:

| reported as | formula | reads as |
|---|---|---|
| `rloss_skill` | $1 - R(\hat\varepsilon)\,/\,R(0)$ | improvement over "price does nothing" |
| `rloss_vs_constant` | $1 - R(\hat\varepsilon)\,/\,R(\hat\theta^{\text{const}})$ | improvement over one elasticity for everyone |

where $\hat\theta^{\text{const}} = \langle\tilde d,\tilde y\rangle/\langle\tilde d,\tilde d\rangle$
is fitted **on the validation fold itself**. That makes `rloss_vs_constant` the strict test
and **the only honest evidence that estimated heterogeneity is real**.

### 4.3 `oracle` — the grader

$$\mathrm{oracle} \;=\; \sqrt{\frac{1}{\lvert\mathcal{V}\rvert}\sum \big(\hat\varepsilon(X_{it}) - \varepsilon_{it}\big)^2}$$

Requires the true $\varepsilon$, so it exists **only on the semi-synthetic panel**. It is the
*grader* — used to decide which observable objective to trust — and never a legitimate
objective on real data.

### 4.4 Search mechanics

* **TPE with `multivariate=True, group=True`** — models parameter interactions. For boosting,
  `learning_rate` and `n_estimators` are strongly coupled; a search treating them
  independently wastes its budget.
* **Median pruning** — `cv_score` reports its running fold mean after each fold; a trial
  already worse than the median of completed trials at the same fold is abandoned. Verified
  to reach the same optimum as an unpruned search.
* **Nested CV** (`nested_cv`) — a single tuned score is the maximum over many noisy
  estimates and carries the maximum's upward bias. The inner–outer gap *is* the selection
  bias and is worth reporting.

---

## 5. Evaluation metrics

Defined in `elasticity_lab.evaluation`. Nobody observes an elasticity, so these are built
around one idea: **you cannot check one row's elasticity, but you can check a group's**,
because a group contains price variation.

### 5.1 Predictive

`rmse`, `mae`, `r2`, `level_mape`, `bias` on held-out $y$. Necessary, nearly useless: a model
can nail all five while getting the price derivative completely wrong.

### 5.2 `gates` — grouped average treatment effects

Sort validation rows by $\hat\varepsilon$, cut into $G$ bins (default 5), and re-estimate each
bin's elasticity **from the data**:

$$\hat\varepsilon_g \;=\; -\,\frac{\sum_{i \in g}\tilde d_i\,\tilde y_i}{\sum_{i \in g}\tilde d_i^{\,2}}, \qquad \widehat{\mathrm{se}}_g \;=\; \frac{\sqrt{\sum_{i\in g}\tilde d_i^{\,2}(\tilde y_i - \hat\theta_g\tilde d_i)^2}}{\sum_{i \in g}\tilde d_i^{\,2}}$$

The `eps_predicted` column is what the model says; `eps_estimated` is what the data say.
**A real ranking makes the estimated column rise.** A flat estimated column beside a steep
predicted one is the single most common way an elasticity model is wrong while looking fine.

Reported attributes: `monotone`, `spread` = $\hat\varepsilon_G - \hat\varepsilon_1$, and
`spread_se`. A constant-elasticity model gets one group and `monotone = None` — "monotone"
and "not monotone" are both wrong answers to a question it never asked.

Continuous-treatment adaptation of Chernozhukov, Demirer, Duflo & Fernández-Val (2018).

### 5.3 `blp_calibration` — is the *scale* of the heterogeneity right?

GATES checks the ordering; the Best-Linear-Predictor test checks the magnitude. Regress

$$\tilde y \;=\; \beta_0\,\tilde d \;+\; \beta_1\,\tilde d\,\big(\hat\varepsilon(x) - \overline{\hat\varepsilon}\big) \;+\; \text{error}$$

by OLS with HC1 standard errors.

| quantity | reads as |
|---|---|
| $-\beta_0$, reported as `blp_avg_elasticity` | the average elasticity **according to the data** |
| $\beta_1 \approx -1$ | the predicted spread is correctly scaled (negative because $\theta = -\varepsilon$) |
| $\beta_1 \approx 0$ | the predictions carry no usable heterogeneity |
| $\lvert\beta_1\rvert < 1$ | the model **over-states** how much elasticities differ — the usual outcome for a flexible learner |
| $t(\beta_1) < -1.96$ | `heterogeneity_detected` |
| `implied_spread` $= \lvert\beta_1\rvert \cdot \operatorname{sd}(\hat\varepsilon)$ | the data's estimate of the true elasticity spread, **in elasticity units** |

`blp_avg_elasticity` is deliberately *not* called `avg_elasticity`: the latter is the model's
own mean prediction and a different quantity. Naming them alike let one silently overwrite
the other.

`implied_spread` exists because $t(\beta_1)$ alone misleads. On the real panel a tuned
`ShrunkPerProduct` returned $\beta_1 = -21.0$ at $t = -2.2$ — apparently strong evidence —
but its own spread was $0.003$, so the implied spread is $0.07$ elasticity units and there
is nothing to act on.

### 5.4 `policy_value` — what the model is worth

Log revenue is $\log p + \log q$, so nudging log price by $\Delta$ changes log revenue by
$\Delta\,(1 - \varepsilon)$. The model's implied rule is

$$\Delta(x) \;=\; \begin{cases} +\Delta & \hat\varepsilon(x) < 1 \quad\text{(inelastic: raise the price)}\\ -\Delta & \text{otherwise}\end{cases}$$

Its value needs the *true* elasticity, so it is estimated group-wise exactly as in §5.2: bin
by $\hat\varepsilon$, identify each bin's elasticity from its own price variation, and average.

| reported as | meaning |
|---|---|
| `value_model` | $\sum_g w_g\,\Delta_g\,(1 - \hat\varepsilon_g^{\text{data}})$ |
| `value_best_uniform` | the best single $\pm\Delta$ applied to everyone, **including doing nothing** |
| `lift_over_uniform` | the difference — **the number that justifies a heterogeneous model** |

`lift_over_uniform` is routinely zero or negative. Reporting it is what stops a model being
adopted on the strength of a good-looking score.

### 5.5 `oracle_metrics` — synthetic only

`oracle_rmse`, `oracle_bias`, `oracle_mae`, `oracle_corr`, and `share_of_mse_from_bias` =
$\mathrm{bias}^2/\mathrm{RMSE}^2$, which separates "wrong level" from "wrong ranking".

**`oracle_bias` and `oracle_corr` measure different achievements and the best model differs
between them.** `ShrunkPerProduct` has the best correlation and the worst bias;
`DoubleMLPartialling` has the best bias and, being constant, no correlation at all.

### 5.6 `elasticity_summary` — the sanity checks

`mean`, `sd`, `p5`, `median`, `p95`, plus two that matter:

* **`pct_negative`** — the share of rows with $\hat\varepsilon < 0$, i.e. **upward-sloping
  demand**. A few may be genuine; many means the heterogeneity is noise. On the real panel
  the untuned S-learner reports 6.7%.
* **`pct_inelastic`** — the share with $\hat\varepsilon < 1$, where a price rise *raises*
  revenue. This is the decision boundary of §5.4.

---

## 6. Where each symbol lives in the code

| maths | code | module |
|---|---|---|
| $y$ | `OUTCOME` = `"log_qty"` | `features` |
| $d$ | `TREATMENT` = `"log_price"` | `features` |
| $r$ | `"rel_price"` | `features` |
| $X$ | `CONTROL_FEATURES` (19 names) | `features` |
| $\tilde y$ | `ry` | `tuning.CausalScorer.residuals` |
| $\tilde d$ | `rt` | `tuning.CausalScorer.residuals` |
| $\hat g$, $\hat m$ | `my`, `mt` | `tuning`, `models` |
| $\theta$ | `theta_`, `beta_[0]` | `models` |
| $\varepsilon$ | `elasticity()`, `avg_elasticity_` | `models` |
| $\delta$ | `delta` | `models.LightGBMDemand` |
| $\tau^2$, $w_i$ | `tau2_`, `weights_` | `models.ShrunkPerProduct` |
| $\alpha_i$, $\gamma_t$ | `prod_fe_`, `week_fe_` | `models.TwoWayFixedEffects` |
| $R(\hat\varepsilon)$ | `rloss` | `tuning.CausalScorer.r_loss` |
| $\beta_0, \beta_1$ | `beta0`, `beta1` | `evaluation.blp_calibration` |
| $\mathcal{T}_f, \mathcal{V}_f$ | `scorer.folds_[f]` | `tuning`, `splits` |

---

## 7. References

* Robinson, P. (1988). Root-N-consistent semiparametric regression. *Econometrica* 56(4).
* Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., Robins,
  J. (2018). Double/debiased machine learning. *Econometrics Journal* 21(1).
* Chernozhukov, V., Demirer, M., Duflo, E., Fernández-Val, I. (2018). Generic machine
  learning inference on heterogeneous treatment effects. NBER w24678.
* Nie, X., Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects.
  *Biometrika* 108(2).
* Deaton, A. (1988). Quality, quantity, and spatial variation of price. *AER* 78(3).
* Frisch, R., Waugh, F. (1933); Lovell, M. (1963) — the demeaning theorem behind §3.2.
* Efron, B., Morris, C. (1975). Data analysis using Stein's estimator — the shrinkage of §3.7.
