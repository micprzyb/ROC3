# Price-elasticity models on public retail data

A worked build of price-elasticity models on the UCI *Online Retail II* panel, with the
hyperparameter-optimisation question treated as the central research problem rather than a
step at the end.

**Code:** `elasticity_lab/` · **Notebooks:** `notebooks/` · **Full-scale runs:**
`experiments/10_*.py`–`12_*.py`

This document is the reference. The notebooks are the tutorial:

| notebook | question |
|---|---|
| `01_data_and_the_identification_problem` | Is the question even answerable on this data, and why does the obvious regression fail? |
| `02_model_zoo` | Seven estimators — what each assumes and what each recovers. |
| `03_hyperparameter_optimization` | **How do you tune a model whose target you cannot observe?** |
| `04_comparison_and_selection` | Is the heterogeneity real, and what is it worth? |

---

## 1. The data

Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/502/online+retail+ii>

1,067,371 transaction lines from a UK online giftware wholesaler, 2009-12-01 to 2011-12-09.
Public, no registration. After cleaning (returns, credit notes, non-product stock codes,
non-positive prices) and aggregation to product × ISO-week, keeping products active in ≥ 20
weeks:

| | |
|---|---|
| product-weeks | 183,451 raw; 180,233 after dropping rows without a lag |
| products | 3,218 |
| weeks | 104 raw; 103 modellable |
| median weeks per product | 53 |
| products with sd(log price) > 0.10 | **80.8%** |
| products with sd(log price) < 0.02 (uninformative) | 10.1% |
| panel fingerprint | `37e45faa3a46` |

The dataset was chosen because it has genuine price variation. Most public retail data does
not, and without price variation an elasticity is not identified at all.

### 1.1 The price-measurement trap

The obvious price for a product-week is `revenue / quantity`. **It is unusable.** In this
data larger order lines are discounted:

```
within product-week:  d log(line price) / d log(line quantity) = −0.1212
                      (t = −500,  n = 1,037,081)
```

so a week containing one big order mechanically shows a low unit value and a high quantity.
That is **division bias** (Borjas 1980; Deaton 1988) — a downward-sloping "demand curve"
produced by arithmetic. Measured on this panel:

| price measure | two-way FE elasticity | se |
|---|---|---|
| `price_modal` (default) | **1.571** | 0.008 |
| `price_median` | 1.555 | 0.008 |
| `price_unitvalue` | 2.004 (**+27.6%**) | 0.008 |

The standard errors are tiny in both cases. This is a bias no amount of data fixes and no
confidence interval warns you about. `elasticity_lab.data` therefore defaults to the
**modal posted line price**, which is not a function of the week's quantity.

### 1.2 Features, and the two rules

`elasticity_lab.features` builds 19 controls and 4 price features under two rules:

1. **Nothing looks forward.** Every lag and rolling statistic uses `shift(1)` inside the
   product.
2. **Nothing is a function of the current week's quantity.** The same trap as the unit
   value.

The split between `CONTROL_FEATURES` and `PRICE_FEATURES` is load-bearing: the price
features are functions of the treatment and must never enter a nuisance model that is
supposed to be price-free.

### 1.3 Cross-validation

`KFold(shuffle=True)` leaks twice on a panel — through time (lags and rolling means) and
through product identity. `elasticity_lab.splits` provides four designs and measures both
channels:

| design | weeks shared train/valid | % valid products also in train |
|---|---|---|
| `NaiveKFold` (wrong) | 515 | 100% |
| `RollingOriginSplit` | 0 | 100% |
| `PurgedRollingSplit` (gap=12) | 0 | 93% |
| `GroupedProductSplit` | all | 0% |

Rolling origin is the default for HPO because it matches deployment. `NaiveKFold` is
provided so the leak can be measured rather than asserted.

---

## 2. Why the naive regression fails — two reasons, not one

`elasticity_lab.simulate` builds a semi-synthetic twin: real prices, real covariates, real
calendar, simulated quantities with a known per-product elasticity.

```
log price  =  real log price  +  θ·u  +  noise
log qty    =  3.0 + α_product + γ_week + g(X) − ε_product·log price + λ·u + noise
```

with `u = h(X)` a **nonlinear function of observed controls** — removable in principle,
which is what makes the benchmark a fair test of tuning rather than a rigged one. Making
`u` genuinely unobservable would be more pessimistic and completely uninformative: no
method could succeed and every hyperparameter would look equally bad.

`naive_bias_decomposition` splits the naive estimate's error into channels, and the split
is an **exact identity**, checked against the realised OLS coefficient. At the setting the
notebooks use (θ = 0.6, λ = 0.8), on the full panel:

```
                       channel  contribution
          TRUE mean elasticity       +1.6560
        product effect (alpha)       -0.0091
           week effect (gamma)       -0.0027
          baseline demand g(X)       -0.0514
                  confounder u       -0.3117
                residual noise       -0.0003
      elasticity heterogeneity       -0.1169
      = implied naive estimate       +1.1639
     actual naive OLS estimate       +1.1639
  identity check (should be 0)       +0.0000
```

The confounder is the biggest channel here, which is exactly what you would expect at
θ = 0.6 — we set it that way. The row worth stopping on is the last one.

**Elasticity heterogeneity is a source of error that has nothing to do with confounding.**
A pooled regression does not estimate the average elasticity; it estimates

```
Cov(log p, ε·log p) / Var(log p)
```

a **variance-weighted** average in which products whose price moved most dominate. If those
products differ systematically in elasticity from the rest — and they do, here and in
reality — the pooled number is wrong even with no confounding at all. Setting θ = λ = 0 and
re-running settles it:

```
          TRUE mean elasticity       +1.6560
        product effect (alpha)       -0.0102
           week effect (gamma)       -0.0018
          baseline demand g(X)       +0.2934
                  confounder u       -0.0000
                residual noise       +0.0001
      elasticity heterogeneity       -0.2952
      = implied naive estimate       +1.6423
     actual naive OLS estimate       +1.6423
```

The confounder channel is now exactly zero and the heterogeneity channel is **larger than
before, not smaller** (−0.295 against −0.117). It is a second, independent defect, and
controlling for confounders — however well — does not touch it. Two problems, two tools:

| problem | fix |
|---|---|
| confounding | controls, fixed effects, orthogonalisation |
| heterogeneity weighting | estimate `ε(x)` per unit and average deliberately |

---

## 3. The model zoo

`elasticity_lab/models.py`. Common interface: `fit(df)`, `predict_log_qty(df)`,
`elasticity(df)`, `average_elasticity`. Elasticity is always reported as a **positive**
number.

| model | assumes | output |
|---|---|---|
| `PooledLogLog` | constant ε; linear controls | one number |
| `TwoWayFixedEffects` | constant ε; additive product & week effects | one number |
| `PoissonGLM` | constant ε; multiplicative demand, count outcome | one number |
| `LightGBMDemand` (S-learner) | nothing about functional form | per row |
| `DoubleMLPartialling` | constant ε; learnable nuisances | one number, orthogonal |
| `RLearnerHeterogeneous` | nothing; learnable nuisances | per row, orthogonal |
| `ShrunkPerProduct` | ε varies by product only | per product |

### 3.1 Untuned results, full panel (180,233 rows, 3,218 products)

**Real panel** — no ground truth; this is the situation you are actually in:

| model | mean ε | sd(ε) | in-sample RMSE | seconds |
|---|---|---|---|---|
| `pooled_loglog` | 1.510 | 0 | 0.989 | 1.1 |
| `twoway_fe` | 1.509 | 0 | 0.980 | 2.3 |
| `poisson_glm` | 1.618 | 0 | 1.165 | 7.4 |
| `lgbm_demand` | 2.182 | 1.889 | **0.728** | 4.7 |
| `dml_partialling` | 1.030 | 0 | 0.708 | 19.7 |
| `r_learner` | 1.160 | 0.794 | 0.708 | 20.7 |
| `shrunk_per_product` | 1.745 | 0.737 | 1.035 | 0.6 |

Estimates range from 1.03 to 2.18. Nothing in this table says which is right.

**Semi-synthetic panel** — graded against `true_elasticity` (true mean **1.656**):

| model | mean ε | bias | RMSE vs truth | corr with truth |
|---|---|---|---|---|
| `dml_partialling` | 1.582 | **−0.074** | **0.713** | — (constant) |
| `r_learner` | 1.523 | −0.133 | 0.899 | +0.376 |
| `poisson_glm` | 2.202 | +0.546 | 0.895 | — |
| `twoway_fe` | 0.964 | −0.692 | 0.991 | — |
| `pooled_loglog` | 0.871 | −0.785 | 1.058 | — |
| `lgbm_demand` | 1.079 | −0.578 | 1.090 | +0.316 |
| `shrunk_per_product` | 0.531 | −1.093 | 1.223 | **+0.734** |

Three findings:

**The flexible predictive model is not the accurate one.** `lgbm_demand` has the best
predictive RMSE and an elasticity attenuated by 35%. This is not bad luck. Price explains a
small share of the variance in `log q` compared with product identity and seasonality, so
shrinking the price dimension is a *good move* for prediction. Nothing in the squared-error
loss says otherwise.

**Orthogonalisation is what fixes it.** `dml_partialling` uses the same gradient boosting,
the same features and the same hyperparameters, and is nearly unbiased — because the price
effect comes from residuals rather than from a fitted surface.

**Getting the mean right and getting the ranking right are different achievements.**
`shrunk_per_product` has the *best* rank correlation (+0.734) and the *worst* bias
(−1.093). `dml_partialling` has the best bias and cannot rank at all. Which matters depends
on the decision — one average price move, or a different move per product.

**Heterogeneity is expensive in data.** The notebooks run on an 800-product subsample so
they execute in minutes, and there the ordering changes in an instructive way: `r_learner`
becomes the *worst* model (oracle RMSE 1.56 against Double ML's 0.81) with a predicted
spread more than twice the true one. It is the same orthogonal score as `dml_partialling`
plus a second-stage model of `eps(x)`, and at 800 products that second stage fits noise.
The method is not broken — at 3,218 products it lands at bias −0.133 — it simply needs more
data than its constant-effect sibling, which is the right default until heterogeneity can be
demonstrated (§5.1).

### 3.2 Three implementation traps, each of which changed the answer

**Perturbing price consistently.** `rel_price`, `price_change` and `is_discounted` are all
functions of `log_price`. A finite-difference elasticity that moves `log_price` alone holds
the discount depth fixed while changing the price, and measures the wrong derivative.
`perturb_price` moves all of them together.

**Collinear controls in a fixed-effects model.** A control constant within week
(`week_idx`, `woy_sin`, `is_q4`) is exactly collinear with the week effects; one constant
within product (`prod_log_price_mean`) with the product effects. After demeaning those
columns are zero up to floating-point residue, and OLS by pseudo-inverse hands them
enormous coefficients fitted to that residue. Leaving them in produced a **held-out RMSE of
3.6 × 10⁷** against a constant-mean baseline of 1.75. On this panel the columns separate
cleanly:

| group | absorbed sd / raw sd |
|---|---|
| exactly absorbed (9 columns) | ≤ 2.5 × 10⁻⁸ |
| *nearly* absorbed — `market_log_qty`, `market_log_price` | 0.9%–1.5% |
| genuinely within-varying (9 columns) | ≥ 23% |

The near-absorbed pair is the dangerous one: it survives a floating-point-scale test and
still gets an unstable coefficient. `collinear_tol` defaults to 2%. Dropping them fixed the
held-out RMSE (0.98 vs a 1.75 baseline) **and** moved the synthetic elasticity from 0.149 to
0.925 — the near-collinearity was corrupting the slope, not just the levels.

**Poisson by L-BFGS, not IRLS.** IRLS weights equal the fitted mean; with weekly quantities
in the thousands the working Hessian is badly conditioned and statsmodels' solver does not
finish in any reasonable time. `sklearn.linear_model.PoissonRegressor` on the standardised
design converges in ~1.6 s.

### 3.3 A performance note

`n_jobs=-1` is the wrong default for LightGBM on data this size. On the 24-core host used
here, one fit of 400 trees on ~22k × 23:

| `n_jobs` | idle host | host under load |
|---|---|---|
| 1 | 0.33 s | 1.01 s |
| 4 | 0.12 s | 0.76 s |
| 8 | **0.09 s** | **0.80 s** |
| −1 | **30.68 s** | **679 s** |

A **340× slowdown idle, 849× under contention** — the trees are small, so the threads spend
their time synchronising rather than working, and the pathology gets worse the busier the
machine is. Across a hyperparameter search that is the difference between minutes and days.
`elasticity_lab.models.N_THREADS` fixes it at `min(8, cores/2)`; override with
`ELASTICITY_LAB_THREADS`.

(Both columns are measured; `experiments/13_elasticity_models_numbers.py` re-measures the
ratio on whatever machine it runs on and asserts only that it exceeds 20×, because the
absolute figure is machine- and load-dependent while the conclusion is not.)

---

## 4. Hyperparameter optimisation — the central problem

`elasticity_lab/tuning.py`.

Ordinary HPO minimises held-out predictive loss. For an elasticity model that is the wrong
objective. The measurable loss is the error in predicting `log q`; the quantity of interest
is `d log q / d log p`. A search minimising the former will buy RMSE by flattening the price
dimension and return a well-tuned model with an attenuated elasticity.

### 4.1 Three objectives

| name | definition | available on |
|---|---|---|
| `rmse` | held-out RMSE of `log q` | any data |
| `rloss` | orthogonal R-loss (below) | any data |
| `oracle` | RMSE against `true_elasticity` | synthetic only — the **grader**, never a legitimate objective |

### 4.2 The R-loss

Write `log q = θ(x)·log p + g(x) + η` with `θ = −ε`. Residualise on the controls using
nuisance models fitted on the fold's **training** rows:

```
ry = log q − Ê[log q | x]        rt = log p − Ê[log p | x]
R(θ̂)  =  (1/n) Σ ( ry_i − θ̂(x_i)·rt_i )²
```

Its population minimiser over all functions is exactly `θ(·)`, so being good at this *is*
being good at the thing we want. And it is **model-agnostic** — it consumes only
`θ̂(x_i)`, so a fixed-effects regression and a boosted S-learner are comparable on one axis
using only observed data. (Robinson 1988; Nie & Wager 2021.)

**The nuisance models are fitted once per fold and shared by every candidate.** Otherwise a
candidate could win by being paired with a lucky nuisance fit. `CausalScorer` caches them.

Two normalisations, because a raw R-loss means nothing alone:

| reported as | formula | reads as |
|---|---|---|
| `rloss_skill` | `1 − R(model)/R(θ=0)` | improvement over "price has no effect" |
| `rloss_vs_constant` | `1 − R(model)/R(best constant θ)` | improvement over one elasticity for everyone |

`rloss_vs_constant` is the strict one and the **only honest evidence that estimated
heterogeneity is real** — the best constant is fitted on the same validation fold.

### 4.3 Search mechanics

* **TPE, `multivariate=True, group=True`** — models interactions. `learning_rate` and
  `n_estimators` are strongly coupled; a search treating them independently wastes its
  budget.
* **Median pruning** — `cv_score` reports its running fold mean after each fold; a trial
  already worse than the median of completed trials at the same fold is abandoned.
* **Every trial records every metric.** A trial optimising RMSE still stores its R-loss and
  oracle error, so a completed search can be re-read under a different objective with no
  refitting. This is what makes the rank-correlation analysis free.
* **Nested CV** (`nested_cv`) — a single tuned score is the maximum over many noisy
  estimates and carries the maximum's upward bias. The gap between inner and outer is the
  selection bias, and worth reporting.

### 4.4 Results

`experiments/11_hpo_experiment.py 1200 30` — 1,200 products, 30 trials, 4 rolling-origin
folds, every model tuned twice under identical budgets, both winners graded against
`true_elasticity`. 373 completed trials, ~2 hours. `experiments/14_hpo_analysis.py`
re-reads the saved trial histories (every trial recorded *all* metrics, so this costs a
second rather than another two hours).

**Does the objective change the answer?**

| model | oracle [rmse] | oracle [rloss] | gain | % | RMSE given up |
|---|---|---|---|---|---|
| `lgbm_demand` | 2.6043 | **0.9911** | +1.6132 | **+62%** | +0.149 |
| `pooled_loglog` | 1.2544 | **0.8773** | +0.3771 | +30% | +0.463 |
| `r_learner` | 0.9708 | **0.8183** | +0.1525 | +16% | +0.262 |
| `shrunk_per_product` | 1.3797 | **1.2948** | +0.0849 | +6% | +0.005 |
| `poisson_glm` | 1.0463 | 1.0454 | +0.0009 | +0.1% | +0.174 |
| `twoway_fe` | 1.0266 | 1.0262 | +0.0004 | +0.0% | +0.001 |
| `dml_partialling` | **0.7630** | 0.8105 | −0.0475 | −6% | +0.211 |

**6 of 7 models recover elasticity better when tuned on the R-loss.** The largest gain is
the S-learner's — a 62% reduction in elasticity error, bought for 0.15 of predictive RMSE.

**But the level is the wrong statistic.** It depends on where one search happened to land.
HPO does not need an objective numerically close to the truth — a monotone transform of the
truth would be perfect while being numerically nothing like it. It needs one that **orders
configurations the way the truth does.** Spearman rho against the oracle, across every trial:

| model | orthogonal | trials | oracle spread | ρ(R-loss) | ρ(predictive RMSE) |
|---|---|---|---|---|---|
| `pooled_loglog` | | 52 | 0.938 | **+0.999** | −0.538 |
| `lgbm_demand` | | 57 | **4.185** | **+0.936** | −0.439 |
| `twoway_fe` | | 54 | 0.271 | **+0.922** | +0.555 |
| `shrunk_per_product` | | 53 | 0.232 | **+0.903** | +0.232 |
| `r_learner` | ✓ | 53 | 1.066 | **+0.741** | −0.204 |
| `poisson_glm` | | 47 | 0.218 | −0.143 | −0.870 |
| `dml_partialling` | ✓ | 57 | **0.121** | **−0.967** | +0.399 |

Median ρ: **+0.903** for the R-loss, **−0.204** for predictive RMSE. Predictive RMSE
**anti-ranks for 4 of 7 models** — for those, a better-predicting configuration is reliably
a worse elasticity estimate, so tuning on it is worse than not tuning at all.

### 4.5 The exception, and why it does not overturn the recommendation

`dml_partialling` is the one model where the R-loss ranks configurations **backwards**
(ρ = −0.97). The mechanism is not mysterious. A Double ML estimate is Neyman-orthogonal, so
its elasticity barely moves with its own nuisance hyperparameters — and the R-loss is scored
with a **fixed** nuisance model shared by all candidates. Whatever residual confounding that
scoring model leaves behind defines the θ it prefers, and that θ is slightly biased, so
candidates closer to the truth score marginally worse.

**The R-loss is therefore only as good as the nuisance model used to score it**, and for an
estimator that is already orthogonal there is little left for it to fix.

What makes the mistake cheap is not orthogonality but **how much is at stake in that
search**. The `oracle spread` column is the range of elasticity error across all
configurations tried. For `dml_partialling` it is **0.121**, the narrowest in the table, so
ranking it backwards costs −0.0475. For `lgbm_demand` it is **4.185**, the widest by a
factor of four, and there the R-loss gains +1.61. That asymmetry is the argument: the
R-loss helps most exactly where the choice matters most, and its one failure is on the model
where the choice matters least.

(`r_learner` is also orthogonal and the R-loss ranks it +0.741, so this is not a property of
orthogonal estimators as a class. Its second-stage model of ε(x) has hyperparameters that
genuinely move the answer — spread 1.07 — and the R-loss selects among them correctly.
`poisson_glm` is a different non-result: its elasticity hardly depends on its
hyperparameters at all, spread 0.218 over 47 trials, so there is nothing for any objective
to select between.)

### 4.6 What the search actually found, and why

The two winning S-learner configurations differ most in one place — and it is not a knob
anyone would have flagged in advance:

| | tuned on R-loss | tuned on RMSE |
|---|---|---|
| `delta` (finite-difference step) | **0.184** | **0.015** |
| `n_estimators` | 400 | 1150 |
| `num_leaves` | 38 | 72 |

`delta` is the step used to turn a fitted demand surface into an elasticity:
`eps = −[f(x, p+δ) − f(x, p−δ)] / 2δ`. `experiments/15_why_delta_matters.py` isolates it by
holding one fitted model fixed and varying only the step — so nothing about the fit changes,
only the ruler:

| δ | rows with an exactly-zero difference | mean ε | bias | RMSE vs truth |
|---|---|---|---|---|
| 0.005 | 12.0% | 1.228 | −0.434 | **4.241** |
| 0.015 | 2.4% | 1.179 | −0.482 | 2.344 |
| 0.050 | 0.7% | 1.172 | −0.490 | 1.278 |
| 0.180 | 0.3% | 1.184 | −0.478 | 0.823 |
| 0.300 | 0.2% | 1.207 | −0.455 | **0.741** |

**The elasticity error moves by a factor of 5.7 while the fitted model is unchanged.**

The intuitive explanation — that a small step attenuates the elasticity toward zero — is
wrong, and the table says so: the **bias is flat to within 0.02** across the whole sweep.
What δ controls is **variance**. A boosted tree is piecewise constant in log price, so a
step smaller than the typical distance to a price split divides an almost-always-zero
numerator by an almost-zero denominator. The average survives; the per-row predictions do
not. (A second plausible guess also fails: deeper trees make the dead zone *smaller*, not
larger — 16.0% at 8 leaves against 1.8% at 255 — because more leaves means more price splits
to cross.)

Two things follow.

* **A model can have a respectable average elasticity and worthless per-row predictions.**
  Reporting only the mean hides this entirely; `oracle_rmse` and the BLP calibration of §5
  do not.
* **Predictive RMSE cannot see any of it.** The fitted surface is identical for every row of
  that table, so predictive loss is *exactly the same* while the elasticity error moves from
  4.24 to 0.74. A search minimising predictive loss is choosing δ at random. The R-loss can
  see it, because a high-variance elasticity fits the residualised outcome badly.

### 4.7 The rule

* **Tune on the R-loss.** It ranks correctly for 5 of 7 models, at a median cost of +0.17 in
  predictive RMSE — the right trade, because predictive RMSE is not the deliverable.
* **Do not tune on predictive RMSE.** It anti-ranks for 4 of 7.
* **Invest the search budget where the spread is large.** A search whose `oracle spread` is
  0.12 is not worth 30 trials whatever the objective.
* **Remember the R-loss inherits its scoring nuisance model's bias.** If your estimator is
  already orthogonal, the R-loss has little to add and can mislead at the margin.

---

## 5. Evaluation without labels

`elasticity_lab/evaluation.py`. Nobody observes an elasticity, so the usual
prediction-vs-target comparison does not exist. Three families of answer, in increasing
order of usefulness.

**Predictive metrics** — necessary, nearly useless. A model can nail these while getting
the price derivative completely wrong.

**Grouped identification** — the honest core. You cannot check one row's elasticity, but you
can check a *group's*, because a group contains price variation.

* `gates(ry, rt, eps_hat)` — sort by predicted elasticity, cut into bins, re-estimate each
  bin's elasticity orthogonally from the data. A real ranking makes the estimated column
  rise. A flat estimated column next to a steep predicted one is the most common way an
  elasticity model is wrong while looking fine.
* `blp_calibration(ry, rt, eps_hat)` — regress `ry ~ β₀·rt + β₁·rt·(ε̂ − mean ε̂)`.
  `β₁ ≈ −1` means the predicted spread is correctly scaled; `β₁ ≈ 0` means no usable
  heterogeneity; `|β₁| < 1` means the model over-states how much elasticities differ, which
  is the usual outcome for a flexible learner. The `t` on `β₁` is a formal test.

Continuous-treatment adaptations of Chernozhukov, Demirer, Duflo & Fernández-Val (2018).

**Decision value** — `policy_value(...)`. Log revenue is `log p + log q`, so a price nudge
`d` changes log revenue by `d·(1 − ε)`: raise the price where `ε < 1`, cut it where
`ε > 1`. The counterfactual is estimated group-wise, exactly as in GATES. Always reported
against `value_best_uniform`, the best single ±δ applied to everyone including doing
nothing. `lift_over_uniform` is routinely zero or negative, and reporting it is what stops a
model being adopted on the strength of a good-looking score.

### 5.1 What the battery says, on both panels

`experiments/12_final_comparison.py 900 15` — every model tuned on the R-loss, then the
full battery on the last rolling-origin fold.

**Semi-synthetic** (true mean elasticity 1.66), sorted by the oracle column the real panel
does not have:

| model | own mean ε | BLP mean ε | sd(ε) | rloss_vs_const | β₁ | t(β₁) | GATES ↑ | lift vs flat | oracle RMSE | oracle corr |
|---|---|---|---|---|---|---|---|---|---|---|
| `poisson_glm` | 1.671 | 1.343 | 0 | −0.019 | | | | 0 | **0.720** | |
| `dml_partialling` | 1.533 | 1.343 | 0 | −0.006 | | | | 0 | 0.739 | |
| `r_learner` | 1.367 | 1.249 | 0.500 | **+0.013** | −0.71 | −8.5 | ✓ | **+0.0041** | 0.751 | +0.439 |
| `pooled_loglog` | 1.097 | 1.343 | 0 | −0.011 | | | | 0 | 0.941 | |
| `twoway_fe` | 1.089 | 1.343 | 0 | −0.011 | | | | 0 | 0.946 | |
| `lgbm_demand` | 0.791 | 1.239 | 0.538 | −0.018 | −0.53 | −6.9 | ✓ | −0.0085 | 1.119 | +0.500 |
| `shrunk_per_product` | 0.686 | 1.177 | 0.498 | +0.003 | −0.97 | −12.6 | ✓ | −0.0009 | 1.145 | **+0.680** |

`r_learner` is the only model that both beats the constant on held-out R-loss and beats the
flat pricing policy, and it is third by oracle error — the observable checks and the
unobservable truth agree.

**Which observable metric would have chosen correctly?** Spearman against `oracle_rmse`,
sign-adjusted so positive means "agrees with the truth":

| observable metric | ρ |
|---|---|
| `lift_over_uniform` | **+0.670** |
| `rmse` / `r2` / `rloss_skill` / `rloss_vs_constant` | −0.179 |

The **decision-value metric is the best model selector** — better than the R-loss
normalisations, which is not surprising once stated: it is the metric closest to what the
model is for. Read with care, though: seven models with the top three within 0.03 of each
other makes this rank correlation noisy. The safe reading is that `lift_over_uniform`
deserves a place in the selection battery, not that it dominates.

### 5.2 Does the answer depend on the world?

Untuned defaults across four data-generating processes, ranked by oracle RMSE (1 = best):

| model | exogenous price | mild confounding | heavy confounding | Poisson outcome |
|---|---|---|---|---|
| `dml_partialling` | 3 | **1** | **1** | **1** |
| `pooled_loglog` | 2 | 2 | 3 | 2 |
| `lgbm_demand` | 6 | 3 | 6 | 3 |
| `poisson_glm` | 4 | 4 | 2 | 4 |
| `twoway_fe` | 5 | 6 | 4 | 6 |
| `shrunk_per_product` | **1** | 5 | 5 | 5 |
| `r_learner` | 7 | 7 | 7 | 7 |

Double ML wins wherever there is confounding to remove, which is the case it was designed
for. `shrunk_per_product` wins **only** under exogenous prices — with nothing to confound
them, within-product slopes are unbiased and shrinkage is pure gain. That is the concrete
form of "no method is universally best" (Curth & van der Schaar 2021): the ranking is a
property of the world as much as of the estimator, so report the range. `r_learner` is last
everywhere here because these runs use 300 products; see §3.1 on its data appetite.

### 5.3 The real panel — the honest deliverable

No ground truth. Every model tuned on the R-loss, 900 products, 15 trials:

| model | own mean ε | BLP mean ε | sd(ε) | rloss_vs_const | t(β₁) | GATES ↑ | lift vs flat | checks passed |
|---|---|---|---|---|---|---|---|---|
| `r_learner` | 1.305 | 1.353 | 0.413 | **+0.0018** | −3.53 | ✗ | −0.0003 | **2 / 4** |
| `dml_partialling` | 1.414 | 1.284 | 0 | −0.0017 | | | 0 | 0 |
| `poisson_glm` | 1.430 | 1.284 | 0 | −0.0021 | | | 0 | 0 |
| `pooled_loglog` | 1.491 | 1.284 | 0 | −0.0043 | | | 0 | 0 |
| `shrunk_per_product` | 1.526 | 1.295 | 0.003 | −0.0058 | −2.22 | ✗ | 0 | 1 / 4 |
| `twoway_fe` | 1.544 | 1.284 | 0 | −0.0068 | | | 0 | 0 |
| `lgbm_demand` | **1.929** | 1.296 | **1.194** | **−0.115** | −0.39 | ✗ | **−0.0057** | 0 / 4 |

**The deliverable for this dataset is: an average elasticity of roughly 1.3–1.5, and no
per-product heterogeneity anyone should act on.**

The constant-elasticity models agree closely (1.41–1.54) and the BLP estimate from the data
is 1.28 — a reassuringly tight consensus for a number estimated seven different ways.

`lgbm_demand` is the cautionary row and worth reading across. It reports the widest spread
in the table (sd 1.19) and the highest average (1.93). Every check says the spread is
noise: it is **worse than a single constant** on held-out R-loss (−0.115), its β₁ is
statistically indistinguishable from zero (t = −0.39), its GATES is not monotone, and acting
on it **loses** 0.0057 of log revenue against the best flat move. A practitioner reading
only "the model found elasticities from 0.2 to 7" would deploy it.

`shrunk_per_product` shows why β₁ must be read with its scale. Its t = −2.22 looks
significant, but the tuned model's own spread is 0.003, so |β₁| = 21.0 and the
`implied_spread` = |β₁|·sd(ε̂) is only **0.07** elasticity units. Significant and not worth
acting on.

### 5.4 A selection rule

1. Tune on the **R-loss**, never on predictive RMSE.
2. Check `rloss_vs_constant` on held-out data. ≤ 0 → use the constant-elasticity model.
3. Check the BLP `beta1_t` **next to `implied_spread`**. Not significantly negative → no
   detectable heterogeneity, whatever the spread of predictions looks like; significant
   with a tiny `implied_spread` → detectable and not worth acting on.
4. Check GATES monotonicity. Non-monotone → the ranking is unreliable even if the average is
   fine.
5. Check `lift_over_uniform`. If targeting does not beat the best flat move, the complexity
   is not paying.
6. Report **nested-CV** numbers, and sanity-check `pct_negative` — a model producing many
   upward-sloping demand curves is telling you its predictions are noise.

Steps 2–5 are computable without ever knowing a single true elasticity.

A constant-elasticity model failing checks 3–5 is not a mark against it: it never claimed
heterogeneity. The checks decide between models that *do* make the claim, and tell you when
to fall back to one that does not.

---

## 6. Reproducing

```bash
pip install -r requirements.txt

# data (downloads ~45 MB from UCI on first call, then caches to parquet)
python -c "from elasticity_lab.data import get_panel; print(get_panel().shape)"

python experiments/10_model_zoo_smoke.py            # the zoo, full panel        (~2 min)
python experiments/11_hpo_experiment.py 1200 30     # the objective experiment    (~2 h)
python experiments/12_final_comparison.py 900 15    # tuned models, all diagnostics (~45 min)
python experiments/13_elasticity_models_numbers.py  # re-checks every number in §1-§3
python experiments/14_hpo_analysis.py               # re-reads 11's trials         (~1 s)
python experiments/15_why_delta_matters.py          # the delta mechanism         (~5 min)

cd notebooks && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Notebooks default to a subsample (`N_PRODUCTS = 700`–`800`) so they execute in minutes; set
it to `None` for the full-panel numbers quoted above. Subsampling is by **whole product** —
dropping random rows would tear holes in the lag structure.

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
* Borjas, G. (1980). The relationship between wages and weekly hours of work. *J. Human
  Resources* 15(3).
* Curth, A., van der Schaar, M. (2021). Nonparametric estimation of heterogeneous treatment
  effects. *AISTATS*.
* Akiba, T. et al. (2019). Optuna. *KDD*.

See `docs/ELASTICITY.md` and `docs/ELASTICITY_TUTORIAL.md` for the theory of elasticity
evaluation from a price-test classifier, which this work applies to observational data.
