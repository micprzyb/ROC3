# Plan: why the GLM won, and twelve ways to make that stop being true

**The challenge.** A logistic regression beat gradient boosting, a T-learner and the
arm-posterior model on every measure that mattered. That is *prima facie* evidence the
flexible models are under-optimised, and the criticism is largely correct: **not one model
in `experiments/18` or `19` had a single hyperparameter tuned.** They ran on library
defaults.

But before tuning anything, it is worth being precise about *how many* separate advantages
the GLM was enjoying, because two of them are not about tuning at all and one of them is
my fault.

---

## 0. Diagnosis: four distinct reasons the GLM won

**(a) It is correctly specified.** The simulator generates $s = \sigma(a(x) - \beta(x)\log p)$
and `ConversionGLM` fits exactly that, with $\varepsilon = \beta(1-s)$ in closed form. No
other model gets a free functional form. Any benchmark in which the parametric model matches
the DGP is rigged in its favour, and the honest fix is **I12**, not better tuning.

**(b) It pools all three arms.** With 0.1/0.8/0.1 the arm-based methods use the 20% that
carries price variation; the GLM also uses the 80% control traffic to pin down $a(x)$. This
is a real structural advantage and it survives any amount of tuning of the others.

**(c) The treatment feature is wrong for a tree — my error.** This is the important one.

$$\log p \;=\; \underbrace{\log(\text{technical premium})}_{\text{also a feature}} \;+\; \log(\text{arm multiplier})$$

A linear model separates these exactly: with both terms in the design, the coefficient on
$\log p$ *is* the arm effect. **A tree cannot.** A split on `log_price` at some threshold
mixes "this is an expensive customer" with "this customer got the dear arm" — and the first
varies by a factor of ten across the book while the second varies by ±10%. Every split the
tree spends on `log_price` is almost entirely about customer identity.

This is exactly the `rel_price` lesson from the retail panel, where the whole reason that
feature exists is to normalise price *within* a product. I engineered it there and then
failed to carry it over here.

**(d) The elasticity is read off the wrong scale.** `ConversionGBM` differences *predicted
probabilities*. The model is additive in the **logit**, and for bounded demand
$\varepsilon = -(\partial\eta/\partial\log p)(1-s)$. Differencing $\eta$ and multiplying by
$(1-s)$ is both more stable and structurally correct.

So: (a) is unfair and needs a different simulator, (b) is real and permanent, (c) and (d)
are bugs, and tuning addresses none of them. Tuning is idea **I5** of twelve.

---

## 1. Twelve ideas

Ordered by expected value, with what would falsify each.

### I1 — Give the models the treatment, not the price *(feature engineering)*

Add `arm_log_multiplier` (the randomised part of the price, and nothing else) and drop or
keep `log_price` alongside. For a tree this converts the treatment from a variable it cannot
isolate into one it can split on cleanly at two thresholds.

*Expected:* the largest single gain of the twelve. *Falsified if:* the GBM does not improve.

### I2 — Monotone constraints in the price direction

LightGBM's `monotone_constraints` can force conversion to be non-increasing in the treatment.
Guarantees $\varepsilon \ge 0$ by construction and regularises exactly the dimension that
matters. The published cost of monotonicity in GBM credit models is **0–2.9%** of accuracy —
close to free on large data.

*Falsified if:* accuracy falls materially, or the negative-elasticity rate does not go to zero.

### I3 — Differentiate in logit space

$\varepsilon = -(\partial\eta/\partial\log p)\,(1-s)$ with $\eta$ the raw score. Removes the
probability-scale differencing that made $\delta$ so delicate in the retail work, and bakes
in the bounded-demand structure that §1.2 of [`INSURANCE.md`](INSURANCE.md) establishes.

### I4 — Competitive position as an explicit feature

The DGP drives conversion through $\log(\text{competitor}) - \log(\text{price})$. Both terms
are available; neither model is given the **difference**. A tree would need many splits to
approximate a difference of two continuous features. Add it.

*This generalises beyond the simulator:* in real motor pricing, position against the market
is the single strongest conversion feature, and it is always a difference.

### I5 — Actually tune the hyperparameters *(the stated criticism)*

Optuna over every model. Search spaces per model, the finite-difference $\delta$ included as
a first-class parameter (it was the most consequential one in the retail study), graded on
**profit regret** and selected on an estimable objective.

### I6 — AIPW / doubly-robust arm effects *(a missing model)*

The propensity is **known exactly**, so the augmented estimator

$$\hat\mu_k(x) \;=\; \hat m_k(x) \;+\; \frac{\mathbb{1}[A = k]}{\pi_k}\bigl(Y - \hat m_k(x)\bigr)$$

is unbiased whatever $\hat m$ does, and strictly lower variance than plain IPW. This is the
standard uplift construction in the industry and it is entirely absent from the model set.
With 0.1/0.8/0.1 the variance reduction is the whole game.

### I7 — Control-anchored T-learner

The current T-learner fits three independent models and so throws away 80% of the data when
estimating each side arm. Instead: fit $\hat s_{\text{control}}$ on the 80% (very precise),
then model only the **increment** $\eta_k - \eta_{\text{control}}$ for $k \ne$ control, with
the control fit as an `init_score` offset. Same idea as `GLMThenGBM`, applied across arms.

### I8 — Recalibrate the probabilities

Boosted probabilities are shrunk toward the base rate; in the retail work that shrinkage
caused a 27% elasticity attenuation that VUS provably could not see. Isotonic or Platt
recalibration on held-out data, then recompute the elasticity.

### I9 — R-learner with a binary outcome and known propensity

The orthogonal score that won the retail benchmark, absent here. With randomisation the
treatment residual is exactly $\mathbb{1}[A=k] - \pi_k$ — no nuisance model needed for it,
which removes the main source of error the retail version had.

### I10 — Saturated treatment instead of a log-linear slope

Only three prices were run, so a single slope in $\log p$ is an *assumption*, not a
necessity. Estimate a free effect per arm and let the curvature show. Connects directly to
the arc-elasticity machinery already built in `pricelevels`.

### I11 — Give the GLM interactions too, so the comparison is fair

If the GBM's advantage is meant to be interactions, the honest baseline is a GLM **with**
price × feature interactions and splines — the specification the actuarial literature
actually recommends. Otherwise "GBM beats GLM" only ever measures whether the analyst wrote
the interactions down.

### I12 — Break the GLM's free lunch: a misspecified simulator

Add DGP variants where demand is *not* logit-linear in $\log p$:

* a **probit** link;
* a logit with a **nonlinear index** (thresholds, a shopping-behaviour kink);
* **heterogeneous curvature** so no single link fits everyone.

Any claim that a flexible model beats a GLM is only interesting under misspecification. This
is the idea that makes the whole comparison meaningful, and it belongs in the same run.

---

## 2. Plan

**Phase A — the three bugs and the fair baseline** (`insurance.py`)
I1, I3, I4, I11. Cheap, and they change what "tuned" even means, so they come first.

**Phase B — new models**
I6 (AIPW), I7 (control-anchored T-learner), I9 (R-learner), I2 + I10 in the GBM.

**Phase C — tuning** (`experiments/20`)
I5: Optuna per model, graded on profit regret.

**Phase D — the fair test** (`experiments/21`)
I12: rerun the tuned set under logit, probit and kinked DGPs. Report per-DGP, never pooled.

**Predictions, recorded now.**

1. I1 alone closes most of the GBM's gap; the GBM's ε-RMSE falls below 1.0 at ±10%.
2. Monotone constraints cost < 3% of log-loss and take negative elasticities to exactly 0.
3. AIPW beats plain IPW on variance at every arm width — this is close to a theorem.
4. Under a **misspecified** DGP the tuned GBM beats the GLM; under the logit DGP it does not.
5. Tuning helps the flexible models materially and the GLM barely at all (it has two
   hyperparameters, one of which is a regularisation strength on a correctly specified model).
6. The GLM **with interactions** (I11) closes much of whatever gap remains — meaning the
   headline is "write down your interactions", not "use a GBM".

Prediction 6 is the one I most expect to be wrong, and it is the most useful either way.


---

## 3. RETRACTED — results, and four predictions that failed

> **This section is withdrawn as evidence about model quality.** Every comparison below is
> run on simulated data, so it reports which model matches my generator. The GLM's wins in
> §3.3 in particular are an artefact: the simulator draws a logit and the GLM fits a logit.
>
> Two things in here are not model rankings and do survive:
>
> * **§3.1 / §3.2** — that tuning on a predictive objective moved profit regret in the
>   *wrong direction* while an oracle objective moved it a long way in the right one. This
>   is a statement about the relationship between two loss functions on the same fitted
>   models, and the same finding was established independently on **real data** in the
>   retail study (`ELASTICITY_MODELS.md` §4.4). The simulation adds a second instance, not
>   the evidence.
> * **The `monotone` finding in §3.3** — that offering a law-of-demand constraint as a
>   tunable, under a predictive objective, causes the search to switch it off. That is a
>   property of the search procedure and holds whatever the data.
>
> Everything else — which model wins under which link, the interaction results, the
> ε-RMSE table — is withdrawn.

`experiments/20_insurance_v2.py` (150,000 quotes, 10 trials/model/objective, four demand
shapes) and `experiments/21_observable_objective.py`.

### 3.1 The criticism was correct, and tuning is not the fix

Nothing in experiments 18–19 was tuned. Tuning it changes the picture — but mostly for the
worse. Regret **saved** by tuning on held-out log-loss (positive = helped):

| model | logit | probit | kinked | mixed |
|---|---|---|---|---|
| `glm (v1)` | +0.09 | +0.11 | +0.00 | −0.11 |
| `gbm (v1)` | **−6.98** | **−8.65** | −4.97 | **−8.69** |
| `gbm (v2)` | −4.96 | −6.86 | +19.25 | +0.19 |
| `tlearner (anchored)` | −1.46 | −1.33 | −3.28 | −1.42 |

**Tuning on predictive log-loss made the flexible models worse at pricing**, by up to 8.7
units of regret. The GLM is unaffected either way — it has one hyperparameter on a
correctly-specified model, so **prediction 5 is confirmed**.

### 3.2 But the capacity is there — the selection criterion is throwing it away

What an *oracle* objective would have bought, same budget:

| model | logit | probit | kinked | mixed |
|---|---|---|---|---|
| `glm (v1)` | +0.09 | +0.11 | +0.00 | +0.08 |
| `gbm (v2)` | **+6.23** | **+6.10** | **+19.25** | **+7.23** |
| `tlearner (anchored)` | +3.08 | +3.40 | +6.16 | +2.32 |
| `aipw arm effects` | +3.13 | +2.61 | +16.08 | +4.74 |

So the flexible models are not short of capacity or of trials. They are short of a
**selection objective that knows what they are for** — the same finding as the retail R-loss
study, arriving in a completely different setting.

### 3.3 Where my predictions failed

**Prediction 4 — refuted.** I expected the tuned GBM to beat the GLM under misspecification.
It does not. Tuned ε-RMSE, best per column in bold:

| model | kinked | logit | mixed | probit |
|---|---|---|---|---|
| `glm (v1)` | **0.377** | **0.841** | 1.352 | 1.244 |
| `glm+interactions (v2)` | 0.756 | 0.864 | 1.333 | 1.231 |
| `tlearner (anchored)` | 0.870 | 1.286 | **1.292** | 1.395 |
| `gbm (v2)` | 0.991 | 1.840 | 1.880 | 2.085 |

The GLM wins three of four *even when it is misspecified*. Only under `mixed` — where the
link's curvature varies across customers — does anything beat it.

**Prediction 6 — refuted, and in the opposite direction.** Giving the GLM interactions and
splines made it **worse**, not better: on `kinked` its regret went from 0.12 to 16.19. The
extra flexibility bought overfitting, not expressiveness.

**Prediction 2 — refuted, instructively.** Monotone constraints did *not* eliminate negative
elasticities in `gbm (v2)`: 31–37% of them are negative. The reason is that I let the search
*choose* `monotone`, and a log-loss objective **switches the constraint off** — unconstrained
models fit log-loss better. A constraint that encodes a law of demand is not a hyperparameter
to be traded against predictive fit, and offering it as one silently discards it.
`experiments/21` hard-codes it.

**Prediction 1 — partly confirmed.** The I1–I4 fixes improved the GBM's ε-RMSE by about 23%
(logit 2.383 → 1.840, probit 2.600 → 2.085, mixed 2.575 → 1.880) but did not close the gap.

### 3.4 The proposed fix also mostly failed

If log-loss is the wrong objective, the natural replacement is the value of the model's
implied pricing policy, estimated from held-out quotes — unbiased because the propensity is
known. Regret saved against not tuning:

| objective | logit | kinked |
|---|---|---|
| log-loss | −2.39 | +6.31 |
| IPW policy value | −6.88 | +1.58 |
| AIPW policy value | −4.08 | −2.03 |

**The policy objectives are not reliably better than log-loss, and on `logit` they are
worse.** One clear success: the anchored T-learner on `kinked` reaches regret **0.214**
against an oracle bound of **0.143**, where log-loss selection gives 0.921. Everywhere else
they add noise.

The reason is the allocation again. A policy that recommends a side arm can only be scored
on the 10% of validation quotes that received it, so the objective a search is optimising is
itself estimated from a tenth of the data — and with a dozen trials the search fits that
noise. **The 0.1/0.8/0.1 split does not merely weaken estimation; it makes decision-based
model selection unreliable too.**

### 3.5 What this actually recommends

1. **Do not tune a pricing model on predictive loss.** It made every flexible model worse
   here, by up to 8.7 units of regret.
2. **Impose the monotonicity, never offer it.** Left to a predictive objective the search
   removes it and a third of the elasticities go negative.
3. **A correctly-specified parametric model is very hard to beat**, and stays hard to beat
   under mild misspecification. Reach for flexibility when the *shape* varies across
   customers (`mixed`), which is the case none of the parametric forms can cover.
4. **Fix the allocation first.** §6.4 of [`INSURANCE.md`](INSURANCE.md) showed 0.2/0.6/0.2
   beats 0.1/0.8/0.1 for *estimation*; §3.4 here shows it is also the binding constraint on
   *selection*. Both problems have the same cheap fix.
5. The open problem is a **low-variance observable causal objective** for this design. AIPW
   was the right instinct and is not enough on its own; a sensible next step is variance
   reduction on the policy-value estimator itself — a doubly-robust score with a fitted
   baseline, or switching from a hard arm choice to a smoothed one so the estimate is not
   carried by 10% of rows.
