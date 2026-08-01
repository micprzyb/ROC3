# Plan: price-test classifiers, the VUS link, and two-stage residual models

**Status: plan only. Nothing here is implemented yet.** Written before any code, as
requested. Notation follows [`MODELS.md §0`](MODELS.md#0-notation) throughout; symbols
introduced here are added to the table in §2.1 below.

---

## 1. The four questions

1. **Price-test-first models.** Instead of regressing $\log q$ on $\log p$, predict *which
   price the sale happened at* and read the demand curve off the classifier — the
   construction of [`PRICETEST.md`](PRICETEST.md), which assumed randomisation, transported
   to observational data.
2. **Constrained classification.** The same, with the hard logical constraint
   $0 < p_1 < p_2 < p_3 < 1$ (and its $K$-level generalisation) enforced rather than hoped
   for.
3. **VUS as the quality measure**, and the question that matters: **how does a model's VUS
   relate to the quality of the elasticity it implies?**
4. **Two-stage residual models** — a simple first stage (a GLM), a richer second stage on
   its residuals.

§2–§6 are the research. §7–§10 are the build.

---

## 2. Research: the price-level classifier

### 2.1 Additional notation

| symbol | meaning |
|---|---|
| $K$ | the number of **price levels**. $K = 3$ mirrors the user's $-10\%/0\%/+10\%$ test; the design is general in $K$. |
| $\ell_{it} \in \{1,\dots,K\}$ | the **price level** of product-week $(i,t)$, **ordered by increasing price**: $\ell = 1$ is the cheapest. |
| $\bar r_k$ | the mean relative log price $r$ within level $k$. By construction $\bar r_1 < \dots < \bar r_K$. |
| $s_k(x)$ | **demand at level $k$**: $\mathbb{E}[\,q \mid X = x,\ \ell = k\,]$. |
| $\pi_k(x)$ | the **propensity**: $\mathbb{P}(\ell = k \mid X = x)$ — how often the seller puts this product-week at level $k$. |
| $b_k(x)$ | the **unit-weighted posterior**: the probability that a randomly chosen *sold unit* from covariate value $x$ came from a week at level $k$. |
| $c_k > 0$ | a **gauge factor** (§4). |
| $\Delta^{K-1}$ | the probability simplex on $K$ classes. |

### 2.2 The randomised case, recalled

In a genuine price test with arms $k$ assigned with known probabilities $\pi_k$, Bayes gives,
for a customer who bought,

$$\mathbb{P}(\text{arm} = k \mid x,\ \text{bought}) \;=\; \frac{\pi_k\, s_k(x)}{\sum_j \pi_j\, s_j(x)}$$

With equal randomisation the $\pi$'s cancel and **the arm posterior is the demand curve,
normalised**. That is the identity [`PRICETEST.md`](PRICETEST.md) is built on, and the
source of the user's constraint: monotone demand ($p_1 \le p_2 \le p_3 \Rightarrow s_1 \ge s_2 \ge s_3$)
forces the posterior to be ordered.

This is a **choice-based / endogenously stratified sample** in the sense of Manski & Lerman
(1977): the sample is selected on the outcome (only buyers appear) and the label being
predicted is the treatment. The literature's correction — reweight by the known assignment
probabilities — is exactly the division by $\pi_k$ above. It is also the *label shift* /
*prior correction* problem of Elkan (2001) and Saerens et al. (2002).

### 2.3 The observational generalisation

We have no arms. We construct them: cut the relative price $r$ at its $K$-quantiles, so
level 1 is the deepest discount and level $K$ the highest relative price, and the levels are
balanced by construction (so the *marginal* $\pi_k \approx 1/K$, mirroring equal
randomisation — though $\pi_k(x)$ still varies).

**Lemma 1 (observational price-test identity).**

$$b_k(x) \;=\; \frac{\pi_k(x)\, s_k(x)}{\sum_j \pi_j(x)\, s_j(x)} \qquad\Longrightarrow\qquad \frac{s_k(x)}{s_j(x)} \;=\; \frac{b_k(x)\,/\,\pi_k(x)}{b_j(x)\,/\,\pi_j(x)}$$

*Proof.* The expected number of units observed at level $k$ for covariates $x$ is
(frequency of level $k$) × (units per occurrence) $= \pi_k(x)\,s_k(x)$. Normalising over $k$
gives $b$. The ratio follows. $\square$

**Corollary.** If $\pi_k(x) \equiv 1/K$ the propensity cancels and $s \propto b$: §2.2 is the
special case of Lemma 1 under randomisation. The observational estimator *reduces to the
price-test estimator exactly* when the price test is real.

So the estimator is **two classifiers and a slope**:

| step | what is fitted | on what |
|---|---|---|
| **A** | $\hat\pi_k(x)$ — the generalised propensity score | multiclass on product-weeks, unweighted |
| **B** | $\hat b_k(x)$ — the unit-weighted posterior | multiclass on product-weeks, `sample_weight` $= q_{it}$ |
| **C** | $\log \hat s_k(x) = \log \hat b_k(x) - \log \hat\pi_k(x)$ | pointwise |
| **D** | $\hat\varepsilon(x) = -\,\text{slope of } \log\hat s_k(x) \text{ on } \bar r_k$ | weighted least squares over $k$ |

Step D, written out, with weights $\omega_k$ (uniform by default):

$$\hat\varepsilon(x) \;=\; -\,\frac{\sum_k \omega_k\,(\bar r_k - \bar r_\omega)\,\log \hat s_k(x)}{\sum_k \omega_k\,(\bar r_k - \bar r_\omega)^2}, \qquad \bar r_\omega = \frac{\sum_k \omega_k \bar r_k}{\sum_k \omega_k}$$

The unknown common factor in $\hat s$ is an additive constant in $\log \hat s$ and drops out
of a slope. Good.

**Arc elasticities come free.** For $K > 2$ we also get

$$\hat\varepsilon_{k,k+1}(x) \;=\; -\,\frac{\log \hat s_{k+1}(x) - \log \hat s_k(x)}{\bar r_{k+1} - \bar r_k}$$

so the estimator recovers the **shape** of the demand curve, not just one slope. No model in
the existing zoo can do that: they all impose constant elasticity or read a single local
derivative. This is a genuine capability gain and worth reporting on its own.

### 2.4 What this buys, and what it costs

**Buys.** (i) Nonparametric in price — the demand curve is $K$ free points, so curvature is
visible. (ii) It is the *only* member of the zoo that is literally the user's price-test
model, so a real experiment and this panel can be analysed with one piece of code. (iii) The
$\pi$-division is an inverse-propensity correction, so it is a recognisable causal estimator
with known behaviour, not an ad-hoc device. (iv) The `roc3` diagnostic suite applies to it
directly.

**Costs.** (i) Discretisation throws away within-level price variation. (ii) It is a *ratio*
estimator: $\hat b_k/\hat\pi_k$ blows up where $\hat\pi_k \approx 0$, the classic IPW
instability, so it needs propensity clipping and an overlap diagnostic. (iii) It inherits
the same unconfoundedness assumption as everything else; discretising does not weaken it.
(iv) It is *not* orthogonal as written — errors in $\hat\pi$ enter at first order. A
doubly-robust version is possible and is listed as future work in §10, not built now.

---

## 3. Research: the constraint

### 3.1 What the constraint is on

The user's constraint is $0 < p_1 < p_2 < p_3 < 1$, meaning the cheapest arm has the highest
posterior. Under randomisation that is the same statement as **monotone demand**,
$s_1 \ge s_2 \ge \dots \ge s_K$. Observationally the two come apart: the posterior $b$ is
distorted by the propensity, so

> the constraint must be imposed on the **demand** $s$, not on the raw posterior $b$.

Imposing it on $b$ would be wrong here — it would force the seller's pricing policy to be
monotone too, which is not a law of anything.

Equivalently, since $\bar r_k$ is increasing, the constraint is exactly

$$\hat\varepsilon_{k,k+1}(x) \;\ge\; 0 \quad \text{for every } k \text{ and every } x.$$

**Every arc elasticity is non-negative.** This matters concretely: the untuned S-learner
reports upward-sloping demand for **6.7%** of real product-weeks. The constrained classifier
cannot do that — it is ruled out by construction, not by luck.

### 3.2 Three ways to enforce it

**(C1) Monotone-logit parameterisation — the principled one.** Parameterise the log demand
curve as a decreasing sequence by construction:

$$\log \hat s_k(x) \;=\; -\sum_{j < k} \operatorname{softplus}\big(u_j(x)\big), \qquad \operatorname{softplus}(z) = \log(1 + e^{z})$$

so $\log\hat s_1 = 0$ (the free additive constant, fixed by convention) and each subsequent
level is strictly lower. Then fit the multinomial likelihood for the *observed* level with
the estimated propensity as a known **offset**:

$$\hat b_k(x) \;=\; \operatorname{softmax}_k\Big(\log\hat\pi_k(x) \;+\; \log\hat s_k(x)\Big)$$

maximising $\sum_{it} q_{it}\,\log \hat b_{\ell_{it}}(X_{it})$ (unit-weighted, i.e. weighted
by quantity — the choice-based likelihood).

With $u_j(x) = a_j + w_j^{\top}x$ this is $(K-1)(1 + \dim X) = 40$ parameters for $K = 3$,
fitted by L-BFGS with an analytic gradient. Cheap, deterministic, and every parameter is
interpretable: $w_j$ says which covariates make arc $j$ steeper.

**(C2) Isotonic repair — the model-agnostic one.** Take *any* unconstrained pair of
classifiers, form $\log \hat s_k(x)$, and project onto the decreasing cone by the
pool-adjacent-violators algorithm (PAVA), weighted by $\hat\pi_k(x)$. This is the $L_2$
projection onto $\{v : v_1 \ge \dots \ge v_K\}$ and is exact and $O(K)$.

It lets us answer a clean question: **is the benefit of the constraint in the fitting or in
the repair?** If C2 ≈ C1, the constraint is just post-processing and any model can have it.
If C1 > C2, fitting under the constraint genuinely uses the information.

**(C3) Penalised loss** — add $\lambda \sum_k \max(0, \log\hat s_{k+1} - \log\hat s_k)^2$.
Considered and **not planned**: it is C1 with a tunable failure mode, and C1 costs nothing
more.

### 3.3 Gauge or ceiling?

[`CONSTRAINED.md`](CONSTRAINED.md) established that this distinction decides everything, and
the two readings give opposite answers:

* a constraint on the **model's output** is a *gauge choice* — VUS is bit-for-bit unchanged
  and nothing is learned;
* a constraint on the **true posterior** is a fact about the world — it cuts ROC space with
  $K-1$ planes, the perfect corner becomes unreachable, and VUS is capped at a computable
  $\mathrm{VUS}_{\max}(\pi) < 1$.

Here monotone demand is a **fact about the world** (a law of demand), so we are in the second
regime and the ceiling applies. That is not a limitation to apologise for; it is the reason
raw VUS is uninterpretable on this task and must be normalised. Which brings us to the
question the user actually asked.

---

## 4. Research: how VUS relates to elasticity quality

This is the part with a real answer, and it is sharper than expected: **VUS and the
elasticity *level* are provably independent**, and VUS speaks only to *heterogeneity*.

### 4.1 The gauge theorem, and its consequence

**Theorem (gauge invariance; established in `CONSTRAINED.md`, verified numerically to
machine precision).** For any positive $c_1,\dots,c_K$, replacing the score vector
$b$ by the renormalised $\big(c_k b_k\big)_k$ leaves the ROC surface and every VUS
estimator **bit-for-bit unchanged**.

**Proposition 1 (the gauge moves the elasticity).** Under the same transform,

$$\log\hat s_k \;\longmapsto\; \log\hat s_k + \log c_k + \text{const}, \qquad\text{hence}\qquad \hat\varepsilon_{k,k+1} \;\longmapsto\; \hat\varepsilon_{k,k+1} \;-\; \frac{\log c_{k+1} - \log c_k}{\bar r_{k+1} - \bar r_k}.$$

*Proof.* Immediate from step C of §2.3 and the definition of the arc elasticity; the
renormalising denominator is common to all $k$ and cancels in differences. $\square$

**Corollary (the answer).** The gauge group has $K-1$ effective dimensions (overall scale
does nothing), and Proposition 1 says it acts on the $K-1$ arc elasticities by an arbitrary
translation. So the group **acts transitively on the space of elasticity vectors while
leaving VUS exactly fixed**. Therefore:

> **VUS carries no information whatsoever about the elasticity level.** A model with
> $\mathrm{VUS} = 1$ can have any elasticity you like, including the wrong sign. A model at
> chance VUS can have a perfect average elasticity.

What fixes the level is the **calibration** — knowing $\pi_k$. In a randomised price test
$\pi_k = 1/K$ is known *by design*, which is precisely what randomisation buys. Observationally
the level is only as good as $\hat\pi$, and errors in $\hat\pi$ pass into the elasticity
one-for-one while leaving VUS untouched.

This is the 3-class version of a familiar fact — AUC is invariant to monotone rescaling and
therefore says nothing about calibration — but it is much sharper here, because the
transformation that VUS cannot see is *exactly* the one that sets the answer.

### 4.2 What VUS does measure

**Proposition 2 (homogeneous elasticity ⇒ chance VUS, under multiplicative demand).**
Suppose $s_k(x) = s(x)\exp(-\varepsilon_0 \bar r_k)$ — constant elasticity — and
$\pi_k(x) \equiv \pi_k$. Then

$$b_k(x) = \frac{\pi_k e^{-\varepsilon_0 \bar r_k}}{\sum_j \pi_j e^{-\varepsilon_0 \bar r_j}}$$

is **constant in $x$**: the baseline $s(x)$ cancels exactly. A constant score vector cannot
discriminate, so the Bayes-optimal VUS is chance. Yet $\varepsilon_0$ is *exactly* recoverable
from that constant posterior by Lemma 1. $\square$

So on the count/multiplicative panel, **VUS above chance is evidence of elasticity
heterogeneity, and nothing else.**

**Proposition 3 (bounded demand breaks Proposition 2).** If demand is a purchase
*probability*, $s_k(x) = \sigma(a(x) - \beta \bar r_k)$, then $\log s_k - \log s_j$ depends on
$a(x)$ even for constant $\beta$, because the local elasticity of a logistic curve,
$-\beta \bar r(1 - \sigma)$, varies along the curve. The posterior varies with $x$ and VUS
exceeds chance from **baseline heterogeneity alone**. $\square$

**This distinction matters and is easy to get wrong.** In the binary price-test setting of
`PRICETEST.md`, a VUS above chance is *not* by itself evidence that elasticity varies. On the
count panel here, it is. The two settings must not be conflated.

### 4.3 The ceiling, and the only VUS number worth reporting

`roc3.pricetest.ceiling_from_elasticity` computes $\mathrm{VUS}_{\max}$ from the demand
response. Its behaviour is the key to interpretation:

* at $\varepsilon_0 = 0$ the levels are indistinguishable in principle, the priors are
  uniform, and $\mathrm{VUS}_{\max} = $ chance exactly;
* as $\varepsilon_0$ grows the arms separate and the ceiling rises — at unit elasticity with
  $\pm 10\%$ prices the attainable band is only $[0.167,\ 0.219]$.

So the **ceiling is set by the elasticity level** and the **position within the band is set
by heterogeneity**. The two are cleanly separated, and the only interpretable number is

$$\mathrm{VUS}_{\text{norm}} \;=\; \frac{\mathrm{VUS} - 1/K!}{\mathrm{VUS}_{\max} - 1/K!} \;\in\; [0, 1] \qquad (\texttt{roc3.constrained.normalized\_vus})$$

read as **"what fraction of the achievable heterogeneity did the model find?"**

A raw VUS of 0.20 is near-perfect against a ceiling of 0.219 and near-worthless against a
ceiling of 0.9. Reporting raw VUS on this task is close to meaningless, and this is the main
practical recommendation of §4.

### 4.4 Falsifiable predictions

Stated now, before implementation, so they can fail.

| # | prediction | how it is tested | what would refute it |
|---|---|---|---|
| **P1** | With `elasticity_sd = 0` on the multiplicative panel, the classifier's VUS is at chance, while its recovered $\bar\varepsilon$ is accurate. | one synthetic frame | VUS materially above chance |
| **P2** | Sweeping `elasticity_sd` $\in \{0, 0.2, 0.4, 0.6, 0.8\}$, $\mathrm{VUS}_{\text{norm}}$ rises monotonically while `oracle_bias` stays flat. | 5 frames | non-monotone, or bias tracking the sweep |
| **P3** | Across models and hyperparameter settings, $\mathrm{VUS}_{\text{norm}}$ correlates strongly with `oracle_corr` and **not** with $\lvert$`oracle_bias`$\rvert$. | Spearman over all trials | either correlation coming out the other way |
| **P4** | A random gauge $c$ leaves VUS unchanged to $10^{-12}$ and shifts each arc elasticity by exactly $-(\log c_{k+1} - \log c_k)/(\bar r_{k+1} - \bar r_k)$. | direct | any discrepancy — this one is a theorem, so a failure is a bug |
| **P5** | Measured VUS never exceeds `ceiling_from_elasticity(measured ε)` beyond sampling error. | direct | an excess, which would indicate leakage |

**P3 is the one that answers the user's question**, and P1/P2 are what make P3 interpretable.
P4 is a correctness test on the implementation.

---

## 5. Research: two-stage residual models

### 5.1 The idea and its names

Fit something simple, then fit something flexible on what is left over. In actuarial pricing
this is the **offset** construction — and, as the CAS literature notes, "when the offset is
the predicted value from a preliminary model, specifying it as an offset is equivalent to
using the residual from the preliminary model as the target". Wüthrich & Merz's **Combined
Actuarial Neural Net (CANN)** is the same idea with a network as stage two. LightGBM exposes
it as `init_score`.

The user's phrasing — *"the first model is simple (like a GLM) and the next model is built on
the residues from the first model (perhaps with more features)"* — is exactly this, and the
"more features" part is the interesting bit: stage 1 gets a small interpretable feature set,
stage 2 gets everything.

### 5.2 Why it might help an *elasticity* model specifically

A boosted tree is piecewise constant and cannot represent a smooth log-linear price slope
efficiently; it spends splits approximating a straight line, and regularisation flattens it
(this is the §3.4 failure of `MODELS.md`). A GLM represents that slope in one coefficient,
exactly and by construction. So:

> anchor the price effect in the GLM, and let the tree model only the *deviation*.

The deviation is smaller and centred near zero, so the flattening bias has less to bite on.

### 5.3 Three variants, and what distinguishes them

| | stage 1 | stage 2 sees price? | elasticity |
|---|---|---|---|
| **T1 `GLMOffsetBoost`** | Poisson GLM / log-log OLS on `SIMPLE_FEATURES` | **no** | $\varepsilon_0$ from stage 1 alone |
| **T2 `GLMResidualBoost`** | same | **yes** | $\varepsilon_0 + $ finite difference of stage 2 |
| **T3 `AnchoredRLearner`** | same, gives $\theta_0$ | orthogonal residuals | $\varepsilon_0 + h(x)$ |

**T1** is the honest workhorse: stage 2 cannot touch the price derivative, so the elasticity
is a clean, interpretable GLM coefficient while the predictions get the benefit of the tree.
Its bias is exactly the GLM's bias — no better, no worse. Its value is that it decouples
"predict well" from "estimate the elasticity", which is the entire tension of this project.

**T2** is the version a practitioner would write, and I expect it to be **worse than T1 at
elasticity while better at prediction** — the tree re-imports the flattening problem on the
correction term. If that is what happens it is a useful negative result; if T2 beats T1 the
anchoring argument in §5.2 is wrong and should be withdrawn.

**T3** is the methodologically interesting one. Writing $\theta(x) = \theta_0 + h(x)$ in the
R-loss gives

$$\sum_i \big(\tilde y_i - \theta_0\tilde d_i - h(x_i)\tilde d_i\big)^2 \;=\; \sum_i \big(\tilde y_i' - h(x_i)\tilde d_i\big)^2, \qquad \tilde y_i' \equiv \tilde y_i - \theta_0 \tilde d_i$$

— **the R-learner on a re-residualised outcome**. So T3 is a plain R-learner whose second
stage only has to learn the *deviation* from a sensible anchor rather than the whole
function from a noisy pseudo-outcome. Given that `RLearnerHeterogeneous` is the model that
most obviously runs out of data (worst in the zoo at 800 products), this is a principled fix
and my best guess for the largest gain in this batch.

Note T3 remains orthogonal: the anchor changes only the starting point of the second stage,
not the estimating equation.

### 5.4 `SIMPLE_FEATURES`

Stage 1 needs a small, defensible set:
`log_price`, `lag_log_qty`, `roll4_log_qty`, `woy_sin`, `woy_cos`, `is_q4`,
`prod_log_qty_mean` — 7 columns against 23. Stage 2 gets the full set.

---

## 6. Predictions for the benchmark, stated in advance

So they can be wrong.

1. The **constrained** classifier beats the unconstrained one on `oracle_rmse`, and by more
   at larger $K$ (more binding inequalities).
2. Both classifier variants have **`pct_negative` = 0** by construction (constrained) or
   materially above 0 (unconstrained).
3. The classifier's **average** elasticity is worse than Double ML's, because it is IPW and
   not orthogonal; its **`oracle_corr`** is competitive, because ranking is what the
   posterior is good at.
4. **T3** beats plain `RLearnerHeterogeneous`.
5. **T1** beats `PoissonGLM` on RMSE and ties it on elasticity — by construction, so this is
   a correctness check rather than a finding.
6. $\mathrm{VUS}_{\text{norm}}$ ranks the heterogeneous models about as well as
   `rloss_vs_constant` does, and both beat predictive RMSE.

---

## 7. Build plan

### 7.1 `elasticity_lab/pricelevels.py` (new)

```python
PriceLevels(K=3, scheme="quantile")      # cut r into K ordered levels; stores bar_r
    .fit(df) / .transform(df) -> level array, and level_prices()

PriceLevelClassifier(ElasticityModel)    # the unconstrained model  [name: "pricelevel_clf"]
    params: K, constraint="none", clip=0.02, weight_scheme="uniform", + LightGBM params
    fit():   A = multiclass LGBM  -> pi_hat        (unweighted)
             B = multiclass LGBM  -> b_hat         (sample_weight = qty)
    elasticity(df): steps C and D of §2.3
    posterior(df):  b_hat, for VUS scoring
    demand_curve(df): log s_hat, K columns
    arc_elasticities(df): K-1 columns

MonotonePriceLevelClassifier(PriceLevelClassifier)   # [name: "pricelevel_clf_mono"]
    constraint="softplus"  -> C1, L-BFGS on the offset multinomial likelihood
    constraint="isotonic"  -> C2, PAVA repair of any base classifier

vus_report(model, df, levels) -> dict
    raw VUS / HUM, chance, ceiling_from_elasticity, normalized_vus,
    check_monotone_demand, overlap diagnostics
```

Reused from `roc3` rather than reimplemented: `roc_surface`, `vus_forced_choice`, `hum`,
`constrained.vus_ceiling`, `constrained.normalized_vus`, `pricetest.ceiling_from_elasticity`,
`pricetest.posterior_prevalences`, `pricetest.check_monotone_demand`, `pricetest.audit`.

### 7.2 `elasticity_lab/twostage.py` (new)

`GLMOffsetBoost`, `GLMResidualBoost`, `AnchoredRLearner`, plus `SIMPLE_FEATURES`.

### 7.3 Edits to existing modules

* `models.py` — register the five new models; extend `MODEL_REGISTRY`.
* `tuning.py` — a search space per new model. The classifier's `K` and `clip` are
  hyperparameters; so is `constraint`, so the search can *choose* whether to constrain.
* `evaluation.py` — add `vus_metrics(...)` to the battery, guarded so it only fires for
  models that expose `posterior()`.

### 7.4 Experiments

| script | what it settles |
|---|---|
| `16_pricelevel_models.py` | the new models on both frames, all $K \in \{3,4,5\}$; predictions 1–3 |
| `17_vus_elasticity_link.py` | **P1–P5**; the heterogeneity sweep; the gauge test |
| `18_twostage.py` | predictions 4–5 |
| `19_full_zoo.py` | all twelve models, one table, tuned on the R-loss |

### 7.5 Documents

* `MODELS.md` — a new §3.9–§3.13 for the five new models, same format.
* `ELASTICITY_MODELS.md` — a new §6 for the VUS link.
* `IDEAS_LOG.md` — section K, including whatever refutes the predictions above.
* `notebooks/05_pricetest_classifiers_and_vus.ipynb` — the tutorial.

### 7.6 Order of work

1. `PriceLevels` + `PriceLevelClassifier` (unconstrained) + the gauge test **P4** — P4 is a
   theorem, so it is the fastest way to know the plumbing is right.
2. The constrained variants (C1, C2).
3. `17_vus_elasticity_link.py` — P1, P2, P3, P5.
4. `twostage.py` + `18_twostage.py`.
5. Search spaces, full-zoo run, docs, notebook.

---

## 8. Risks, and what would sink each piece

| risk | why it is plausible | mitigation | if it happens anyway |
|---|---|---|---|
| **IPW instability** — $\hat b_k/\hat\pi_k$ explodes where a level is rare for some $x$ | the seller's pricing is far from random; some products never discount | clip $\hat\pi$ at `clip`; report the overlap diagnostic $\min_k \hat\pi_k$ distribution | report the trimmed-sample estimate and say what fraction was trimmed |
| **Discretisation kills the signal** — within-level price variation is most of the information | 80% of products have sd(log price) > 0.10, much of it within a level | try $K = 3,4,5$; report the trade-off | a negative result worth documenting: discretisation is not free |
| **C1 does not converge** — L-BFGS on a softplus-composed multinomial likelihood | non-convex in $w$ | analytic gradient, multiple restarts, fall back to C2 | ship C2 only, and say why |
| **P3 comes out flat** — VUS_norm correlates with nothing | too few models, or ceiling estimation noisy | vary hyperparameters to get ≥ 30 points, not just 7 models | report it; the theory in §4.1–4.2 stands regardless of whether the *empirical* correlation is detectable at this sample size |
| **The ceiling is not computable** — `ceiling_from_elasticity` assumes 3 arms with given multipliers | our levels are quantiles, not fixed multipliers | derive the multipliers from $\bar r_k$; for $K > 3$ fall back to $1/K!$ and report raw HUM | restrict the ceiling analysis to $K = 3$ |
| **Everything is slower than budgeted** | two extra LightGBM multiclass fits per model, times a search | classifiers are cheap relative to DML; cap $K \le 5$ | reduce the full-zoo trial count and say so |

---

## 9. What is deliberately *not* being built

* **A doubly-robust price-level estimator.** The right next step (it would make the
  classifier orthogonal and fix the §2.4(iv) weakness), but it is a separate piece of theory
  and would double this batch. Recorded in §10.
* **Cross-price effects / substitution.** Every model here treats products as independent.
* **C3, the penalised constraint.** Subsumed by C1.
* **A negative-binomial member.** Overdispersion is real and the Poisson standard errors are
  too small, but it does not bear on the four questions asked.

---

## 10. Ideas considered and rejected, with reasons

**Predict the level from a purchase-only sample, without the propensity.** This is the
literal transposition of the price test and it is *wrong observationally*: it estimates
$b_k$, which mixes demand with the seller's pricing policy. It is included as an ablation
(`constraint="none", clip=None, use_propensity=False`) precisely to measure how wrong —
under randomisation the two coincide, so the gap is a direct measure of the confounding.

**Ordinal / cumulative-link (proportional odds) parameterisation** instead of C1. It
constrains the *cumulative* distribution to shift monotonically with a latent index, which
is a statement about stochastic ordering across $x$ — a different and stronger assumption
than the one we have (ordering *within* the pmf at fixed $x$). Rejected as answering the
wrong question.

**Conditional logit / BLP-style demand estimation.** The natural home for this problem in
industrial organisation, and much more powerful for substitution. Rejected for this batch:
it needs a market definition and a choice set, neither of which the product-week panel
supplies, and it would not answer any of the four questions.

**Using VUS as the *tuning* objective.** Tempting, and wrong for exactly the reason §4.1
establishes: VUS is blind to the elasticity level, so a search maximising it would optimise
half the problem while leaving the other half free. Worth *measuring* per trial (so it can
be correlated with the oracle), never worth *maximising*.

**Per-product price levels** rather than global quantiles of $r$. More faithful to "this
product's own discount ladder", but it makes $\bar r_k$ product-specific, so the levels are
not comparable across products and the classifier has no shared meaning for its labels.
Rejected; the relative price $r$ already normalises per product, which was the point of
defining it that way.

---

## 11. References added by this plan

* Manski, C., Lerman, S. (1977). The estimation of choice probabilities from choice-based
  samples. *Econometrica* 45(8).
* Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*. — prior correction.
* Saerens, M., Latinne, P., Decaestecker, C. (2002). Adjusting the outputs of a classifier to
  new a priori probabilities. *Neural Computation* 14(1).
* Hirano, K., Imbens, G. (2004). The propensity score with continuous treatments. — the
  generalised propensity score behind §2.3.
* Wüthrich, M., Merz, M. (2019). Combined actuarial neural network (CANN). — §5.1.
* Yan, J. et al. (2009). Applications of the offset in property-casualty predictive
  modeling. *CAS Forum*. — the offset/residual equivalence.
* Barlow, R., Bartholomew, D., Bremner, J., Brunk, H. (1972). *Statistical Inference under
  Order Restrictions*. — PAVA, for C2.
* Mossman, D. (1999); Scurfield, B. (1996); Nakas, C., Yiannoutsos, C. (2004) — VUS/HUM,
  already in [`REFERENCES.md`](REFERENCES.md).
