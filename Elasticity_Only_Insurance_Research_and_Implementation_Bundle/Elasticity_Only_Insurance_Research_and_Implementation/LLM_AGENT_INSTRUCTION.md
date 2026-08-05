# LLM agent instruction: real-data-only insurance price-elasticity modeling

## 0. Role, mission, and evidence standard

You are an applied machine-learning researcher and production Python engineer. Your task is to use a **real randomized motor-insurance price test** to learn where the insurer can safely raise or lower price. Focus on **relative demand, finite-difference price elasticity, and pricing decisions among the tested arms**. Do not make base-conversion prediction the main task.

The experiment has three arms:

- `H`: approximately +10% price;
- `C`: control price;
- `L`: approximately -10% price.

The nominal assignment may be 10/80/10, but always use the **logged row-level assignment probabilities** `pi_H`, `pi_C`, and `pi_L`.

Your evidence standard is strict:

1. Every empirical result must come from the supplied real randomized quote table.
2. Every reported candidate-model score must be strictly out of fold or on a final untouched chronological holdout.
3. Synthetic observations may be used only for deterministic unit tests of code or derivatives. Never use generated data to select, rank, or recommend a model for the insurer.
4. Never report individual elasticity RMSE, individual treatment-effect correlation, or “ground-truth” customer elasticity on real data; these targets are unobserved.
5. Preserve positive and negative findings, failed models, warnings, and dead ends in the research log.
6. If the experiment table or required fields are missing, stop and produce a missing-data report. Do not invent data and do not substitute a public observational dataset.

## 1. Core estimand

For pre-treatment customer and vehicle features `X=x`, define

- `T in {H,C,L}` as randomized price arm;
- `Y in {0,1}` as purchase;
- `pi_t(x) = P(T=t | X=x)` as the known randomized propensity;
- `q_t(x) = P(Y=1 | T=t, X=x)` as arm-specific quote conversion;
- `r_t(x) = P(T=t | Y=1, X=x)` as the buyer-side arm posterior.

Use the exact Bayes identity

```text
r_t(x) = pi_t(x) q_t(x) / sum_j pi_j(x) q_j(x).
```

The relative demand to control is identified by

```text
RR_tC(x) = q_t(x) / q_C(x)
          = [r_t(x) / pi_t(x)] / [r_C(x) / pi_C(x)].
```

For a fixed 10/80/10 assignment:

```text
RR_HC(x) = 8 r_H(x) / r_C(x)
RR_LC(x) = 8 r_L(x) / r_C(x).
```

Demand monotonicity means

```text
RR_HC(x) <= 1 <= RR_LC(x),
```

or equivalently

```text
r_H(x)/pi_H(x) <= r_C(x)/pi_C(x) <= r_L(x)/pi_L(x).
```

The adjacent finite-difference elasticity magnitudes are

```text
beta_H(x) = -log RR_HC(x) / log(1.10)
beta_L(x) = -log RR_LC(x) / log(0.90).
```

Because `log(0.90)<0`, both magnitudes are positive under decreasing demand.

Do not silently call these derivatives at a point. They are finite-difference elasticities over the tested intervals. The scalar model below adds a local constant-elasticity assumption.

## 2. Why a base-conversion model is not required for the price choice

Let `g_t(x)` be expected contribution if the policy is purchased at arm `t`. It must include the business-relevant arm-specific premium, expected claims, commissions, taxes, payment expenses, and other variable costs.

Expected contribution per quote is

```text
V_t(x) = g_t(x) q_t(x)
       = q_C(x) g_t(x) RR_tC(x).
```

The positive common factor `q_C(x)` cancels when choosing among arms:

```text
d*(x) = argmax_t g_t(x) RR_tC(x),  with RR_CC(x)=1.
```

Therefore, the primary model can be elasticity-only. Report the relative contribution factor

```text
B_t(x) = [g_t(x)/g_C(x)] RR_tC(x)
```

and relative lift `B_t(x)-1`.

A base-conversion model is optional and secondary. It may be useful for variance reduction in doubly robust policy evaluation, total-volume forecasting, or covariate-shift weighting. It must not dominate model training or model selection.

## 3. Required inputs and immediate stop conditions

Read `DATA_CONTRACT.md`. At minimum require one row per eligible randomized quote with:

- unique `quote_id`;
- stable `customer_id` or household cluster;
- `quote_timestamp`;
- `eligible`;
- `arm`;
- `purchase`;
- `pi_H`, `pi_C`, `pi_L`;
- explicit pre-treatment feature list.

For business policy evaluation also require `contribution_H`, `contribution_C`, and `contribution_L`.

Stop model fitting and issue an error report if any of the following occurs:

- assignment propensities are missing, nonpositive, or do not sum to one;
- an arm has no eligible quotes or no buyers;
- assignment was not randomized or the randomization mechanism cannot be reconstructed;
- eligibility or rows were removed after seeing arm or outcome;
- feature timing is unknown;
- customer grouping is absent despite repeated quotes;
- the final outcome window is undefined;
- the experiment implementation changed without a corresponding block, propensity, or version field.

## 4. Reproducible project initialization

Create a run directory containing:

```text
run_<UTC timestamp>_<git or code hash>/
  config_resolved.yaml
  environment.txt
  data_fingerprint.json
  audit/
  folds/
  optuna/
  models/
  oof_predictions/
  holdout_predictions/
  metrics/
  plots/
  research_log.md
  model_cards/
```

Record:

- SHA-256 hash of the exact input file bytes;
- row count and column names;
- package versions;
- random seeds;
- code commit or source hash;
- resolved configuration;
- every data exclusion and its reason.

Never overwrite a completed run directory.

## 5. Phase A: audit the randomized experiment before machine learning

### 5.1 Population accounting

Produce a CONSORT-like flow table:

1. all records received;
2. eligible randomized quotes;
3. outcomes observed within the defined window;
4. buyers and nonbuyers by arm;
5. repeated customers and repeated quote opportunities;
6. rows excluded, with pre-specified reasons.

Report arm-level quote and buyer counts. Use side-arm buyer counts, not total quote count, when judging heterogeneity capacity.

### 5.2 Assignment audit

For each arm, compare observed assignment count with the sum of logged propensities. Report standardized residuals and inspect them by:

- calendar week or day;
- randomization block;
- tariff version;
- channel;
- geography;
- device or acquisition source;
- base-premium band.

Unexpected residual patterns are implementation issues, not modeling opportunities.

### 5.3 Covariate balance

On all eligible quotes, compare pre-treatment covariates across H/C and L/C. Report standardized mean differences for numeric variables and distribution differences for categorical variables. Large systematic imbalance indicates an assignment or logging problem. Do not “fix” genuine randomization imbalance by propensity modeling unless the logged design probabilities are wrong; use the known randomized propensities.

### 5.4 Price-display compliance

Compare displayed premium divided by base premium with the assigned multiplier. Quantify rounding, minimum premiums, fees, installments, manual overrides, and failed displays. The primary causal analysis is intention-to-treat by randomized arm. A realized-dose analysis is secondary and should be treated as an instrumental-variable or compliance problem, not as a replacement for the randomized assignment.

### 5.5 Outcome integrity

Verify that purchase is measured identically across arms and that the window is not truncated by time. Perform pre-specified sensitivity analyses for early cancellation or payment failure only if those outcomes are consistently available.

### 5.6 Leakage audit

The model feature list must be explicit. Reject:

- arm and assigned multiplier;
- displayed experimental price;
- any variable computed after assignment using the displayed price;
- purchase, policy ID, payment choice, cancellation, or claims observed after purchase;
- agent or customer actions that happened after seeing price.

Pre-randomization base premium, expected cost, tariff version, uncertainty, and competitor position may be used.

## 6. Phase B: define folds once and reuse them for every model

### 6.1 Final chronological holdout

Reserve the most recent 15–25% of customer or household clusters as a final holdout. Assign a cluster to the holdout using its latest eligible quote time, so no cluster appears in both development and holdout data. Do not inspect holdout model rankings during Optuna tuning.

### 6.2 Nested development cross-validation

Within the development period:

- use 5 outer group-disjoint folds for model comparison;
- use 4 inner group-disjoint folds for Optuna;
- stratify buyer folds by arm using `StratifiedGroupKFold` when feasible;
- verify every validation fold contains enough H/C and L/C buyers;
- use the exact same outer folds for every candidate.

Also run 2–4 rolling forward-time backtests. A model that wins random group CV but fails forward-time validation is not production eligible.

### 6.3 Calibration splits

Calibration must use predictions from observations that were not used to train the base model. For each outer fold:

1. generate inner out-of-fold raw scores on the outer-training set;
2. fit the risk-ratio calibrator on those inner OOF scores;
3. refit the base learner on all outer-training buyers;
4. apply the learned calibrator to the outer-validation raw scores.

Never fit a probability or elasticity calibrator on in-sample predictions.

## 7. Feature preparation

Use only pre-treatment features. Preserve business meaning and prefer stable features over opaque identifiers.

### 7.1 Candidate feature groups

Consider:

- base premium and log base premium;
- expected claim cost and intended margin;
- tariff uncertainty, model version, and out-of-distribution score;
- customer, driver, vehicle, geography, and coverage attributes;
- channel and acquisition context;
- new business versus renewal;
- pre-treatment competitor gap or market rank;
- quote-time seasonality and market conditions;
- historical customer behavior known before this quote.

Do not automatically include every available field.

### 7.2 Fold-local preprocessing

Fit imputers, category maps, encoders, scalers, and feature-selection steps inside each training fold only.

- For linear models: median-impute numeric features, add missing indicators, standardize, and one-hot encode categories.
- For Random Forest: impute numeric variables and use fold-local one-hot or ordinal encoding. Avoid target encoding based on randomized arm.
- For LightGBM: native categorical columns are acceptable if category levels are fixed from the training fold. One-hot encoding is a valid portability baseline.
- For XGBoost: use one-hot encoding by default for portability. Native categorical handling may be used only with version-controlled serialization and fold-local category handling.

Combine rare categorical levels using a minimum frequency chosen before or inside inner CV. Do not use buyer arm as a target encoder because randomization noise will be amplified.

## 8. Mandatory model baselines

Fit these before high-capacity models.

### 8.1 No-effect model

Use

```text
r_t(x)=pi_t(x), RR_HC=RR_LC=1.
```

This is the null benchmark for buyer log loss and information gain.

### 8.2 Global unrestricted two-gap model

Fit constant `log RR_HC` and `log RR_LC` by the pairwise propensity-offset likelihoods. Fit both unconstrained and monotone-constrained versions. This estimates the portfolio-level finite-difference response without heterogeneity.

### 8.3 Global scalar elasticity model

Fit a single nonnegative `beta`:

```text
r_t(x) proportional to pi_t(x) exp[-beta z_t],
z_t = log(P_t/P_C).
```

This is the most important anchor. A flexible model must demonstrate real OOF improvement over it. If it does not, deploy the global model or do not personalize.

### 8.4 Penalized pairwise logistic model

Fit H/C and L/C separately using inverse-propensity buyer weights or a fixed offset implementation. Include main effects and a small, pre-specified interaction set. This is the interpretable heterogeneous benchmark.

## 9. Main model family: pairwise buyer likelihoods

For contrast `t` versus control among buyers:

```text
P(T=t | T in {t,C}, Y=1, X=x)
  = pi_t(x) q_t(x) / [pi_t(x)q_t(x)+pi_C(x)q_C(x)].
```

Therefore

```text
logit P(T=t | pair, buyer, X=x)
  = log[pi_t(x)/pi_C(x)] + log RR_tC(x).
```

Fit two models:

- H versus C, estimating `log RR_HC(x)`;
- L versus C, estimating `log RR_LC(x)`.

Reconstruct a coherent three-arm buyer posterior:

```text
r(x) = softmax(log pi(x) + [log RR_HC(x), 0, log RR_LC(x)]).
```

This reconstruction is mandatory. Do not treat two one-versus-rest probabilities as independent final probabilities.

### 9.1 XGBoost pairwise model

Use native binary logistic loss with the known row-specific propensity log odds as `base_margin`. The booster learns the residual `log RR_tC(x)`. This is preferable to class balancing because it retains the exact conditional likelihood without increasing variance through inverse-propensity reweighting.

Set a global pairwise log-RR MLE as the initial residual intercept, then let trees learn heterogeneous deviations. Use `tree_method='hist'`, early stopping, shallow trees, large child weights, and strong regularization.

Do not set `scale_pos_weight` to buyer-class frequency. Keep it equal to one; the unequal buyer-arm frequency is part of the estimand.

### 9.2 LightGBM pairwise model

Use `init_score = log(pi_t/pi_C) + global_log_RR_tC` on the training and validation pairwise buyer rows. Set `boost_from_average=False`. The learned tree score is a heterogeneous deviation from the global log risk ratio. At prediction, add the stored global log RR to the tree contribution; the row-specific propensity offset is used only when converting the risk ratio into the buyer pair probability.

Use conservative `num_leaves`, `max_depth`, `min_data_in_leaf`, and `min_sum_hessian_in_leaf`. Leaf-wise growth can overfit weak interaction signal unless these controls are strong.

### 9.3 Random-Forest pairwise model

scikit-learn Random Forest has no fixed logit-offset API. On pairwise buyers, weight every row by

```text
w_i = 1 / pi_{T_i}(X_i).
```

In the weighted buyer population,

```text
P_w(T=t | pair, buyer, X=x) = q_t(x) / [q_t(x)+q_C(x)],
```

so the predicted logit is `log RR_tC(x)`.

Use `class_weight=None`. Automatic balanced class weights would change the target again. Fit large, heavily smoothed forests and calibrate the log-risk-ratio score out of fold.

Random Forest is a valuable robustness benchmark and nonlinear challenger, but its raw class probabilities are often too coarse for elasticity ratios. Calibration and large leaves are mandatory.

### 9.4 Penalized scikit-learn logistic model

Use the same inverse-propensity weighting as Random Forest. Use `saga` for L2 or elastic-net regularization. Scale numeric inputs. Do not use automatic balanced class weights. Use this model to test whether complex trees add genuine heterogeneity beyond stable additive effects.

## 10. Scalar elasticity family

Assume local constant elasticity over the tested range:

```text
log RR_tC(x) = -beta(x) [z_t-z_C], beta(x)>=0.
```

The buyer NLL for row `i` is

```text
L_i = beta(x_i) z_{T_i}
      + log sum_j pi_ij exp[-beta(x_i) z_ij].
```

Its derivative with respect to beta is

```text
z_{T_i} - E_r[z | x_i],
```

and its curvature in beta is

```text
Var_r(z | x_i) >= 0.
```

### 10.1 XGBoost and LightGBM custom objectives

Let the raw tree score be `f(x)` and set

```text
beta(x) = softplus(f(x)).
```

Use gradient

```text
g_f = [z_T - E_r(z)] sigmoid(f).
```

Use a positive Fisher/Gauss-Newton Hessian

```text
h_f = Var_r(z) sigmoid(f)^2 + epsilon.
```

Do not rely on the exact raw-score Hessian because the nonlinear positivity transform can make it negative. Custom tree objectives must be row-additive; the scalar model satisfies that requirement.

Because the tested log-price doses are only about +/-0.1, Fisher curvature per buyer is small. Use much smaller `min_child_weight` / `min_sum_hessian_in_leaf` ranges than ordinary binary logistic boosting, but compensate with shallow trees, large data leaves, and strong L2 regularization.

### 10.2 Random-Forest moment model

For a Random Forest scalar benchmark, inverse-propensity-weight buyers and regress the observed centered log-price dose on `X`. Under the scalar exponential-family model, the weighted conditional mean dose is a one-to-one decreasing function of `beta`. Invert that function by vectorized bisection.

Use squared-error regression only; absolute-error regression targets a median and is not valid for the mean-moment inversion. Treat this as a robust benchmark, not as a likelihood-equivalent estimator.

### 10.3 When to prefer scalar over two-gap models

Prefer scalar when:

- side-arm buyer counts are limited;
- +/-10% is a local experiment;
- H/C and L/C effects are statistically compatible;
- flexible two-gap models are unstable across folds or time;
- the scalar model has equal or better OOF likelihood.

Permit separate H/C and L/C surfaces only after they improve held-out likelihood and calibration. Apparent upper/lower asymmetry can arise from population mixing, noise, or different side-arm precision, not necessarily true individual curvature.

## 11. Multiclass models: benchmark, not automatic default

Train an inverse-propensity-weighted multiclass buyer classifier to estimate

```text
s_t(x) = q_t(x) / sum_j q_j(x).
```

Convert to risk ratios by `s_t/s_C`. Test multinomial logistic regression, Random Forest, LightGBM, and XGBoost.

Use multiclass models as benchmarks because:

- the control class dominates raw buyer frequency;
- the model spends capacity on a redundant common normalization;
- monotonicity is not automatic;
- pairwise models allow contrast-specific regularization and diagnostics.

If a multiclass model wins, project its log density scores onto the ordered cone, recalibrate, and quantify the fraction and magnitude of changed predictions.

## 12. Calibration, shrinkage, and constraints

Raw tree probabilities are not sufficient. Elasticity depends on odds and probability ratios, so calibration errors are amplified.

For each contrast, fit on inner OOF predictions:

```text
log RR_cal(x) = alpha + s [raw(x)-center],  s>=0.
```

Estimate `alpha` and `s` in the exact pairwise buyer likelihood with the known propensity offset. Interpret:

- `s=0`: no reproducible heterogeneity; collapse to a global effect;
- `0<s<1`: raw model is over-dispersed and should be shrunk;
- `s≈1`: dispersion is approximately calibrated;
- `s>1`: raw model is under-dispersed.

Enforce physical monotonicity after calibration:

```text
log RR_HC <= 0
log RR_LC >= 0.
```

Hard clipping is transparent. A smooth softplus projection may be used during optimization. Always report:

- raw violation rate;
- average and maximum projection distance;
- percentage of predictions exactly on the boundary;
- score before and after projection.

A high projection rate is evidence of a weak or misspecified model, not proof that post-processing solved it.

Use isotonic calibration only when the calibration sample is sufficiently large and each contrast has ample side-arm buyers. Prefer affine/sigmoid calibration at smaller sample sizes.

## 13. Optuna protocol

### 13.1 General design

Create a separate Optuna study for each:

- contrast H/C and model family;
- contrast L/C and model family;
- scalar model family;
- multiclass benchmark family.

Do not place every family in one giant conditional search space. Separate studies give TPE enough comparable trials and preserve clear failure logs.

Use:

```python
TPESampler(
    seed=SEED,
    n_startup_trials=30,
    multivariate=True,
    group=True,
)
MedianPruner(
    n_startup_trials=15,
    n_warmup_steps=1,
    n_min_trials=5,
)
```

Persist studies in SQLite or another RDB. Use `n_jobs=1` at the Optuna level and allocate threads inside the learner, or run distributed workers with each learner constrained to a known thread count. Avoid nested uncontrolled parallelism.

For each trial:

1. loop over the fixed inner group folds;
2. fit all preprocessing inside the fold;
3. use native early stopping for boosting;
4. evaluate pairwise conditional NLL or scalar buyer NLL;
5. call `trial.report(mean_completed_fold_score, fold_number)`;
6. prune only after at least one complete validation fold;
7. save all fold scores, best iteration counts, and warnings as trial attributes.

Do not optimize AUC, VUS, accuracy, F1, in-sample loss, or estimated policy value.

### 13.2 Trial budgets

Use these as starting budgets, not guarantees:

| Family | Trials per contrast/model |
|---|---:|
| Penalized logistic | 40–80 |
| Random Forest pairwise | 80–150 |
| LightGBM pairwise | 150–250 |
| XGBoost pairwise | 150–250 |
| Random-Forest moment scalar | 60–120 |
| LightGBM scalar custom | 100–200 |
| XGBoost scalar custom | 100–200 |
| Multiclass benchmark | 80–150 |

After choosing hyperparameters, refit the top configurations over 3–5 fixed random seeds. Do not tune the random seed.

### 13.3 Dynamic minimum leaf rule

Let `n_pair` be the number of buyers in a contrast and `rho` the side-arm fraction within that pair. Set the minimum search bound to contain approximately 10 minority-arm buyers per leaf:

```text
leaf_lower = max(20, ceil(10/rho)).
```

Set the upper bound to the smaller of 25% of pair buyers and about 100 minority-arm buyers:

```text
leaf_upper = max(leaf_lower,
                 min(floor(0.25*n_pair), ceil(100/rho))).
```

If this leaves room for only a few leaves, that is correct evidence that rich heterogeneity is not supported.

### 13.4 LightGBM pairwise search space

Optimize:

| Parameter | Range | Notes |
|---|---|---|
| `learning_rate` | 0.005–0.10 log | Use up to 5,000 rounds with early stopping. |
| `max_depth` | 2–7 | Weak signal favors shallow trees. |
| `num_leaves` | 4 to min(64, 2^depth−1) | Respect depth. |
| `min_data_in_leaf` | dynamic log-int | Highest-impact variance control. |
| `min_sum_hessian_in_leaf` | 0.1–50 log | Important for pairwise logistic curvature. |
| `feature_fraction` | 0.5–1.0 | Feature subsampling. |
| `bagging_fraction` | 0.6–1.0 | Row subsampling. |
| `bagging_freq` | 1–10 | Use when fraction <1. |
| `lambda_l1` | 1e-8–100 log | Include zero as separate categorical branch if desired. |
| `lambda_l2` | 1e-3–100 log | Usually important. |
| `min_gain_to_split` | 0 or 1e-5–1 log | Conditional zero/positive branch. |
| `max_bin` | 63, 127, 255, 511 | Lower values regularize and speed training. |

Fix `boosting_type='gbdt'`, `boost_from_average=False`, and `feature_pre_filter=False` during tuning. Test DART only as a separate late-stage study because its early-stopping and calibration behavior differ.

### 13.5 XGBoost pairwise search space

Optimize:

| Parameter | Range | Notes |
|---|---|---|
| `eta` | 0.005–0.10 log | Up to 5,000 rounds with early stopping. |
| `grow_policy` | `depthwise`, `lossguide` | Conditional space. |
| `max_depth` | 2–7 | For depthwise. |
| `max_leaves` | 8–64 log-int | For lossguide. |
| `min_child_weight` | 1–100 log | Main smoothing control. |
| `subsample` | 0.6–1.0 | Row subsampling. |
| `colsample_bytree` | 0.5–1.0 | Feature subsampling. |
| `gamma` | 0 or 1e-6–10 log | Split-gain threshold. |
| `reg_alpha` | 1e-8–100 log | L1. |
| `reg_lambda` | 1e-3–100 log | L2. |
| `max_bin` | 64, 128, 256, 512 | Histogram resolution. |
| `max_delta_step` | 0, 1, 2, 5 | Can stabilize imbalanced logistic updates. |

Fix `tree_method='hist'` and `scale_pos_weight=1`.

### 13.6 Random-Forest pairwise search space

Set `n_estimators=1000` initially. Verify that OOB or held-out score has stabilized; then do not waste Optuna trials tuning tree count.

Optimize:

| Parameter | Range | Notes |
|---|---|---|
| `criterion` | `log_loss`, `gini` | Log-loss is natural, but test both. |
| `max_depth` | 4–16 | Large leaves remain the main guardrail. |
| `min_samples_leaf` | dynamic log-int | Most important. |
| `max_features` | `sqrt`, `log2`, or 0.2–1.0 | Conditional branch. |
| `max_samples` | 0.5–1.0 | Requires bootstrap. |
| `min_impurity_decrease` | 0 or 1e-8–1e-3 log | Additional pruning. |
| `ccp_alpha` | 0 or 1e-8–1e-2 log | Cost-complexity pruning. |

Fix `bootstrap=True`, `class_weight=None`, and use inverse-propensity `sample_weight`.

### 13.7 Penalized logistic search space

Optimize:

- `C`: 1e-4–100 log;
- L2 versus elastic net;
- `l1_ratio`: 0–1 for elastic net;
- optional interaction set as a small categorical structural choice.

Use `saga`, `max_iter>=5000`, scaled numeric features, and no class weights.

### 13.8 Scalar custom-objective search spaces

For XGBoost:

- `eta`: 0.005–0.05 log;
- `max_depth`: 1–4;
- `min_child_weight`: 1e-4–1 log;
- `subsample`: 0.6–1;
- `colsample_bytree`: 0.5–1;
- `gamma`: 1e-7–1 log;
- `reg_alpha`: 1e-8–100 log;
- `reg_lambda`: 1e-3–100 log;
- `max_bin`: 64, 128, 256.

For LightGBM:

- `learning_rate`: 0.005–0.05 log;
- `max_depth`: 1–4;
- `num_leaves`: 2–min(15,2^depth);
- `min_data_in_leaf`: 1%–20% of buyers, log-int, with a minimum of 30;
- `min_sum_hessian_in_leaf`: 1e-5–0.5 log;
- `feature_fraction`: 0.5–1;
- `bagging_fraction`: 0.6–1;
- `lambda_l1`: 1e-8–100 log;
- `lambda_l2`: 1e-3–100 log.

The small Hessian ranges are specific to the price-dose curvature. Do not copy ordinary binary-logistic defaults blindly.

### 13.9 Random-Forest moment search space

Fix `criterion='squared_error'` and approximately 1,000 trees. Optimize:

- `max_depth`: 3–12;
- `min_samples_leaf`: 0.01–0.15 of buyers, log;
- `max_features`: 0.3–1;
- `max_samples`: 0.6–1;
- `ccp_alpha`: 1e-9–1e-2 log;
- `beta_max`: 10, 20, 30 as a sensitivity bound.

## 14. Hyperparameters that matter most

Prioritize tuning effort in this order:

1. leaf/child sample and Hessian constraints;
2. tree depth or leaf count;
3. L2 regularization;
4. learning rate with early-stopped round count;
5. row and feature subsampling;
6. L1 and split-gain thresholds;
7. calibration slope and global shrinkage.

Do not spend major search budget on:

- random seed;
- `n_estimators` for Random Forest after convergence;
- cosmetic logging parameters;
- class weights that alter the target;
- extremely deep trees that the leaf constraints will prevent anyway.

## 15. Primary model-quality metrics

### 15.1 Buyer multiclass NLL

For coherent posterior `r_hat`:

```text
NLL_buyer = -mean_{buyers} log r_hat_{T_i}(X_i).
```

This is a proper score for the flip posterior.

### 15.2 Pairwise conditional NLL

For contrast `t/C`:

```text
eta_i = log(pi_it/pi_iC) + log RR_hat_tC(X_i)
NLL_tC = mean_{pair buyers} [log(1+exp eta_i) - y_i eta_i].
```

Use

```text
NLL_equal = 0.5 NLL_HC + 0.5 NLL_LC
```

as the main cross-family metric. It prevents the 80% control arm and unequal pair sizes from determining the result by themselves.

### 15.3 Elasticity information gain

Within each outer fold, compare a heterogeneous model with the fitted global pairwise model:

```text
IG_t = NLL_global,t - NLL_model,t.
```

Report millinats per pair buyer and a customer-clustered paired confidence interval. The heterogeneous model is not credible if the interval includes zero across development CV and the forward holdout.

### 15.4 Calibration

For each contrast, evaluate on held-out predictions:

```text
logit P(T=t | pair,buyer,X)
  = log(pi_t/pi_C) + a + b * predicted_log_RR.
```

Report calibration intercept and nonnegative slope. Also create 4–5 large groups by predicted score and fit an intercept-only log RR within each group using the propensity offset.

Do not create ten small deciles when side-arm buyer counts are low.

### 15.5 Ranking and targeting

Use top-fraction curves and, where implemented with valid inference, RATE/TOC-style metrics to test whether highly ranked quotes show larger randomized price response. Ranking metrics are supplementary: they do not replace probability calibration or pairwise NLL.

VUS, multiclass AUC, and ordinary accuracy are secondary diagnostics only. They do not validate the numerical odds ratios needed for elasticity.

### 15.6 Stability

Report:

- fold-to-fold metric standard deviation;
- seed-to-seed prediction correlation;
- Spearman correlation of elasticity scores;
- Jaccard overlap of top 5%, 10%, and 20% adjustment sets;
- calibration and information gain by time period;
- performance by channel and tariff version;
- extrapolation and missingness sensitivity.

## 16. Business policy evaluation without a conversion model

Construct the three-arm policy

```text
policy(x) = argmax_t contribution_t(x) * RR_hat_tC(x).
```

Also construct selective policies that adjust only the top 5%, 10%, 20%, 40%, 60%, or 100% of quotes by predicted positive relative contribution gain, leaving the rest at control.

On strictly held-out eligible quotes, estimate policy contribution with inverse propensity scoring:

```text
V_IPS(d) = (1/N) sum_i I[T_i=d(X_i)] Y_i g_{T_i}(X_i) / pi_{i,T_i}.
```

Also report the self-normalized/Hajek version. Compare:

- control for everyone;
- each uniform tested arm;
- global scalar policy;
- each calibrated flexible model;
- the constrained ensemble;
- selective adjustment policies.

Use customer-cluster bootstrap confidence intervals for paired policy differences. Choose business thresholds on inner folds and evaluate them on outer folds. The final chronological holdout is used once for confirmation.

An optional base-conversion model may support a doubly robust value estimator, but IPS remains the required model-free benchmark because the randomization propensities are known.

## 17. Ensembling

After generating OOF log-risk-ratio predictions from eligible candidates, fit a convex stack separately for H/C and L/C:

```text
log RR_stack = sum_k w_k log RR_k,
w_k>=0, sum_k w_k=1.
```

Optimize weights on OOF pairwise NLL, then recalibrate and impose sign constraints. A convex combination of constrained log RRs preserves the correct sign.

Do not stack in-sample predictions. Keep the global scalar model in the library so the stack can shrink toward it.

## 18. Interpretation and feature importance

Use held-out permutation importance with **pairwise conditional NLL** as the scoring rule. Do not rely on tree impurity importance; it is biased toward high-cardinality variables and can describe overfitting.

For the top stable features:

- plot partial dependence or accumulated local behavior for calibrated log RR, not arm probability;
- stratify by tariff version and time;
- check that the relationship is operationally plausible;
- inspect interaction stability across folds and seeds;
- record whether the feature is approved for pricing use.

Never interpret a split or SHAP contribution as a causal effect of that feature. Randomization identifies the price-arm effect conditional on features; it does not randomize customer characteristics.

## 19. Model-selection gates

Use a gated rather than single-number selection rule.

### Gate 1: experiment validity

Pass assignment, eligibility, outcome, leakage, and compliance audits.

### Gate 2: global signal

The global pairwise/scalar model must show a stable nonzero price response. If not, report insufficient evidence and do not fit highly personalized policies.

### Gate 3: proper-score improvement

A flexible model must improve equal-contrast NLL over the global anchor with a paired customer-clustered confidence interval and confirm in forward-time testing.

### Gate 4: calibration

The model must have acceptable calibration after OOF affine shrinkage, low projection burden, and sensible grouped response ordering.

### Gate 5: stability

Top adjustment sets and score rankings must be stable across folds, seeds, time, and major operational segments.

### Gate 6: business value

Among models passing the previous gates, choose the one with the highest **lower confidence bound** on held-out contribution lift relative to control or the best uniform tested arm. Prefer the simpler model when confidence intervals overlap materially.

No empirical winner may be declared until all gates are evaluated on real data.

## 20. Final refit and deployment artifacts

After final model choice:

1. freeze feature definitions, transformations, hyperparameters, calibrators, and constraints;
2. refit on the full development period only;
3. evaluate once on the final chronological holdout;
4. if accepted, refit on all approved data for deployment while preserving the frozen specification;
5. save model, preprocessor, calibration objects, contribution-policy logic, schema, package versions, and training-data hash;
6. build runtime checks for missing categories, out-of-range values, tariff version, and propensity/design changes;
7. produce an abstention flag when a quote is outside training support or model disagreement is high.

Monitor after deployment:

- randomized shadow-test buyer NLL;
- global and grouped risk ratios;
- adjustment frequency;
- policy IPS value;
- calibration drift;
- feature and tariff-version drift;
- constraint/projection rates.

Retain a continuing randomized control sample. A fully deterministic personalized policy removes the data needed to learn future elasticity.

## 21. Required outputs from the agent

Produce at least:

1. `data_validation_issues.csv`;
2. `experiment_summary.csv`;
3. assignment, balance, and dose-compliance audits;
4. immutable fold assignment file;
5. Optuna database and trial tables for every family;
6. OOF predictions for every candidate with `quote_id`, fold, raw and calibrated log RRs, elasticities, constraint flags, and policy arm;
7. final holdout predictions;
8. proper-loss, calibration, stability, and policy-value tables;
9. selective policy curves;
10. customer-cluster bootstrap intervals;
11. held-out permutation importance;
12. one model card per candidate;
13. a model-comparison decision log;
14. `research_log.md` listing positive findings, negative findings, failures, and unresolved risks;
15. reproducibility manifest and SHA-256 hashes.

## 22. Prohibited shortcuts and known dead ends

Do not:

- train only on buyers and then claim absolute conversion probability;
- balance buyer arms to one third without exact prior/weight correction;
- use `class_weight='balanced'` and interpret the resulting probabilities as risk ratios;
- oversample or SMOTE buyer arms as if this created experimental information;
- include displayed experimental price as a feature when arm is the target;
- tune on AUC, VUS, accuracy, or policy value alone;
- compare models on different folds;
- calibrate on training predictions;
- use row-random splits when customers repeat;
- average individual elasticities and call the result a marginal group elasticity without checking the aggregation;
- infer true individual elasticity from one observed arm;
- claim that a flexible tree is better because it has lower training loss;
- select a model from synthetic data;
- hide failed trials or constraint violations.

## 23. Recommended final candidate set

Unless the real data indicate otherwise, carry these candidates to outer validation:

1. no-effect null;
2. global unrestricted two-gap model;
3. global scalar elasticity model;
4. penalized pairwise logistic;
5. calibrated pairwise Random Forest;
6. calibrated pairwise LightGBM with propensity offsets;
7. calibrated pairwise XGBoost with propensity offsets;
8. Random-Forest scalar moment model;
9. LightGBM scalar custom objective;
10. XGBoost scalar custom objective;
11. inverse-propensity multiclass benchmarks;
12. convex OOF ensemble including the global anchor.

The expected practical front-runners are the global scalar anchor and calibrated pairwise LightGBM/XGBoost models. Random Forest is the principal robustness benchmark. This is a research prior, not an empirical conclusion; let the real nested OOF and chronological results decide.

## 24. Completion checklist

Before declaring the task complete, answer every item with evidence:

- [ ] Did the input pass the real randomized-data contract?
- [ ] Were all exclusions pre-treatment and documented?
- [ ] Are folds customer/household disjoint?
- [ ] Is there an untouched chronological holdout?
- [ ] Did every model use identical outer folds?
- [ ] Were preprocessing and calibration fold-local?
- [ ] Were logged row-level propensities used exactly?
- [ ] Were class balancing and oversampling avoided or exactly corrected?
- [ ] Were global scalar and global two-gap anchors fitted?
- [ ] Did Optuna optimize proper buyer likelihoods?
- [ ] Were native early stopping and pruning kept inside inner CV?
- [ ] Were probabilities calibrated out of fold?
- [ ] Were monotonicity violations and projection burden reported?
- [ ] Was heterogeneous improvement tested against the global model?
- [ ] Were forward-time results reported?
- [ ] Was the pricing policy evaluated by randomized IPS/SNIPS?
- [ ] Were uncertainty and cluster dependence handled?
- [ ] Were top-k adjustment sets stable?
- [ ] Were negative findings and dead ends retained?
- [ ] Were no synthetic model rankings used?
- [ ] Is the final recommendation conditional on real held-out evidence?

Use the functions and examples in the accompanying `elasticity_only` Python package. Keep all model-specific code behind a common interface that produces calibrated `log_rr_H`, `log_rr_L`, coherent buyer posteriors, finite-difference elasticities, and arm-specific relative contribution scores.
