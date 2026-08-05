# Elasticity-only modeling for randomized motor-insurance price tests

## Executive conclusion

The business objective is not primarily to predict whether a customer buys. It is to identify quote contexts in which a tested price increase or decrease produces the best relative contribution. That objective can be addressed without a customer-level base-conversion model.

For arm `t`, let `q_t(x)` be conversion and let `g_t(x)` be contribution if purchased. Relative to control,

```text
q_t(x)=q_C(x) RR_tC(x).
```

The expected contribution is therefore

```text
g_t(x) q_t(x)=q_C(x) g_t(x) RR_tC(x).
```

The common factor `q_C(x)` cancels from the price choice. Among the tested arms, the optimal quote-level action is

```text
argmax_t g_t(x) RR_tC(x).
```

This supports an elasticity-only research program. The randomized experiment still needs all eligible quotes for auditing and policy evaluation, but the primary shape estimator can be trained on buyers through the exact Bayes-flip conditional likelihood.

The most promising practical strategy is a **model ladder**, not a single assumed winner:

1. global scalar elasticity as the mandatory anchor;
2. penalized pairwise logistic as an interpretable heterogeneous baseline;
3. calibrated H/C and L/C LightGBM models with propensity offsets;
4. calibrated H/C and L/C XGBoost models with propensity offsets;
5. heavily smoothed pairwise Random Forest as a robustness benchmark;
6. scalar custom-objective LightGBM/XGBoost and Random-Forest moment models;
7. a constrained convex ensemble of real out-of-fold predictions.

The winner must be selected on real nested out-of-fold and chronological-holdout evidence. No synthetic model ranking is used or supplied.

## 1. Target, identification, and interpretation

### 1.1 Bayes-flip identity

Let the randomized arm be `T in {H,C,L}` and purchase be `Y`. The row-level randomization probabilities are `pi_t(x)`. Define

```text
q_t(x)=P(Y=1 | T=t, X=x)
r_t(x)=P(T=t | Y=1, X=x).
```

Bayes' rule gives

```text
r_t(x)=pi_t(x)q_t(x)/sum_j pi_j(x)q_j(x).
```

Consequently,

```text
RR_tC(x)=q_t(x)/q_C(x)
        =[r_t(x)/pi_t(x)]/[r_C(x)/pi_C(x)].
```

This is exact and does not rely on conversion being rare. The randomized-trial case-only literature gives a related relative-risk interpretation of treatment assignment among cases [1].

For fixed 10/80/10 assignment,

```text
RR_HC=8r_H/r_C
RR_LC=8r_L/r_C.
```

The economically natural monotonicity condition is

```text
RR_HC<=1<=RR_LC,
```

which is equivalent to

```text
r_H/pi_H <= r_C/pi_C <= r_L/pi_L.
```

### 1.2 What is and is not identified

The buyer-side posterior identifies conditional relative conversion risks. It does not identify absolute `q_C(x)` by itself. For the present business objective, that is not fatal because `q_C(x)` cancels from the arm choice.

The learned function should be described as a **conditional marginal finite-difference response**. If unobserved tariff competitiveness affects both baseline conversion and price sensitivity, the estimated response averages over that unobserved state. It is not the latent willingness-to-pay of a physically identifiable individual.

Mandatory insurance does not imply mandatory purchase from this insurer. The outcome is own-quote acceptance, including substitution to competitors, abandonment, delay, and channel switching.

### 1.3 Finite-difference elasticity

For nominal multipliers 1.10, 1.00, and 0.90,

```text
beta_H(x)=-log RR_HC(x)/log(1.10)
beta_L(x)=-log RR_LC(x)/log(0.90).
```

These are adjacent finite-difference magnitudes. A scalar model assumes

```text
log RR_tC(x)=-beta(x)[log P_t-log P_C], beta(x)>=0.
```

This is a useful regularizer for a three-price experiment, but it must be tested against separate H/C and L/C responses.

## 2. Why buyer-only learning is attractive and where it can fail

### 2.1 Advantage

A full binary conversion loss is dominated by baseline purchase prediction when conversion is around 5%. The buyer conditional likelihood removes the common conversion scale and places all optimization pressure on the randomized arm composition among purchasers. It also makes the known treatment propensity an offset rather than a relationship the learner must rediscover.

This is a robustness advantage under nuisance misspecification. It is not a claim that conditioning creates information. A correctly specified full likelihood remains at least as informative as its conditional component.

### 2.2 Buyer-distribution weighting

The conditional likelihood weights regions of feature space according to where buyers occur. This is often aligned with contribution: errors in regions with almost no sales matter less economically. If the scientific goal instead requires an elasticity surface representative of all quote opportunities, one could reweight buyers toward the quote distribution, but that requires a conversion or density-ratio model and can cause high variance. That is an optional sensitivity analysis, not the default.

### 2.3 Major failure modes

Buyer-only learning can fail when:

- propensity logging is wrong;
- post-randomization variables leak arm identity;
- one customer appears across folds;
- sample sizes in side-arm buyers are too small for heterogeneity;
- price response changes with unmeasured competitive position;
- the model is selected by ranking metrics without probability calibration;
- class balancing changes the buyer-arm posterior;
- tree models fit random arm imbalances in high-dimensional feature space.

These risks justify strong pooling, calibration, forward validation, and explicit model-selection gates.

## 3. Recommended model hierarchy

### 3.1 Global models

Fit three global references:

1. no effect: `RR_HC=RR_LC=1`;
2. unrestricted global two-gap model;
3. global scalar `beta>=0`.

The flexible model must improve over the global model out of fold. A statistically detectable average effect does not imply learnable heterogeneity.

### 3.2 Pairwise conditional models

For `t` equal to H or L, among buyers in `{t,C}`:

```text
logit P(T=t | T in {t,C},Y=1,X)
 = log(pi_t/pi_C)+log RR_tC(X).
```

This suggests two binary models. They are coherent after reconstructing

```text
r=softmax(log pi+[log RR_HC,0,log RR_LC]).
```

Pairwise models have practical advantages:

- they use standard binary learners;
- each contrast can have different regularization;
- calibration is easy to inspect;
- upper and lower effects can differ;
- equal-contrast scoring prevents the control arm from hiding side-arm errors.

### 3.3 LightGBM and XGBoost with offsets

For XGBoost, use the propensity log odds as `base_margin`; for LightGBM use `init_score`. Both learners then estimate residual log risk ratio. XGBoost documents custom objectives as requiring row-additive, smooth losses and discusses Hessian constraints and Fisher approximations [12–14]. LightGBM exposes `init_score` in its Dataset API and emphasizes the interaction of leaves, depth, and minimum leaf size in tuning [10–11].

For pairwise models, native binary logistic loss plus an offset is preferable to a custom objective. It is simpler, exact, and uses library-tested derivatives.

### 3.4 Random Forest

Random Forest has no fixed logit offset in the scikit-learn API. Inverse-propensity weighting among pairwise buyers yields a target whose logit is the risk ratio. The forest should use large leaves, shallow-to-moderate depth, `class_weight=None`, and out-of-fold calibration. scikit-learn allows `log_loss`, `min_samples_leaf`, feature subsampling, bootstrap subsampling, and cost-complexity pruning [8].

Random Forest is unlikely to be the most statistically efficient model with few side-arm buyers, but it is useful because its inductive bias differs from boosting. Agreement between a smoothed forest and boosted models is stronger evidence than a single tree family winning by a tiny score margin.

### 3.5 Scalar custom objective

The scalar buyer likelihood is convex in `beta` for fixed features. With `beta=softplus(f)`, use the gradient

```text
[z_T-E_r(z)] sigmoid(f)
```

and positive Fisher curvature

```text
Var_r(z) sigmoid(f)^2+epsilon.
```

The raw-score exact Hessian can be negative because of the positivity transform, so the Fisher Hessian is the stable tree-boosting choice. Keep trees very shallow and use strong shrinkage.

### 3.6 Random-Forest moment inversion

Inverse-propensity-weighted buyer arm dose has a conditional mean that is monotone in scalar `beta`. A RandomForestRegressor can estimate that mean and invert it by bisection. This gives Random Forest a one-dimensional elasticity target without inventing an individual label. It is less efficient than maximum likelihood but valuable as a structurally different benchmark.

### 3.7 Multiclass and ensemble models

An inverse-propensity multiclass classifier estimates normalized arm risks. It is a benchmark rather than the first choice because it does not directly exploit the two contrasts or monotonicity.

After OOF predictions exist, fit a convex stack in log-risk-ratio space. Include the global scalar anchor. Recalibrate the stack and enforce signs. This frequently gives a useful variance-control mechanism: when flexible models disagree, the global model receives more weight.

## 4. Calibration is part of the model

Elasticity is a function of odds ratios, so probability miscalibration is amplified. scikit-learn's calibration guide recommends calibration on independent data and warns that isotonic calibration overfits when the calibration sample is small; its documentation also notes characteristic calibration behavior of Random Forest [6–7].

Fit an affine risk-ratio calibrator on inner OOF predictions:

```text
log RR_cal=alpha+s(raw-center), s>=0.
```

This simultaneously calibrates the average effect and shrinks heterogeneous dispersion. A slope near zero is a principled result: the base model has not learned reproducible heterogeneity.

Use 4–5 large grouped calibration bins. Do not rely only on calibration slope, because a globally linear correction can hide local failures.

Monotonicity should be enforced after calibration. Report raw violation rate and projection distance. Post-processing is a safety layer, not evidence that the unconstrained model was correct.

## 5. Evaluation focused on elasticity

### 5.1 Proper likelihood scores

The primary scores are:

- buyer multiclass NLL;
- H/C pairwise conditional NLL;
- L/C pairwise conditional NLL;
- equal-contrast NLL, the average of the two pair losses.

Cross-entropy is a proper score for the probabilities from which risk ratios are derived. VUS, multiclass AUC, and accuracy measure discrimination but not numerical probability ratios. They can be secondary diagnostics, never selection criteria.

### 5.2 Information gain over global response

For each outer fold, fit the global response on the outer training set and compare held-out NLL. Report the gain in millinats per pair buyer with a customer-clustered paired interval. This isolates the question that matters: do features locate meaningful elasticity differences beyond the portfolio average?

### 5.3 Calibration and heterogeneity

Use held-out calibration intercept/slope and grouped pairwise log-RR estimates. Work on large groups because the H and L side-arm buyer counts are the limiting information.

HTE calibration research emphasizes that discrimination and calibration are distinct; a model can rank effects but exaggerate their magnitude [2].

### 5.4 Ranking and prioritization

For “best places to adjust,” report top-fraction curves. RATE/TOC methods provide a general framework for evaluating whether a score prioritizes units with larger treatment effects [3]. In this application, pairwise likelihood and contribution-policy evaluation remain primary because RATE is naturally expressed on risk-difference or reward scales, while the requested estimand is a risk ratio.

### 5.5 Policy value

Evaluate the pricing rule by inverse propensity scoring on all held-out eligible quotes. The randomization probabilities are known, so no conversion model is required. Compare control, uniform H, uniform L, global scalar, flexible models, and selective top-k policies. Use customer-cluster bootstrap confidence intervals.

Doubly robust policy evaluation can be an optional variance-reduction check when a conversion/reward model is available [4], but it should not replace the model-free IPS benchmark.

### 5.6 Stability and temporal validity

A model must survive:

- outer group folds;
- multiple seeds;
- rolling forward-time tests;
- tariff-version and channel slices;
- top-k set overlap;
- missingness and out-of-support checks.

Small NLL improvements that do not reproduce in time should be treated as noise.

## 6. Cross-validation and leakage control

Use a final group-disjoint chronological holdout and nested group CV in development. `StratifiedGroupKFold` attempts to preserve class proportions while keeping groups non-overlapping [5]. Because the target among buyers is arm, stratify buyer folds by H/C/L and verify the side-arm count in every fold.

Calibration is itself a fitted model. Generate inner OOF raw scores, fit the calibrator on those scores, refit the base learner on the full outer-training set, and evaluate calibrated predictions on the outer fold.

All preprocessing is fold-local. This includes imputation, rare-level grouping, one-hot category maps, scalers, feature selection, and interaction construction.

## 7. Optuna design

Optuna's multivariate/group TPE can model dependencies in conditional spaces, while the median pruner can terminate weak trials after comparable intermediate fold results [15–16]. Use persistent RDB storage and one study per model family and contrast.

The search objective is proper conditional NLL. Policy value is too variable for inner-loop optimization and can induce severe winner's-curse bias. Use policy value only after proper-score and calibration gates.

### 7.1 Hyperparameters with highest value

The most important controls are:

1. minimum observations/minority buyers per leaf;
2. minimum Hessian per leaf or child;
3. depth/leaves;
4. L2 regularization;
5. learning rate and early-stopped rounds;
6. row/feature subsampling;
7. calibration slope.

Random-Forest tree count should normally be fixed once predictions stabilize. Random seeds are not hyperparameters.

### 7.2 Search-space summary

#### LightGBM pairwise

- learning rate: 0.005–0.10 log;
- depth: 2–7;
- leaves: 4–min(64,2^depth−1);
- dynamic `min_data_in_leaf` to retain roughly 10–100 side-arm buyers per leaf;
- `min_sum_hessian_in_leaf`: 0.1–50 log;
- feature fraction: 0.5–1;
- bagging fraction: 0.6–1;
- L1: 1e-8–100;
- L2: 1e-3–100;
- split gain: zero or 1e-5–1;
- max bin: 63, 127, 255, 511.

#### XGBoost pairwise

- eta: 0.005–0.10 log;
- depth: 2–7 or lossguide leaves 8–64;
- child weight: 1–100 log;
- subsample: 0.6–1;
- column sample: 0.5–1;
- gamma: zero or 1e-6–10;
- alpha: 1e-8–100;
- lambda: 1e-3–100;
- max bin: 64, 128, 256, 512;
- max delta step: 0, 1, 2, 5;
- scale positive weight fixed to one.

#### Random Forest pairwise

- trees fixed around 1,000;
- criterion: log loss or Gini;
- depth: 4–16;
- dynamic minimum leaf;
- max features: sqrt, log2, or 0.2–1;
- max samples: 0.5–1;
- impurity decrease: zero or 1e-8–1e-3;
- `ccp_alpha`: zero or 1e-8–1e-2;
- class weight fixed to none.

#### Scalar custom boosting

Use depth 1–4, learning rate 0.005–0.05, strong regularization, and much smaller Hessian thresholds than binary logistic boosting. The small price-dose variance changes the curvature scale.

Detailed spaces and executable suggest functions are in `elasticity_only/optuna_spaces.py`.

## 8. Feature interpretation

Use held-out permutation importance with pairwise NLL. scikit-learn notes that held-out permutation importance measures contribution to generalization and avoids the high-cardinality bias of tree impurity importance, while correlated predictors still require caution [9].

Interpret importance as predictive of price-response variation, not as a causal effect of the feature. The price arm is randomized; customer characteristics are not.

Priority features for investigation include base premium, margin, tariff uncertainty, competitor gap, channel, renewal status, and tariff version. Interactions that do not reproduce across folds or time should be removed even when a tree repeatedly selects them in-sample.

## 9. Decision gates and expected practical winner

The likely practical front-runners are:

- the global scalar model when heterogeneity signal is weak;
- calibrated pairwise LightGBM or XGBoost when stable nonlinear heterogeneity exists;
- a convex ensemble that shrinks those models toward the global anchor.

Random Forest is most valuable as an independent benchmark and a source of ensemble diversity. It should not be favored because it is “nonparametric”; with limited side-arm buyers, its flexibility can be a liability.

Select in this order:

1. validate the experiment;
2. establish the global response;
3. require OOF NLL improvement over global;
4. require calibration and low projection burden;
5. require time and seed stability;
6. choose among eligible models by lower confidence bound on IPS contribution lift;
7. prefer the simpler model if intervals materially overlap.

## 10. What not to do

The principal dead ends are:

- synthetic model tournaments;
- class balancing without exact correction;
- SMOTE or duplicated buyers;
- one-versus-rest probabilities treated as coherent multiclass probabilities;
- using displayed experimental price as a feature;
- row-random CV with repeat customers;
- tuning AUC/VUS or accuracy;
- estimating calibration on training data;
- selecting a policy from plug-in predicted value without randomized evaluation;
- claiming individual true elasticity from three-arm data;
- forcing a highly flexible two-gap surface before the scalar model is beaten.

## 11. Delivered implementation

The accompanying package contains:

- exact Bayes and relative-contribution identities;
- schema and experiment audits;
- group/time splitting;
- fold-safe preprocessing;
- global scalar and global pairwise fits;
- propensity-offset LightGBM and XGBoost pairwise models;
- inverse-propensity scikit-learn and Random-Forest pairwise wrappers;
- scalar custom objectives for LightGBM and XGBoost;
- Random-Forest moment inversion;
- multiclass benchmarks;
- OOF affine calibration and monotonic projection;
- likelihood, calibration, stability, and policy metrics;
- IPS/SNIPS and cluster bootstrap policy evaluation;
- Optuna search-space and nested-CV helpers;
- convex stacking;
- real-data-only examples and configuration.

The unit tests use deterministic toy arrays only to validate identities, gradients, offsets, and APIs. They do not produce or support a model ranking.

## 12. Bottom line

An elasticity-only model is coherent for the stated business objective because base conversion cancels from the quote-level choice among tested prices. The buyer Bayes-flip likelihood is the correct primary target for relative risk. The strongest production research design is:

```text
global scalar anchor
+ calibrated pairwise LightGBM/XGBoost challengers
+ heavily smoothed Random-Forest benchmark
+ constrained OOF ensemble
+ proper conditional likelihood selection
+ randomized IPS contribution evaluation
+ group/time validation.
```

The data, not the algorithm brand, must decide whether personalization is justified. A finding that the global elasticity model cannot be beaten is a successful and commercially useful outcome, not a modeling failure.
