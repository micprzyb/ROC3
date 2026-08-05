# Research log: elasticity-only insurance price testing

Date: 2026-08-05

## Scope and evidence boundary

This continuation deliberately separates three kinds of evidence:

1. **Algebraic identification results**, which follow from Bayes' rule and the randomized assignment mechanism.
2. **Software/API tests**, which use tiny deterministic or generated arrays only to verify formulas, gradients, offsets, cross-validation plumbing, and library compatibility.
3. **Empirical model comparisons**, which require the insurer's real randomized quote-level data and were not performed here because no such table was supplied.

No synthetic model leaderboard is included. No claim is made that LightGBM, XGBoost, Random Forest, or any other flexible model wins on the insurer's portfolio.

## Positive findings

### Base conversion cancels from the quote-level price choice

Let `g_t(x)` be contribution if the customer buys at tested arm `t`, and let `RR_tC(x)=q_t(x)/q_C(x)`. Then

`g_t q_t = q_C * g_t * RR_tC`.

Since `q_C(x)>0` is common across arms, the best tested price for that quote is determined by `argmax_t g_t RR_tC`. This is the central business result: a conversion-scale model is not required to choose among the tested prices, provided arm-specific contribution is known.

### Buyer-side Bayes flip exactly identifies relative demand

The buyer posterior satisfies

`r_t = pi_t q_t / sum_j pi_j q_j`,

so

`q_t/q_C = (r_t/pi_t)/(r_C/pi_C)`.

This is exact and does not depend on the conversion rate being small. The low conversion rate matters for practical signal-to-nuisance considerations, not for identification.

### Pairwise offset likelihood is a practical main model

For H versus C buyers,

`logit P(H | H/C, buyer, X) = log(pi_H/pi_C) + log RR_HC(X)`.

The corresponding identity holds for L versus C. This lets LightGBM and XGBoost use their ordinary binary log-loss objective with a known row-specific initial margin. It avoids a custom multiclass objective and estimates the two risk-ratio contrasts directly.

### Inverse-propensity weighting makes standard scikit-learn models usable

Among pairwise buyers, weighting each row by `1/pi_T` changes the weighted conditional target to `q_t/(q_t+q_C)`. Its logit is the desired log risk ratio. This supports penalized logistic regression and Random Forest without pretending that class balancing creates information.

### Strong shrinkage is structurally necessary

With a 10/80/10 allocation and a roughly 5% conversion rate, the side-arm buyer counts can be small even in a large quote table. Therefore the default hierarchy should start with a global scalar elasticity, then add controlled heterogeneity only when out-of-fold conditional likelihood and forward-time evidence justify it.

### Proper likelihood and randomized policy value answer different questions

Buyer conditional NLL evaluates whether risk ratios are numerically correct. Held-out IPS/SNIPS contribution evaluates whether those ratios produce a useful pricing rule. Neither replaces the other. A flexible model should pass both likelihood/calibration gates and a business-value gate.

## Important qualifications

### Buyer likelihood weights the buyer distribution

The pointwise risk ratio remains identified, but finite-capacity training and model selection emphasize customer regions that generate buyers. This is usually aligned with contribution value. A quote-population-weighted elasticity surface would need covariate-shift weighting, which effectively reintroduces a conversion/density-ratio model and can have high variance.

### Elasticity alone is not the same as optimal pricing without margins

The relative demand model identifies where customers are more or less price sensitive. Choosing whether a 10% increase or decrease is financially attractive also requires arm-specific contribution if a purchase occurs. Base conversion cancels, but margin does not.

### Three price points identify finite differences, not a global demand curve

The scalar model assumes local constant elasticity over the tested interval. The two-gap model allows asymmetric finite differences. Neither justifies extrapolation outside the tested range without additional experiments or structural assumptions.

### Marginal asymmetry is not proof of individual curvature

Different H/C and L/C effects can arise from mixtures of customers with heterogeneous constant elasticities. The two-gap model is operationally useful, but its difference should not automatically be interpreted as each individual's nonlinear demand curve.

### Random Forest is not automatically more robust

A forest can model nonlinear heterogeneity, but its leaf probabilities are unstable when side-arm buyer counts are small. Large leaves, limited depth, inverse-propensity weighting, and out-of-fold affine calibration are essential. Impurity importance is not an acceptable basis for business interpretation.

## Dead ends and rejected shortcuts

- Synthetic model tournaments for selecting the production architecture.
- Individual elasticity RMSE or correlation on real three-arm data; individual truth is unobserved.
- AUC, VUS, accuracy, or F1 as primary tuning criteria; they do not identify calibrated probability ratios.
- `class_weight='balanced'`, SMOTE, or duplicated buyers without exact posterior correction.
- One-versus-rest probabilities treated as a coherent three-arm posterior.
- Using displayed experimental premium or assigned multiplier as a predictor when arm is the target.
- Random row CV when customers or households repeat.
- Fitting calibration on training predictions.
- Optimizing high-variance policy value directly inside every Optuna trial.
- Large unrestricted two-gap trees before the global scalar model is beaten.
- Selecting a model because it has attractive in-sample segment plots.

## Implementation discoveries and corrections

- Pairwise XGBoost and LightGBM wrappers include the global log-risk-ratio intercept as well as the row-specific propensity offset. Unit tests confirm that a no-split model recovers the global conditional-likelihood intercept.
- The Optuna CV helper originally passed early-stopping keyword arguments to every estimator. A new signature-aware fitting helper now passes only supported arguments, so scikit-learn/Random-Forest and native boosting models use the same objective safely.
- The scikit-learn 1.8 logistic API deprecates the older `penalty` selector. The search now tunes `l1_ratio` directly: 0 gives L2, 1 gives L1, and intermediate values give elastic net.
- Optuna's multivariate/group TPE options are still documented as experimental in the current stable documentation. The configuration is retained because it handles conditional spaces well, but versions and warnings must be recorded.
- XGBoost and LightGBM custom scalar objectives use a positive Fisher/Gauss-Newton Hessian because the exact raw-score Hessian can become negative after the softplus positivity transform.

## Software verification completed

- Python compilation passed.
- 22 unit and smoke tests passed.
- Statement coverage was 77% across the package during the final test run.
- Native LightGBM and XGBoost pairwise and scalar wrappers were smoke-tested against the locally installed APIs.
- The four example scripts expose working CLI interfaces.
- The audit, one-trial pairwise tuning, one-trial scalar tuning, and OOF evaluation scripts were executed end-to-end on a temporary API fixture. The fixture and its numerical outputs are not included and are not evidence about insurance model quality.

## Unresolved empirical questions for the real-data agent

1. Does any heterogeneous model improve conditional NLL over the global scalar/two-gap anchor with a customer-clustered confidence interval?
2. Is the improvement stable in a strictly later tariff period?
3. Does the H/C contrast contain enough high-price buyers to support nonlinear segmentation?
4. How much monotonicity projection is required before and after calibration?
5. Are the top price-increase and price-decrease sets stable across folds, seeds, tariff versions, and channels?
6. Does the model policy improve held-out contribution relative to control and the best uniform tested multiplier under IPS/SNIPS?
7. Do competitor-position and tariff-uncertainty variables explain stable elasticity variation, or only transient operational artifacts?
8. Is one scalar elasticity sufficient, or is two-gap asymmetry reproducible and decision-relevant?

These questions cannot be resolved without the actual randomized quote-level table.
