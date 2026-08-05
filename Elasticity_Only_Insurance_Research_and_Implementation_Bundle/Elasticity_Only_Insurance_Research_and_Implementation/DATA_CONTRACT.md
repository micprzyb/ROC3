# Real insurance price-test data contract

This package is deliberately **real-data only**. It does not generate insurance observations and it must not substitute an unrelated public dataset when the randomized experiment table is missing.

## 1. Unit of analysis

Use one row per **eligible randomized quote opportunity**. Include buyers and nonbuyers, even though the elasticity-shape models are trained on buyers. The complete eligible table is required for experiment auditing and held-out policy evaluation.

The unit must correspond to the point at which the randomized price arm was assigned. If a customer can request repeated quotes, preserve all legitimate opportunities but provide a stable customer or household identifier so that folds, bootstrap samples, and holdouts remain group-disjoint.

## 2. Required columns for elasticity shape

| Column | Type | Meaning |
|---|---|---|
| `quote_id` | string | Unique identifier for the randomized quote opportunity. |
| `customer_id` | string | Stable customer or household cluster identifier. |
| `quote_timestamp` | timestamp | Time at which the quote entered the experiment. Store timezone explicitly. |
| `eligible` | 0/1 | Whether the row was in the experiment's eligible population before assignment. |
| `arm` | `H`, `C`, `L` | Assigned high-price, control, or low-price arm. |
| `purchase` | 0/1 | Conversion within the pre-specified outcome window. |
| `pi_H` | float | Logged probability of assignment to H for this row. |
| `pi_C` | float | Logged probability of assignment to C for this row. |
| `pi_L` | float | Logged probability of assignment to L for this row. |

Every propensity must be strictly positive and each row must satisfy

`pi_H + pi_C + pi_L = 1`.

The logged row-level propensities take precedence over the nominal 10/80/10 design. If the experiment used quotas, blocks, channel-specific probabilities, or time-varying allocation, those realized design probabilities must be present.

## 3. Required columns for elasticity-only price decisions

To choose among the tested prices without estimating base conversion, provide the expected contribution obtained **if the customer buys** under every tested price:

| Column | Type | Meaning |
|---|---|---|
| `contribution_H` | float | Expected contribution per purchased policy at the high price. |
| `contribution_C` | float | Expected contribution per purchased policy at control. |
| `contribution_L` | float | Expected contribution per purchased policy at the low price. |

Contribution should use the business definition that will govern deployment: displayed premium minus expected claims, commission, taxes, acquisition expense, payment costs, and other arm-dependent variable costs. If only revenue is available, use price revenue but label the resulting policy as a revenue policy rather than a contribution policy.

For customer `x`, the arm-specific expected contribution is

`q_t(x) * contribution_t(x)`.

Because `q_t(x) = q_C(x) * RR_tC(x)`, the common positive factor `q_C(x)` cancels from the choice among arms:

`argmax_t contribution_t(x) * RR_tC(x)`.

## 4. Strongly recommended experiment fields

| Column | Purpose |
|---|---|
| `randomization_block` | Reproduce blocked or quota-based assignment and randomization tests. |
| `assigned_multiplier` | Intended experimental multiplier. |
| `base_premium` | Pre-randomization premium or technical tariff. |
| `displayed_premium` | Actual premium shown; audit rounding and implementation compliance only. Never use it as an elasticity-model feature. |
| `tariff_version` | Detect tariff drift and implementation changes. |
| `channel` | Channel/aggregator context, if known before assignment. |
| `purchase_timestamp` | Verify the outcome window. |
| `cancelled_early` | Optional sensitivity outcome, defined in advance. |
| `expected_claim_cost` | Build arm-specific contribution and explain economically meaningful heterogeneity. |
| `tariff_uncertainty` | Pre-treatment uncertainty or out-of-distribution score from the tariff model. |
| `competitor_gap` | Pre-treatment competitor price gap or rank, if legally and operationally available before assignment. |

## 5. Feature manifest

Supply an explicit feature manifest rather than selecting all columns automatically. At minimum it should contain:

| Field | Meaning |
|---|---|
| `feature_name` | Input column. |
| `type` | `numeric`, `categorical`, `ordinal`, or `date-derived`. |
| `available_before_assignment` | Must be `true`. |
| `business_definition` | Human-readable definition. |
| `missingness_reason` | Structural, operational, unknown, or not applicable. |
| `expected_stability` | Stable, tariff-version dependent, seasonal, or transient. |
| `approved_for_pricing` | Governance decision. |

## 6. Forbidden feature classes

Do not use any variable that reveals or is caused by randomized price assignment or purchase. Examples include:

- assigned arm or multiplier;
- displayed experimental premium;
- price shown after the multiplier;
- purchase, payment method chosen after purchase, policy identifier, or cancellation observed after purchase;
- agent actions made after seeing the experimental price;
- features recomputed after assignment using the displayed premium;
- future claims or renewals.

The treatment arm is the **target** in buyer-flip models and enters the likelihood through its logged propensity. It is never a predictor.

## 7. Timing and repeated quotes

Document:

1. eligibility timestamp;
2. assignment timestamp;
3. price display timestamp;
4. purchase attribution window;
5. rules for revisited or revised quotes;
6. whether one customer could see different arms;
7. whether manual overrides occurred after randomization.

All rows from one customer or household must remain in the same split. For deployment across time, reserve the most recent eligible period as a final chronological holdout.

## 8. No synthetic fallback

If the required randomized table is absent, the agent must stop after producing a missing-data report. It may run deterministic unit tests of formulas and software APIs, but it must not generate a synthetic insurance dataset, rank models on generated outcomes, or describe those rankings as evidence about the portfolio.
