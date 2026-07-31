#!/usr/bin/env python
"""
Evaluating a price-elasticity model when no elasticity is ever observed.

Every number quoted in ``docs/ELASTICITY.md``.

    .venv/bin/python experiments/08_elasticity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"

from roc3.elasticity import (blp_slope, compare_elasticity_models,  # noqa: E402
                             demand_curve_from_posterior, dr_pseudo_outcome,
                             elasticity_from_posterior, elasticity_report,
                             format_elasticity_report, gates_elasticity,
                             plot_elasticity_diagnostics, policy_value,
                             revenue_optimal_policy, targeting_curve)

FAILURES: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('   ' + detail) if detail else ''}")


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# A price test where we happen to know the truth (so the methods can be graded)
# --------------------------------------------------------------------------------------
M = np.array([0.9, 1.0, 1.1])                  # price multipliers, cheapest first
GAMMA = 4.0                                     # price coefficient on the logit scale
CSHIFT = GAMMA * np.log(M)
Q = np.full(3, 1 / 3)
N = 600_000

rng = np.random.default_rng(0)
# two customer features; only the first drives price sensitivity
z = rng.normal(0, 1.2, N)                       # latent price-INsensitivity
noise = rng.normal(0, 1.0, N)                   # an irrelevant feature
arm = rng.integers(0, 3, N)
BETA = 1.0 / (1.0 + np.exp(-(z[:, None] - CSHIFT[None, :])))
bought = (rng.random(N) < BETA[np.arange(N), arm]).astype(int)

EPS_TRUE = -(np.log(BETA[:, 2]) - np.log(BETA[:, 0])) / (np.log(M[2]) - np.log(M[0]))
POST_TRUE = BETA / BETA.sum(axis=1, keepdims=True)      # = P(arm | x, bought), equal q

head("1  The two structural identities")
print("""  With p_k(x) = P(arm=k | x, bought) and equal randomisation, p_k is proportional
  to beta_k, so the unknown normalising factor cancels out of any ratio.""")
eps_hat = elasticity_from_posterior(POST_TRUE, M, Q)
check("elasticity recovered EXACTLY from the arm posterior",
      np.abs(eps_hat - EPS_TRUE).max() < 1e-10,
      f"max |error| over {N:,} customers = {np.abs(eps_hat - EPS_TRUE).max():.2e}")
pol_true = np.argmax(M[None, :] * BETA, axis=1)
pol_post = revenue_optimal_policy(POST_TRUE, M, Q)
check("revenue-optimal price recovered from the posterior alone",
      float((pol_true == pol_post).mean()) == 1.0,
      f"{(pol_true == pol_post).mean():.6%} agreement -- it is the ROC operating point "
      f"with w proportional to the prices")
p_buy_true = (Q[None, :] * BETA).sum(axis=1)
beta_rec = demand_curve_from_posterior(POST_TRUE, p_buy_true, M, Q)
check("full demand curve recovered from posterior + P(buy|x)",
      np.abs(beta_rec - BETA).max() < 1e-10,
      f"max |error| = {np.abs(beta_rec - BETA).max():.2e}")

D = np.array([bought[arm == k].mean() for k in range(3)])
print(f"\n  per-arm purchase rates {np.round(D, 4)}   "
      f"true elasticity ranges {EPS_TRUE.min():.2f} .. {EPS_TRUE.max():.2f} "
      f"(mean {EPS_TRUE.mean():.2f})")


# --------------------------------------------------------------------------------------
head("2  Fit real models on the buyers only, out of fold")
# --------------------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

X = np.stack([z, noise], axis=1)                # the features a modeller would have
buy = bought == 1
Xb, armb = X[buy], arm[buy]
print(f"  {buy.sum():,} buyers of {N:,} customers")

MODELS = {
    "A: logistic on both features": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)),
    "B: gradient boosting": HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, random_state=0),
    "C: logistic on the IRRELEVANT feature only": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)),
}
POSTS = {}
for name, mdl in MODELS.items():
    cols = [1] if "IRRELEVANT" in name else [0, 1]
    P_oof = cross_val_predict(mdl, Xb[:, cols], armb, cv=3, method="predict_proba")
    full = np.full((N, 3), np.nan)
    full[buy] = P_oof
    # the model is a function of x, so it can be evaluated on non-buyers too
    mdl.fit(Xb[:, cols], armb)
    full[~buy] = mdl.predict_proba(X[~buy][:, cols])
    POSTS[name] = full
    e = elasticity_from_posterior(full, M, Q)
    print(f"  {name:<44} corr(eps_hat, eps_true) = "
          f"{np.corrcoef(e, EPS_TRUE)[0, 1]:+.4f}")


# --------------------------------------------------------------------------------------
head("3  Validation 1 — realised elasticity by predicted-elasticity bin")
# --------------------------------------------------------------------------------------
print("""  There is no elasticity label for any customer.  But randomisation means any
  GROUP of customers contains all three arms, so a group's elasticity is read straight
  off its raw purchase rates.  The model's prediction supplies the groups.""")
for name in MODELS:
    e = elasticity_from_posterior(POSTS[name], M, Q)
    g = gates_elasticity(arm, bought, e, M, Q, n_bins=5)
    b = blp_slope(g)
    print(f"\n  {name}")
    print(f"    {'bin':>4} {'n':>8} {'predicted':>10} {'realised':>10} {'95% CI':>18}"
          f"   purchase rates by arm")
    for i in range(5):
        ci = f"({g['ci_lower'][i]:.2f}, {g['ci_upper'][i]:.2f})"
        print(f"    {i:>4} {g['n'][i]:>8,} {g['predicted'][i]:>10.3f} "
              f"{g['realised'][i]:>10.3f} {ci:>18}   {np.round(g['rates'][i], 4)}")
    print(f"    monotone {g['monotone']}   spread {g['spread']:+.3f} "
          f"+/- {g['spread_stderr']:.3f}   calibration slope {b['slope']:+.3f} "
          f"+/- {b['slope_stderr']:.3f}   p(no signal) {b['p_value_no_signal']:.1e}")
    if "IRRELEVANT" in name:
        check("the useless model shows no elasticity signal",
              b["p_value_no_signal"] > 0.05 or abs(b["slope"]) < 0.2,
              f"slope {b['slope']:+.3f}, p {b['p_value_no_signal']:.3f}")
    else:
        check(f"{name[:1]}: detects real elasticity signal",
              b["p_value_no_signal"] < 1e-3 and g["spread"] > 0,
              f"slope {b['slope']:+.3f}, spread {g['spread']:+.3f}")

# the oracle, for reference
g_or = gates_elasticity(arm, bought, EPS_TRUE, M, Q, n_bins=5)
b_or = blp_slope(g_or)
print(f"\n  ORACLE (the true elasticity)  slope {b_or['slope']:+.3f} "
      f"+/- {b_or['slope_stderr']:.3f}   spread {g_or['spread']:+.3f}")
check("the oracle is calibrated (slope indistinguishable from 1)",
      b_or["p_value_miscalibrated"] > 0.01,
      f"slope {b_or['slope']:.3f}, p(slope=1) = {b_or['p_value_miscalibrated']:.3f}")


# --------------------------------------------------------------------------------------
head("4  Validation 2 — what each model's pricing policy is actually worth")
# --------------------------------------------------------------------------------------
print("""  The arms were randomised, so inverse-probability weighting gives an UNBIASED
  estimate of the revenue any policy would have earned, with no modelling assumptions.""")
flat = {k: policy_value(arm, bought, np.full(N, k), M, Q) for k in range(3)}
for k in range(3):
    print(f"  flat price {M[k]:.2f}                        "
          f"{flat[k]['value']:.5f} +/- {flat[k]['stderr']:.5f}")
v_oracle = policy_value(arm, bought, pol_true, M, Q)
print(f"  personalised, ORACLE                  "
      f"{v_oracle['value']:.5f} +/- {v_oracle['stderr']:.5f}")
truth_v = (M[pol_true] * BETA[np.arange(N), pol_true]).mean()
check("IPW recovers the oracle policy value it cannot see",
      abs(v_oracle["value"] - truth_v) < 4 * v_oracle["stderr"],
      f"IPW {v_oracle['value']:.5f} vs truth {truth_v:.5f} "
      f"(+/- {v_oracle['stderr']:.5f})")
best_flat = max(flat.values(), key=lambda d: d["value"])["value"]
print(f"\n  BEST FLAT price is {best_flat:.5f}.  Benchmark against THAT, not against an")
print("  arbitrary baseline -- a model with no personalisation signal still 'wins' against")
print("  flat-mid simply by discovering that the cheap price is better on average.\n")
print(f"  {'model':<44} {'value':>9} {'vs flat mid':>12} {'vs BEST flat':>13}")
for name in MODELS:
    pol = revenue_optimal_policy(POSTS[name], M, Q)
    v = policy_value(arm, bought, pol, M, Q)
    print(f"  {name:<44} {v['value']:9.5f} "
          f"{100 * (v['value'] / flat[1]['value'] - 1):11.2f}% "
          f"{100 * (v['value'] / best_flat - 1):12.2f}%")
    if "IRRELEVANT" in name:
        check("the useless model adds nothing over the best FLAT price",
              abs(v["value"] - best_flat) < 3 * v["stderr"],
              f"{v['value']:.5f} vs best flat {best_flat:.5f} -- its 'personalised' "
              f"policy is the constant rule 'always cheap'")
print(f"  {'personalised, ORACLE':<44} {v_oracle['value']:9.5f} "
      f"{100 * (v_oracle['value'] / flat[1]['value'] - 1):11.2f}% "
      f"{100 * (v_oracle['value'] / best_flat - 1):12.2f}%")


# --------------------------------------------------------------------------------------
head("5  Validation 3 — the doubly-robust pseudo-outcome, and why it is not enough")
# --------------------------------------------------------------------------------------
beta_hat = demand_curve_from_posterior(POSTS["B: gradient boosting"],
                                       np.full(N, bought.mean()), M, Q)
psi = dr_pseudo_outcome(arm, bought, beta_hat, Q, pair=(0, -1))
uplift_true = BETA[:, 0] - BETA[:, 2]
print(f"  DR pseudo-outcome for the UPLIFT beta_cheap - beta_expensive:")
print(f"    mean {psi.mean():+.5f}   true mean uplift {uplift_true.mean():+.5f}")
check("the DR pseudo-outcome is unbiased for the uplift",
      abs(psi.mean() - uplift_true.mean()) < 4 * psi.std() / np.sqrt(N),
      f"{psi.mean():.5f} vs {uplift_true.mean():.5f}")
print(f"    per-customer noise: sd {psi.std():.3f} against a true spread of "
      f"{uplift_true.std():.3f}  (noise/signal {psi.std() / uplift_true.std():.0f}x)")
from scipy.stats import pearsonr, spearmanr  # noqa: E402
print(f"\n    {'model':<46} {'Pearson':>9} {'Spearman':>10}")
for name in MODELS:
    e = elasticity_from_posterior(POSTS[name], M, Q)
    print(f"    {name:<46} {pearsonr(e, psi).statistic:+9.4f} "
          f"{spearmanr(e, psi).statistic:+10.4f}")
print(f"    {'ORACLE (the true elasticity)':<46} "
      f"{pearsonr(EPS_TRUE, psi).statistic:+9.4f} "
      f"{spearmanr(EPS_TRUE, psi).statistic:+10.4f}")
print("""
  RANK correlation against psi is unsafe.  psi is a near-three-point variable whose
  atoms depend on x, so its RANKS encode beta(x) rather than the uplift; the Spearman
  value swings with the nuisance model even though AIPW is unbiased throughout
  (measured: -0.105 with a poor beta_hat, +0.005 with the true one).  Pearson tracks
  the truth in all three cases.  Use regression against psi, never rank correlation.""")
check("Pearson against psi tracks the (near-zero) truth for the oracle",
      abs(pearsonr(EPS_TRUE, psi).statistic) < 0.02,
      f"{pearsonr(EPS_TRUE, psi).statistic:+.4f} against a true "
      f"corr(elasticity, uplift) of {pearsonr(EPS_TRUE, uplift_true).statistic:+.4f}")
print("""
  Note the target mismatch: psi estimates a DIFFERENCE (uplift), elasticity is a RATIO,
  and the revenue gain from discounting is neither.  They rank customers differently:""")
rev_gain_true = M[0] * BETA[:, 0] - M[1] * BETA[:, 1]
print(f"    Spearman(true elasticity, true uplift)      "
      f"{spearmanr(EPS_TRUE, uplift_true).statistic:+.4f}")
print(f"    Spearman(true elasticity, true revenue gain) "
      f"{spearmanr(EPS_TRUE, rev_gain_true).statistic:+.4f}")
print(f"    Spearman(true uplift, true revenue gain)     "
      f"{spearmanr(uplift_true, rev_gain_true).statistic:+.4f}")


# --------------------------------------------------------------------------------------
head("6  Comparing two models head to head")
# --------------------------------------------------------------------------------------
sub = rng.choice(N, 200_000, replace=False)
eA = elasticity_from_posterior(POSTS["A: logistic on both features"], M, Q)
eB = elasticity_from_posterior(POSTS["B: gradient boosting"], M, Q)
eC = elasticity_from_posterior(POSTS["C: logistic on the IRRELEVANT feature only"], M, Q)
pA = revenue_optimal_policy(POSTS["A: logistic on both features"], M, Q)
pB = revenue_optimal_policy(POSTS["B: gradient boosting"], M, Q)
pC = revenue_optimal_policy(POSTS["C: logistic on the IRRELEVANT feature only"], M, Q)

for lbl, (s1, s2, q1, q2) in {
        "A vs C  (real signal vs none)": (eA[sub], eC[sub], pA[sub], pC[sub]),
        "A vs B  (two real models)": (eA[sub], eB[sub], pA[sub], pB[sub])}.items():
    cmp = compare_elasticity_models(arm[sub], bought[sub], s1, s2, q1, q2, M, Q,
                                    n_bins=5, n_boot=200, seed=1)
    print(f"\n  {lbl}   (paired bootstrap, {cmp['n_boot']} replicates)")
    for key in ("policy_value", "gates_spread", "calibration_slope"):
        d = cmp[key]["a_minus_b"]
        print(f"    {key:<19} a {cmp[key]['a']:+.5f}   b {cmp[key]['b']:+.5f}   "
              f"diff {d['diff']:+.5f}  95% CI ({d['ci'][0]:+.5f}, {d['ci'][1]:+.5f})"
              f"  {'SIGNIFICANT' if d['significant'] else 'not significant'}")
    if "vs C" in lbl:
        check("a real model beats a useless one on policy value",
              cmp["policy_value"]["a_minus_b"]["significant"]
              and cmp["policy_value"]["a_minus_b"]["diff"] > 0)


# --------------------------------------------------------------------------------------
head("7  The full report and the diagnostics figure")
# --------------------------------------------------------------------------------------
rep = elasticity_report(arm, bought, POSTS["B: gradient boosting"], M, Q, n_bins=6,
                        p_buy=np.full(N, bought.mean()))
print(format_elasticity_report(rep))
fr_o, v_o, _ = targeting_curve(arm, bought, M[pol_true] * BETA[np.arange(N), pol_true]
                               - M[1] * BETA[:, 1], pol_true, M, Q)
plot_elasticity_diagnostics(rep, path=FIG / "10_elasticity_diagnostics.png",
                            truth=(fr_o, v_o),
                            title="Grading an elasticity model with no elasticity labels")
print(f"\nwrote {FIG / '10_elasticity_diagnostics.png'}")


head("SUMMARY")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("All checks passed.")
