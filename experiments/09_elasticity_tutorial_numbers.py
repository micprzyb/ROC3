#!/usr/bin/env python
"""
Every number quoted in ``docs/ELASTICITY_TUTORIAL.md``.

The tutorial uses one deliberately tiny, deterministic price test so that each quantity
can be checked with a pocket calculator.  Purchase counts are *constructed* to hit their
expected values exactly -- the arithmetic is then followable, and the statistics are left
to the 600k-customer run in ``08_elasticity.py``.

    .venv/bin/python experiments/09_elasticity_tutorial_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roc3.elasticity import (blp_slope, elasticity_from_posterior,  # noqa: E402
                             gates_elasticity, policy_value,
                             revenue_optimal_policy)

np.set_printoptions(suppress=True, precision=4, linewidth=120)

FAIL = []


def check(name, cond, detail=""):
    if not cond:
        FAIL.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('   ' + detail) if detail else ''}")


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# The toy price test
# --------------------------------------------------------------------------------------
M = np.array([0.9, 1.0, 1.1])           # price multipliers: -10%, 0%, +10%
Q = np.full(3, 1 / 3)                    # equal randomisation
TYPES = ["L (loyal)", "M (middling)", "S (sensitive)"]
BETA_T = np.array([[0.90, 0.85, 0.80],   # P(buy | type, arm) -- never observed
                   [0.60, 0.45, 0.30],
                   [0.40, 0.20, 0.10]])
PER_ARM = 100                            # customers of each type in each arm

head("1  The toy world: three customer types, each with a demand curve")
print(f"  price multipliers (cheapest first): {M}")
print(f"\n  {'type':<14} {'beta@0.9':>9} {'beta@1.0':>9} {'beta@1.1':>9}")
for t, nm in enumerate(TYPES):
    print(f"  {nm:<14} {BETA_T[t,0]:9.2f} {BETA_T[t,1]:9.2f} {BETA_T[t,2]:9.2f}")
print(f"\n  {PER_ARM} customers of each type in each arm -> "
      f"{PER_ARM * 3 * 3} customers in total")

LOGSPAN = np.log(M[2]) - np.log(M[0])
print(f"\n  log(1.1/0.9) = {LOGSPAN:.5f}   (the denominator of every elasticity below)")

EPS_T = -(np.log(BETA_T[:, 2]) - np.log(BETA_T[:, 0])) / LOGSPAN
print(f"\n  true arc elasticity per type:")
for t, nm in enumerate(TYPES):
    print(f"    {nm:<14} -log({BETA_T[t,2]:.2f}/{BETA_T[t,0]:.2f}) / {LOGSPAN:.5f}"
          f" = -({np.log(BETA_T[t,2]/BETA_T[t,0]):+.5f}) / {LOGSPAN:.5f}"
          f" = {EPS_T[t]:.4f}")

# --------------------------------------------------------------------------------------
head("2  What is actually observed")
# --------------------------------------------------------------------------------------
rows = []
for t in range(3):
    for k in range(3):
        n_buy = int(round(BETA_T[t, k] * PER_ARM))
        for i in range(PER_ARM):
            rows.append((t, k, 1 if i < n_buy else 0))
rows = np.array(rows)
typ, arm, bought = rows[:, 0], rows[:, 1], rows[:, 2]
N = len(rows)
print(f"  {N} rows of (customer features -> type, arm shown, bought yes/no).")
print(f"  Nobody's elasticity appears anywhere.  Purchase counts by type and arm:\n")
print(f"  {'type':<14} " + "".join(f"{f'arm {M[k]}':>12}" for k in range(3)))
for t, nm in enumerate(TYPES):
    print(f"  {nm:<14} " + "".join(
        f"{f'{int(bought[(typ==t)&(arm==k)].sum())}/{PER_ARM}':>12}" for k in range(3)))

# --------------------------------------------------------------------------------------
head("3  Fact 1 -- the arm posterior IS the demand curve, up to a constant")
# --------------------------------------------------------------------------------------
POST_T = BETA_T / BETA_T.sum(axis=1, keepdims=True)
print(f"  p_k = q_k beta_k / sum_j q_j beta_j ; with equal q the q's cancel:\n")
print(f"  {'type':<14} {'sum beta':>9} {'p_cheap':>9} {'p_mid':>9} {'p_exp':>9}")
for t, nm in enumerate(TYPES):
    print(f"  {nm:<14} {BETA_T[t].sum():9.2f} {POST_T[t,0]:9.4f} {POST_T[t,1]:9.4f} "
          f"{POST_T[t,2]:9.4f}")

eps_from_post = -(np.log(POST_T[:, 2]) - np.log(POST_T[:, 0])) / LOGSPAN
print(f"\n  elasticity recovered from the posterior alone:")
print(f"  {'type':<14} {'from beta':>10} {'from p':>10} {'difference':>12}")
for t, nm in enumerate(TYPES):
    print(f"  {nm:<14} {EPS_T[t]:10.4f} {eps_from_post[t]:10.4f} "
          f"{eps_from_post[t]-EPS_T[t]:12.2e}")
check("elasticity from the posterior equals elasticity from the demand curve",
      np.allclose(eps_from_post, EPS_T), f"max diff {np.abs(eps_from_post-EPS_T).max():.2e}")

POST = POST_T[typ]
check("roc3.elasticity_from_posterior agrees",
      np.allclose(elasticity_from_posterior(POST, M, Q), EPS_T[typ]))

# --------------------------------------------------------------------------------------
head("4  Fact 2 -- the best price is argmax_k  price_k * p_k(x)")
# --------------------------------------------------------------------------------------
print(f"  {'type':<14} " + "".join(f"{f'm*beta @{M[k]}':>14}" for k in range(3))
      + "   best")
for t, nm in enumerate(TYPES):
    v = M * BETA_T[t]
    print(f"  {nm:<14} " + "".join(f"{v[k]:14.4f}" for k in range(3))
          + f"   {M[int(np.argmax(v))]}")
print(f"\n  the same argmax, using only the posterior (the level c(x) cancels):")
print(f"  {'type':<14} " + "".join(f"{f'm*p @{M[k]}':>14}" for k in range(3)) + "   best")
for t, nm in enumerate(TYPES):
    v = M * POST_T[t]
    print(f"  {nm:<14} " + "".join(f"{v[k]:14.4f}" for k in range(3))
          + f"   {M[int(np.argmax(v))]}")
pol_true = np.argmax(M[None, :] * BETA_T[typ], axis=1)
pol_post = revenue_optimal_policy(POST, M, Q)
check("optimal price from the posterior == optimal price from the demand curve",
      np.array_equal(pol_true, pol_post))

# --------------------------------------------------------------------------------------
head("5  Validation 1 -- realised elasticity per bin, by hand")
# --------------------------------------------------------------------------------------
eps_hat = elasticity_from_posterior(POST, M, Q)
g = gates_elasticity(arm, bought, eps_hat, M, Q, n_bins=3, method="arc")
print("  bins are the model's own prediction; each bin's elasticity is read off its")
print("  raw per-arm purchase rates, which randomisation alone justifies.\n")
print(f"  {'bin':>4} {'n':>6} {'predicted':>10} " + "".join(
    f"{f'rate @{M[k]}':>12}" for k in range(3)) + f"{'realised':>10}")
for b in range(3):
    r = g["rates"][b]
    print(f"  {b:>4} {g['n'][b]:>6} {g['predicted'][b]:>10.4f} "
          + "".join(f"{r[k]:12.4f}" for k in range(3))
          + f"{g['realised'][b]:10.4f}")
print(f"\n  worked for the top bin: -log({g['rates'][2][2]:.2f}/{g['rates'][2][0]:.2f})"
      f" / {LOGSPAN:.5f} = {g['realised'][2]:.4f}")
check("realised elasticity per bin equals the truth (data built to be exact)",
      np.allclose(np.sort(g["realised"]), np.sort(EPS_T)),
      f"realised {np.round(g['realised'],4)} vs true {np.round(np.sort(EPS_T),4)}")
b = blp_slope(g)
print(f"\n  calibration of realised on predicted: slope {b['slope']:.4f}, "
      f"intercept {b['intercept']:+.4f}")
check("a perfect model calibrates at slope 1", abs(b["slope"] - 1) < 1e-6,
      f"slope {b['slope']:.6f}")

print("\n  ... and a model with NO signal (predicts the same elasticity for everyone,")
print("      so the bins are arbitrary):")
rng = np.random.default_rng(0)
eps_flat = rng.permutation(eps_hat)          # same values, attached to the wrong people
g0 = gates_elasticity(arm, bought, eps_flat, M, Q, n_bins=3, method="arc")
b0 = blp_slope(g0)
print(f"  {'bin':>4} {'n':>6} {'predicted':>10} {'realised':>10}")
for i in range(3):
    print(f"  {i:>4} {g0['n'][i]:>6} {g0['predicted'][i]:>10.4f} "
          f"{g0['realised'][i]:>10.4f}")
print(f"  spread {g0['spread']:+.4f} (vs {g['spread']:+.4f} for the good model);  "
      f"slope {b0['slope']:+.4f}")
check("the shuffled model shows a near-zero spread", abs(g0["spread"]) < 0.3,
      f"{g0['spread']:+.4f}")

# --------------------------------------------------------------------------------------
head("6  Validation 2 -- policy value by inverse-probability weighting, by hand")
# --------------------------------------------------------------------------------------
print("  V(policy) = mean over ALL customers of  1{arm shown = arm chosen}/q * price * bought")
print(f"  Here q = 1/3, so every match counts 3x.\n")
for k in range(3):
    pol = np.full(N, k)
    v = policy_value(arm, bought, pol, M, Q)
    n_match = int((arm == k).sum())
    n_buy = int(bought[(arm == k)].sum())
    truth = float((M[k] * BETA_T[typ, k]).mean())
    print(f"  always price {M[k]}:  {n_buy} buyers among the {n_match} shown that arm"
          f"  ->  3 * {M[k]} * {n_buy} / {N} = {v['value']:.5f}"
          f"   (truth {truth:.5f})")
    check(f"IPW recovers the true value of 'always {M[k]}'",
          abs(v["value"] - truth) < 1e-12, f"{v['value']:.6f} vs {truth:.6f}")

v_pol = policy_value(arm, bought, pol_post, M, Q)
truth_pol = float((M[pol_true] * BETA_T[typ, pol_true]).mean())
matched = (arm == pol_post)
print(f"\n  personalised: {int(bought[matched].sum())} buyers among the "
      f"{int(matched.sum())} whose shown arm happened to match the policy")
print(f"     -> {v_pol['value']:.5f}   (truth {truth_pol:.5f})")
check("IPW recovers the value of the personalised policy",
      abs(v_pol["value"] - truth_pol) < 1e-12,
      f"{v_pol['value']:.6f} vs {truth_pol:.6f}")

flats = {k: policy_value(arm, bought, np.full(N, k), M, Q)["value"] for k in range(3)}
best_flat = max(flats.values())
print(f"\n  best FLAT policy = {best_flat:.5f}  (price {M[max(flats, key=flats.get)]})")
print(f"  personalised     = {v_pol['value']:.5f}")
print(f"  -> personalisation is worth {100*(v_pol['value']/best_flat-1):+.2f}% "
      f"over the best flat price,")
print(f"     but {100*(v_pol['value']/flats[1]-1):+.2f}% over flat mid -- which is the "
      f"number that flatters.")

# --------------------------------------------------------------------------------------
head("7  Why the shuffled model still 'beats' flat mid")
# --------------------------------------------------------------------------------------
# a model with no usable features learns the MARGINAL posterior P(arm | bought)
marginal = BETA_T.mean(axis=0)
marginal = marginal / marginal.sum()
POST_FLAT = np.tile(marginal, (N, 1))
pol_flatmodel = revenue_optimal_policy(POST_FLAT, M, Q)
v_flatmodel = policy_value(arm, bought, pol_flatmodel, M, Q)
print(f"  a model with no usable features learns the MARGINAL posterior")
print(f"  P(arm | bought) = {np.round(marginal, 4)}, the same for every customer.")
print(f"  price * p then gives {np.round(M * marginal, 4)}, so it offers price "
      f"{M[pol_flatmodel[0]]} to everyone:")
print(f"  a constant rule worth {v_flatmodel['value']:.5f} -- exactly a flat policy.")
check("the no-signal model's policy value equals a flat policy",
      any(abs(v_flatmodel["value"] - f) < 1e-12 for f in flats.values()),
      f"{v_flatmodel['value']:.5f} vs flats {np.round(list(flats.values()),5)}")
print(f"  vs flat mid that is {100*(v_flatmodel['value']/flats[1]-1):+.2f}% -- "
      f"apparent skill from none.")

head("SUMMARY")
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): " + ", ".join(FAIL))
    sys.exit(1)
print("All checks passed.")
