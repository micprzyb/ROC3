#!/usr/bin/env python
"""
The price-test formulation: derivation checks, the ceiling, and what to report.

Every number quoted in ``docs/PRICETEST.md``.

    .venv/bin/python experiments/07_pricetest.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"

from roc3 import CHANCE_VUS, ordinal_roc_surface, roc_surface  # noqa: E402
from roc3 import vus_forced_choice, vus_ordinal  # noqa: E402
from roc3.constrained import _free_var_polytope, _vertices, vus_ceiling  # noqa: E402
from roc3.core import _pareto_mask, convex_hull_vus  # noqa: E402
from roc3.pricetest import (audit, ceiling_from_elasticity,  # noqa: E402
                            ceiling_from_purchase_rates, check_monotone_demand,
                            format_pricetest_report, plot_ceiling_vs_elasticity,
                            posterior_prevalences, pricetest_report)

FAILURES: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('   ' + detail) if detail else ''}")


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ======================================================================================
head("1  Simulate the price test, and check the constraint is DERIVED not assumed")
# ======================================================================================
print("""  s      = a customer's price-INsensitivity (higher -> buys even when expensive)
  beta_k = P(buy | customer s, arm k) = sigmoid(s - c_k),  c_cheap < c_mid < c_exp
  arms are randomised with equal probability; we then keep only the BUYERS.""")

rng = np.random.default_rng(0)
PRICE = np.array([-0.4, 0.0, 0.4])                     # -10%, 0%, +10% on the logit scale
N = 1_200_000
s = rng.normal(0.0, 1.0, N)
arm = rng.integers(0, 3, N)
beta = 1.0 / (1.0 + np.exp(-(s - PRICE[arm])))
bought = rng.random(N) < beta
s_b, arm_b = s[bought], arm[bought]
D = np.array([float(bought[arm == k].mean()) for k in range(3)])

print(f"\n  buyers: {bought.sum():,} of {N:,}")
print(f"  purchase rate per arm D_k (cheapest first) : {np.round(D, 4)}")
arc_e = -((D[2] - D[0]) / D.mean()) / 0.2
print(f"  implied arc elasticity                     : {arc_e:.2f}")

pi_emp = np.array([float((arm_b == k).mean()) for k in range(3)])
pi_formula = posterior_prevalences(D)
print(f"\n  prevalence among buyers, empirical         : {np.round(pi_emp, 4)}")
print(f"  prevalence among buyers, D_k / sum D_j     : {np.round(pi_formula, 4)}")
check("prevalence among buyers = normalised purchase rates",
      np.allclose(pi_emp, pi_formula, atol=2e-3),
      f"max |diff| = {np.abs(pi_emp - pi_formula).max():.2e}")

# the TRUE posterior over arms for each buyer (equal randomisation)
B = 1.0 / (1.0 + np.exp(-(s_b[:, None] - PRICE[None, :])))
P = B / B.sum(axis=1, keepdims=True)
md = check_monotone_demand(P)
check("the true posterior is ordered p_cheap >= p_mid >= p_exp for EVERY buyer",
      md["satisfied"], f"{md['fraction_satisfied']:.1%} of rows")
print("  -> the ordering constraint is a CONSEQUENCE of monotone demand, not an")
print("     assumption bolted on afterwards.  It therefore constrains the true")
print("     posterior, i.e. it binds every classifier (Reading A of CONSTRAINED.md).")
check("the constraint also forces the prevalences to be ordered",
      pi_emp[0] > pi_emp[1] > pi_emp[2], f"{np.round(pi_emp, 4)}")


# ======================================================================================
head("2  The ceiling is a function of the MEASURED demand response")
# ======================================================================================
info = ceiling_from_purchase_rates(D)
print(f"  computed from the topline alone, before any model is fitted:")
print(f"    prevalence among buyers   {np.round(info['prevalence_among_buyers'], 4)}")
print(f"    VUS ceiling               {info['vus_ceiling']:.4f}")
print(f"    chance                    {CHANCE_VUS:.4f}")
print(f"    pairwise AUC ceilings     "
      + ", ".join(f"{k}:{v:.3f}" for k, v in info["pairwise_auc_ceiling"].items()))
sc = info["scalar_ceilings"]
print(f"    max accuracy              {sc['accuracy']:.4f}")
print(f"    max balanced accuracy     {sc['balanced_accuracy']:.4f}")

print(f"\n  {'|eps|':>6} {'D ratios (c/m, m/e)':>22} {'pi ascending':>26} "
      f"{'ceiling':>9} {'band width':>11}")
for e in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
    inf = ceiling_from_elasticity(e)
    d = inf["demand_ratios"]
    ratios = f"{d['cheap/mid']:.3f}, {d['mid/expensive']:.3f}"
    print(f"  {e:6.1f} {ratios:>22} "
          f"{str(np.round(inf['prevalence_ascending'], 4)):>26} "
          f"{inf['vus_ceiling']:9.4f} {inf['vus_ceiling'] - CHANCE_VUS:11.4f}")
print(f"\n  chance is {CHANCE_VUS:.4f}.  At unit elasticity the ENTIRE achievable range")
print(f"  is [{CHANCE_VUS:.3f}, {ceiling_from_elasticity(1.0)['vus_ceiling']:.3f}] "
      f"-- about five points wide.")
check("ceiling rises monotonically with elasticity",
      all(ceiling_from_elasticity(a)["vus_ceiling"]
          <= ceiling_from_elasticity(b)["vus_ceiling"] + 1e-9
          for a, b in zip([0.5, 1, 2, 3, 5], [1, 2, 3, 5, 8])))
check("ceiling -> chance as elasticity -> 0",
      abs(ceiling_from_elasticity(0.01)["vus_ceiling"] - CHANCE_VUS) < 0.01,
      f"{ceiling_from_elasticity(0.01)['vus_ceiling']:.4f}")


# ======================================================================================
head("3  The IDEAL OBSERVER against that ceiling")
# ======================================================================================
sub = rng.choice(len(P), 40_000, replace=False)
surf = roc_surface(arm_b[sub], P[sub], classes=[0, 1, 2], resolution=200)
ceil = info["vus_ceiling"]
v3 = vus_forced_choice(arm_b[sub], P[sub], classes=[0, 1, 2])
print("  scoring with the TRUE posterior -- no model can do better than this:\n")
print(f"    geometric VUS                     {surf.vus:.4f}")
print(f"    3AFC VUS                          {v3:.4f}")
print(f"    chance                            {CHANCE_VUS:.4f}")
print(f"    ceiling                           {ceil:.4f}")
print(f"    on the conventional [1/6,1] scale {(surf.vus - CHANCE_VUS) / (1 - CHANCE_VUS):+.4f}"
      f"   <- reads as worthless")
print(f"    as a fraction of the attainable   "
      f"{(surf.vus - CHANCE_VUS) / (ceil - CHANCE_VUS):.1%}   <- the honest number")
check("the ideal observer respects its own ceiling", surf.vus <= ceil + 1e-6,
      f"VUS {surf.vus:.4f} vs ceiling {ceil:.4f}")


# ======================================================================================
head("4  The problem is one-dimensional, so the ORDINAL machinery is the right tool")
# ======================================================================================
print("""  beta_k(s) = sigmoid(s - c_k) has the monotone likelihood ratio property in s, so
  the three buyer populations are stochastically ordered along the single index s and
  every Bayes rule is a pair of CUT-POINTS on it.  That is roc3.ordinal, and it gives
  two numbers you can deploy instead of a weight vector.""")
vo = vus_ordinal(arm_b[sub], s_b[sub], classes=[0, 1, 2])
osurf = ordinal_roc_surface(arm_b[sub], s_b[sub], classes=[0, 1, 2], resolution=300)
print(f"\n    P(s_cheap < s_mid < s_expensive)  {vo:.4f}")
print(f"    3AFC VUS on the full posterior    {v3:.4f}")
print(f"    ordinal two-cut-point VUS         {osurf.vus:.4f}")
print(f"    geometric VUS on the posterior    {surf.vus:.4f}")
check("the 1-D ordinal rank VUS matches the full-posterior 3AFC VUS",
      abs(vo - v3) < 0.01, f"{vo:.4f} vs {v3:.4f}")
op = osurf.select("youden")
print(f"\n  {op.describe()}")


# ======================================================================================
head("5  Does restricting to ORDINAL rules tighten the ceiling?  (it does not)")
# ======================================================================================
print("""  Ordinal decision regions must be intervals of the index, which by stochastic
  ordering adds  C[k,.] cumulative sums ordered across k.  Four extra linear
  constraints.  Do they lower the ceiling?""")


def ceiling_ordinal(pi_asc):
    A, b, _ = _free_var_polytope(pi_asc)
    extra = []
    for k in (0, 1):
        r = np.zeros(6); r[2 * (k + 1)] = 1.0; r[2 * k] = -1.0; extra.append(r)
        r = np.zeros(6); r[2 * (k + 1)] = 1.0; r[2 * (k + 1) + 1] = 1.0
        r[2 * k] = -1.0; r[2 * k + 1] = -1.0; extra.append(r)
    A = np.vstack([A, np.array(extra)]); b = np.concatenate([b, np.zeros(4)])
    V = _vertices(A, b)
    S = np.clip(np.stack([V[:, 0], V[:, 3], 1.0 - V[:, 4] - V[:, 5]], 1), 0, 1) + 0.0
    return float(convex_hull_vus(np.unique(np.round(S, 9), axis=0)))


print(f"\n  {'|eps|':>6} {'general ceiling':>16} {'ordinal ceiling':>16} {'difference':>11}")
same = True
for e in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
    pa = ceiling_from_elasticity(e)["prevalence_ascending"]
    g, o = vus_ceiling(pa), ceiling_ordinal(pa)
    same &= abs(g - o) < 1e-4
    print(f"  {e:6.1f} {g:16.4f} {o:16.4f} {g - o:11.4f}")
check("the ordinal restriction does not change the ceiling", same)
pa = ceiling_from_elasticity(1.0)["prevalence_ascending"]
A, b, _ = _free_var_polytope(pa); V = _vertices(A, b)
S = np.clip(np.stack([V[:, 0], V[:, 3], 1 - V[:, 4] - V[:, 5]], 1), 0, 1)
ok = np.ones(len(V), bool)
for k in (0, 1):
    ok &= V[:, 2 * k] >= V[:, 2 * (k + 1)] - 1e-9
    ok &= (V[:, 2 * k] + V[:, 2 * k + 1]) >= (V[:, 2 * (k + 1)] + V[:, 2 * (k + 1) + 1]) - 1e-9
pf = _pareto_mask(S)
print(f"\n  polytope vertices {len(V)}; ordinal-feasible {ok.sum()}; "
      f"on the Pareto frontier {pf.sum()}, of which ordinal-feasible {(pf & ok).sum()}")
print("  -> ordinality removes rules, and even one frontier vertex, but not enough of")
print("     the frontier to change the volume.  A clean negative result: no separate")
print("     ordinal ceiling is needed.")


# ======================================================================================
head("6  The ceiling as a leakage detector")
# ======================================================================================
for v, label in [(0.19, "an honest model"),
                 (0.26, "just over the ceiling"),
                 (0.62, "'our model gets VUS 0.62!'"),
                 (0.16, "a model at chance")]:
    a = audit(v, D)
    print(f"  VUS {v:.2f}  ({label:<28}) -> {a['verdict'].upper():<20}"
          f" ceiling {a['ceiling']:.3f}, {a['fraction_of_attainable']:>6.1%} of attainable")
check("a VUS well above the ceiling is flagged impossible",
      audit(0.62, D)["verdict"] == "impossible")
check("an honest VUS is flagged plausible", audit(0.19, D)["verdict"] == "plausible")


# ======================================================================================
head("7  The full report, and the headline figure")
# ======================================================================================
rep = pricetest_report(arm_b[sub], P[sub], D, classes=[0, 1, 2], resolution=200,
                       marker=s_b[sub])
print(format_pricetest_report(rep))
plot_ceiling_vs_elasticity(path=FIG / "09_pricetest_ceiling.png",
                           observed=(arc_e, surf.vus))
print(f"\nwrote {FIG / '09_pricetest_ceiling.png'}")


# ======================================================================================
head("SUMMARY")
# ======================================================================================
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("All checks passed.")
