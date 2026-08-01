"""
The motor model set, and the two routes to the familiar p_1 < p_2 < p_3 constraint.

Sections
  1  the constraint region really is a triangle, and the gauge maps it to the standard one
  2  weights vs oversampling vs post-hoc transformation -- do they agree on VUS?
  3  the model set, graded against the known elasticity
  4  grouped diagnostics (randomisation makes these assumption-free)
  5  profit: what any of it is worth
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import roc3                                                              # noqa: E402
from elasticity_lab.insurance import (MOTOR_MODELS, ArmPosteriorModel,   # noqa: E402
                                      ConversionGBM, ConversionGLM, GLMThenGBM,
                                      MotorQuoteConfig, TLearnerConversion,
                                      constraint_triangle_vertices, make_quote_panel,
                                      optimal_price, policy_profit,
                                      realised_elasticity_by_bin, to_balanced_gauge,
                                      true_conversion_matrix)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
f4 = lambda v: f"{v:+.4f}"                                               # noqa: E731
OUT = "experiments/out"
FAIL = []


def check(label, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'}  {label}   {detail}", flush=True)
    if not ok:
        FAIL.append(label)


cfg = MotorQuoteConfig(n_quotes=300_000, seed=0)
d = make_quote_panel(cfg)
n_tr = int(0.7 * len(d))
tr, va = d.iloc[:n_tr].copy(), d.iloc[n_tr:].copy()
for f in (tr, va):
    f.attrs.update(d.attrs)
pi = np.asarray(d.attrs["arm_probs"], float)
mult = np.asarray(d.attrs["arm_multipliers"], float)
truth = va["true_elasticity_control"].to_numpy(float)
print(f"{len(d):,} quotes -> train {len(tr):,} / valid {len(va):,}   "
      f"conversion {d.converted.mean():.4f}")
print(f"true elasticity at control: mean {truth.mean():+.4f}, sd {truth.std():.4f}\n")

# ================================================================= 1. the triangle
print("=" * 104)
print("1.  THE ADMISSIBLE REGION IS A TRIANGLE, AND THE GAUGE MAPS IT TO THE STANDARD ONE")
print("=" * 104)
tri = constraint_triangle_vertices(pi)
print("standard ordered chamber {v1>=v2>=v3}      admissible region {b1/pi1>=b2/pi2>=b3/pi3}")
for a, b in zip(tri["standard_chamber"], tri["admissible_region"]):
    print(f"   {np.round(a, 4)}                  ->   {np.round(b, 4)}")
back = to_balanced_gauge(tri["admissible_region"], pi)
check("the gauge sends the admissible region back to the standard chamber",
      np.allclose(back, tri["standard_chamber"]),
      f"max err {np.abs(back - tri['standard_chamber']).max():.2e}")

# how much of the simplex does each occupy?  (area of a triangle in barycentric coords)
area = lambda T: abs(np.linalg.det(np.column_stack([T[1] - T[0], T[2] - T[0]])[:2]))  # noqa
print(f"\n   area(standard chamber)  = {area(tri['standard_chamber']):.4f}")
print(f"   area(admissible region) = {area(tri['admissible_region']):.4f}")
print("   -> at 0.1/0.8/0.1 the admissible set is a different, much smaller triangle.")
print("      Enforcing b1>=b2>=b3 would be enforcing the wrong one.")

# ============================================= 2. weights vs oversample vs transform
print("\n" + "=" * 104)
print("2.  THREE ROUTES TO THE BALANCED COORDINATES")
print("=" * 104)
rows = []
post = {}
for bal in ("none", "weights", "oversample"):
    m = ArmPosteriorModel(balance=bal, constraint="none", seed=0).fit(tr)
    b_bal = m.posterior(va, balanced=True)
    post[bal] = b_bal
    e = m.elasticity(va)
    lv = va["arm"].to_numpy()
    conv = va["converted"].to_numpy() == 1
    vus = roc3.vus_forced_choice(lv[conv], b_bal[conv])
    vus_raw = roc3.vus_forced_choice(lv[conv], m.posterior(va, balanced=False)[conv])
    dec = float(np.mean(np.all(np.diff(b_bal, axis=1) <= 1e-12, axis=1)))
    rows.append({"balance": bal, "mean_eps": e.mean(), "sd_eps": e.std(),
                 "bias": e.mean() - truth.mean(),
                 "rmse": float(np.sqrt(np.mean((e - truth) ** 2))),
                 "corr": float(np.corrcoef(e, truth)[0, 1]),
                 "vus_balanced": vus, "vus_raw": vus_raw,
                 "pct_ordered_in_balanced_coords": dec,
                 "ess_frac": getattr(m, "ess_", 1.0)})
t2 = pd.DataFrame(rows)
print(t2.to_string(index=False, float_format=f4))

check("the gauge leaves VUS unchanged (raw vs balanced coordinates)",
      bool(np.allclose(t2.vus_balanced, t2.vus_raw, atol=1e-12)),
      f"max diff {np.abs(t2.vus_balanced - t2.vus_raw).max():.2e}")
check("weighting DOES change the fit (a different loss gives a different model)",
      abs(t2.loc[0, "mean_eps"] - t2.loc[1, "mean_eps"]) > 1e-6,
      f"none {t2.loc[0,'mean_eps']:+.4f} vs weights {t2.loc[1,'mean_eps']:+.4f}")
print(f"""
  So the user's two routes are exactly as described, and the gauge theorem says why they
  must agree on VUS: the map b -> (b_k/pi_k)/sum is a gauge, and VUS is invariant under it.
  They do NOT agree on the fitted model, because a weighted log-loss is a different
  objective -- weighting spends effective sample size (ess_frac above) to buy coordinates
  the transformation gives for free.""")

# ================================================================= 3. the model set
print("=" * 104)
print("3.  THE MODEL SET")
print("=" * 104)
models = {
    "conversion_glm": ConversionGLM(),
    "conversion_gbm": ConversionGBM(),
    "tlearner": TLearnerConversion(),
    "arm_posterior (weights, isotonic)": ArmPosteriorModel(balance="weights",
                                                           constraint="isotonic"),
    "arm_posterior (none, isotonic)": ArmPosteriorModel(balance="none",
                                                        constraint="isotonic"),
    "arm_posterior (weights, free)": ArmPosteriorModel(balance="weights",
                                                       constraint="none"),
    "glm_then_gbm (price hidden)": GLMThenGBM(stage2_sees_price=False),
    "glm_then_gbm (price shown)": GLMThenGBM(stage2_sees_price=True),
}
rows = []
fitted = {}
for name, m in models.items():
    try:
        m.fit(tr)
    except Exception as ex:
        print(f"  {name}: FAILED {type(ex).__name__}: {ex}", flush=True)
        continue
    fitted[name] = m
    e = np.asarray(m.elasticity(va), float)
    row = {"model": name, "mean_eps": e.mean(), "sd_eps": e.std(),
           "bias": e.mean() - truth.mean(),
           "rmse": float(np.sqrt(np.mean((e - truth) ** 2))),
           "corr": float(np.corrcoef(e, truth)[0, 1]) if e.std() > 1e-9 else np.nan,
           "pct_negative": float(np.mean(e < 0))}
    # conversion accuracy, where the model provides it on the right scale
    try:
        s = np.asarray(m.conversion(va), float)
        y = va["converted"].to_numpy(int)
        row["conv_logloss"] = float(-np.mean(y * np.log(np.clip(s, 1e-9, 1)) +
                                             (1 - y) * np.log(np.clip(1 - s, 1e-9, 1))))
    except NotImplementedError:
        row["conv_logloss"] = np.nan          # the model has no level; see its docstring
    except Exception:
        row["conv_logloss"] = np.nan
    rows.append(row)
t3 = pd.DataFrame(rows).sort_values("rmse")
print(f"true mean elasticity at control = {truth.mean():+.4f}, sd = {truth.std():.4f}\n")
print(t3.to_string(index=False, float_format=f4))
t3.to_csv(f"{OUT}/18_motor_models.csv", index=False)

# ================================================================= 4. grouped diagnostics
print("\n" + "=" * 104)
print("4.  GATES -- REALISED ELASTICITY BY PREDICTED BIN (assumption-free: arms are random)")
print("=" * 104)
for name in ("conversion_glm", "conversion_gbm", "arm_posterior (weights, isotonic)"):
    if name not in fitted:
        continue
    e = np.asarray(fitted[name].elasticity(va), float)
    g = realised_elasticity_by_bin(va, e, n_bins=5)
    print(f"\n--- {name} ---   monotone: {g.attrs['monotone']}")
    print(g[["bin", "n", "eps_predicted", "eps_realised", "se",
             "conv_arm0", "conv_arm2"]].to_string(index=False, float_format=f4))

# ================================================================= 5. profit
print("\n" + "=" * 104)
print("5.  PROFIT -- WHAT THE ELASTICITY IS FOR")
print("=" * 104)
price0 = np.exp(va["log_technical_premium"].to_numpy(float))
cost = va["claims_cost"].to_numpy(float)
rows = []
for lbl, arms in [("all control", np.ones(len(va), int)),
                  ("all cheap", np.zeros(len(va), int)),
                  ("all dear", np.full(len(va), 2))]:
    rows.append({"policy": lbl, **policy_profit(va, arms)})
for name, m in fitted.items():
    e = np.asarray(m.elasticity(va), float)
    # the Lerner rule, discretised onto the three arms the test actually ran
    p_star = optimal_price(lambda p, e=e: e, price0, cost)
    arms = np.argmin(np.abs(np.log(p_star)[:, None] - np.log(price0)[:, None]
                            - np.log(mult)[None, :]), axis=1)
    rows.append({"policy": f"Lerner via {name}", **policy_profit(va, arms)})
t5 = pd.DataFrame(rows).sort_values("profit_per_quote", ascending=False)
best_flat = t5[t5.policy.isin(["all control", "all cheap", "all dear"])].profit_per_quote.max()
t5["lift_over_best_flat"] = t5.profit_per_quote - best_flat
t5["lift_in_se"] = t5.lift_over_best_flat / t5.se
print(t5[["policy", "profit_per_quote", "se", "conversion", "ess",
          "lift_over_best_flat", "lift_in_se", "pct_arm_dear"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print(f"""
  Read the last two columns before the first.  The best FLAT policy earns {best_flat:.2f};
  every targeted policy is compared against that, in units of its own standard error.  With
  only 10% of traffic in each side arm a policy that recommends them is evaluated on about
  9,600 effective quotes, so differences of less than about {2*t5.se.median():.1f} are noise.""")
t5.to_csv(f"{OUT}/18_motor_profit.csv", index=False)

print("\n" + "=" * 104)
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    raise SystemExit(1)
print("all structural checks passed")
