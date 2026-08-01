"""
Widening the side arms, and grading models on profit rather than on squared error.

Two questions, and they are coupled.

  1  How wide do the arms have to be before personalisation is detectable at all?
     At +/-10% the best targeted policy beat the best flat one by 0.06 standard errors --
     which is to say the experiment could not see it.  Widening buys power.

  2  Squared error in the elasticity is the wrong loss for a pricing model, because the
     optimal price moves at rate -c/(eps-1)^2 and profit is locally quadratic, so regret
     scales as (d eps)^2 / (eps-1)^4.  An error at eps = 1.5 costs 256x what the same
     error costs at eps = 3.  This runs both losses side by side and asks which ranking
     a decision-maker should act on.

Widening is not free -- the side arms are deliberately mispriced -- so the cost of the
test is reported next to the information it buys.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from elasticity_lab.insurance import (ArmPosteriorModel, ConversionGBM,   # noqa: E402
                                      ConversionGLM, GLMThenGBM, MotorQuoteConfig,
                                      TLearnerConversion, experiment_cost,
                                      lerner_price, make_quote_panel, optimal_price,
                                      policy_profit, profit_regret, profit_weight,
                                      realised_elasticity_by_bin)

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
f3 = lambda v: f"{v:,.3f}"                                                # noqa: E731
OUT = "experiments/out"

WIDTHS = [0.05, 0.10, 0.20, 0.30, 0.50]
N = 300_000


def build(w, alloc=(0.1, 0.8, 0.1), seed=0):
    cfg = MotorQuoteConfig(n_quotes=N, seed=seed,
                           arm_multipliers=(1 - w, 1.0, 1 + w), arm_probs=alloc)
    d = make_quote_panel(cfg)
    n_tr = int(0.7 * len(d))
    tr, va = d.iloc[:n_tr].copy(), d.iloc[n_tr:].copy()
    for f in (tr, va):
        f.attrs.update(d.attrs)
    return d, tr, va


def models():
    return {"conversion_glm": ConversionGLM(),
            "conversion_gbm": ConversionGBM(),
            "tlearner": TLearnerConversion(),
            "arm_posterior": ArmPosteriorModel(balance="none", constraint="isotonic"),
            "glm_then_gbm": GLMThenGBM(stage2_sees_price=False)}


# ============================================================ 0. the weight function
print("=" * 112)
print("0.  WHERE ELASTICITY ACCURACY ACTUALLY MATTERS")
print("=" * 112)
grid = np.array([1.05, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
w = profit_weight(grid)
wt = pd.DataFrame({"elasticity": [f"{x:.2f}" for x in grid],
                   "relative cost of a unit error": w / w[grid == 3.0]})
print(wt.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
print("""
  Normalised so eps = 3 is 1.0.  Squared error treats every row in this table alike; the
  decision does not.  Note also that 22% of this book sits BELOW eps = 1, where the model
  says 'charge more' without bound and the Lerner price is undefined -- those rows need a
  cap, not a better estimate.""")

# ================================================= 1. the sweep
print("=" * 112)
print("1.  ARM WIDTH: INFORMATION BOUGHT, AND WHAT IT COSTS")
print("=" * 112)
rows, gates_rows = [], []
for wdt in WIDTHS:
    d, tr, va = build(wdt)
    mult = np.asarray(d.attrs["arm_multipliers"], float)
    truth = va["true_elasticity_control"].to_numpy(float)
    price0 = np.exp(va["log_technical_premium"].to_numpy(float))
    cost = va["claims_cost"].to_numpy(float)
    ec = experiment_cost(d)

    flats = {f"all_{k}": policy_profit(va, np.full(len(va), k))
             for k in range(len(mult))}
    best_flat = max(v["profit_per_quote"] for v in flats.values())

    for name, m in models().items():
        try:
            m.fit(tr)
            e = np.asarray(m.elasticity(va), float)
        except Exception as ex:
            print(f"  w={wdt} {name}: {type(ex).__name__}: {ex}", flush=True)
            continue
        reg = profit_regret(va, e, price0=price0)
        p_star = lerner_price(e, cost, price0=price0)
        arms = np.argmin(np.abs(np.log(p_star)[:, None] - np.log(price0)[:, None]
                                - np.log(mult)[None, :]), axis=1)
        pp = policy_profit(va, arms)
        rows.append({
            "arm_width": wdt, "model": name,
            "eps_rmse": float(np.sqrt(np.mean((e - truth) ** 2))),
            "eps_bias": float(e.mean() - truth.mean()),
            "eps_corr": float(np.corrcoef(e, truth)[0, 1]) if e.std() > 1e-9 else np.nan,
            "regret": reg["regret_mean"],
            "pct_of_oracle_profit": reg["pct_of_oracle_profit"],
            "profit": pp["profit_per_quote"], "se": pp["se"],
            "lift_over_flat": pp["profit_per_quote"] - best_flat,
            "lift_in_se": (pp["profit_per_quote"] - best_flat) / max(pp["se"], 1e-9),
            "test_cost_per_quote": ec["cost_per_quote"],
            "best_flat": best_flat})
        g = realised_elasticity_by_bin(va, e, n_bins=5)
        gates_rows.append({"arm_width": wdt, "model": name,
                           "gates_monotone": g.attrs["monotone"],
                           "gates_span_realised": float(g.eps_realised.iloc[-1]
                                                        - g.eps_realised.iloc[0]),
                           "gates_span_predicted": float(g.eps_predicted.iloc[-1]
                                                         - g.eps_predicted.iloc[0])})
    print(f"  width +/-{wdt:.0%} done   (test costs {ec['cost_per_quote']:+.2f}/quote, "
          f"{ec['cost_pct']:+.1%} of control profit)", flush=True)

t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/19_arm_width.csv", index=False)
g = pd.DataFrame(gates_rows)

print("\n--- elasticity accuracy by arm width ---")
print(t.pivot(index="model", columns="arm_width", values="eps_rmse")
      .to_string(float_format=f3))
print("\n--- rank correlation with the truth ---")
print(t.pivot(index="model", columns="arm_width", values="eps_corr")
      .to_string(float_format=f3))
print("\n--- PROFIT REGRET per quote (the loss that matters; lower is better) ---")
print(t.pivot(index="model", columns="arm_width", values="regret")
      .to_string(float_format=f3))
print("\n--- lift over the best FLAT policy, in standard errors ---")
print(t.pivot(index="model", columns="arm_width", values="lift_in_se")
      .to_string(float_format=f3))

cost = t.groupby("arm_width")[["test_cost_per_quote", "best_flat"]].first()
print("\n--- what the test costs ---")
print(cost.to_string(float_format=f3))

# ============================================ 2. does the loss change the ranking?
print("\n" + "=" * 112)
print("2.  SQUARED ERROR vs PROFIT REGRET: DO THEY RANK MODELS THE SAME WAY?")
print("=" * 112)
cmp_rows = []
for wdt, gg in t.groupby("arm_width"):
    gg = gg.dropna(subset=["eps_rmse", "regret"])
    if len(gg) < 3:
        continue
    cmp_rows.append({
        "arm_width": wdt,
        "best_by_eps_rmse": gg.loc[gg.eps_rmse.idxmin(), "model"],
        "best_by_regret": gg.loc[gg.regret.idxmin(), "model"],
        "best_by_profit": gg.loc[gg.profit.idxmax(), "model"],
        "rho(eps_rmse, regret)": gg[["eps_rmse", "regret"]].corr("spearman").iloc[0, 1],
        "agree": gg.loc[gg.eps_rmse.idxmin(), "model"] == gg.loc[gg.regret.idxmin(), "model"]})
c = pd.DataFrame(cmp_rows)
print(c.to_string(index=False, float_format=f3))
c.to_csv(f"{OUT}/19_loss_comparison.csv", index=False)

# ============================================ 3. allocation: is 0.1/0.8/0.1 the problem?
print("\n" + "=" * 112)
print("3.  IS IT THE WIDTH OR THE ALLOCATION?")
print("=" * 112)
alloc_rows = []
for alloc, lbl in [((0.1, 0.8, 0.1), "0.1/0.8/0.1 (live test)"),
                   ((0.2, 0.6, 0.2), "0.2/0.6/0.2"),
                   ((1 / 3, 1 / 3, 1 / 3), "balanced")]:
    for wdt in (0.10, 0.30):
        d, tr, va = build(wdt, alloc=alloc)
        mult = np.asarray(d.attrs["arm_multipliers"], float)
        truth = va["true_elasticity_control"].to_numpy(float)
        price0 = np.exp(va["log_technical_premium"].to_numpy(float))
        cst = va["claims_cost"].to_numpy(float)
        ec = experiment_cost(d)
        best_flat = max(policy_profit(va, np.full(len(va), k))["profit_per_quote"]
                        for k in range(len(mult)))
        m = ConversionGLM().fit(tr)
        e = np.asarray(m.elasticity(va), float)
        p_star = lerner_price(e, cst, price0=price0)
        arms = np.argmin(np.abs(np.log(p_star)[:, None] - np.log(price0)[:, None]
                                - np.log(mult)[None, :]), axis=1)
        pp = policy_profit(va, arms)
        alloc_rows.append({"allocation": lbl, "arm_width": wdt,
                           "eps_rmse": float(np.sqrt(np.mean((e - truth) ** 2))),
                           "regret": profit_regret(va, e, price0=price0)["regret_mean"],
                           "lift_in_se": (pp["profit_per_quote"] - best_flat)
                           / max(pp["se"], 1e-9),
                           "ess": pp["ess"],
                           "test_cost": ec["cost_per_quote"]})
        print(f"  {lbl:24s} w={wdt:.0%} done", flush=True)
a = pd.DataFrame(alloc_rows)
print("\n" + a.to_string(index=False, float_format=f3))
a.to_csv(f"{OUT}/19_allocation.csv", index=False)
print(f"\nwrote {OUT}/19_*.csv")
