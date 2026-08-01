"""
Tune every model on the R-loss and run the full diagnostic battery — on the semi-synthetic
panel (where the verdicts can be checked) and on the real one (where they cannot).

Outputs (experiments/out/):
    12_synthetic_full_battery.csv   every metric, plus the oracle columns
    12_real_full_battery.csv        every metric, real panel
    12_metric_selection_power.csv   which OBSERVABLE metric ranks models like the truth
    12_robustness_across_worlds.csv does the ranking survive a different DGP?
    12_tuned_params.json            the chosen hyperparameters
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from elasticity_lab.bench import real_frame, synth_frame, frame_summary
from elasticity_lab.evaluation import compare_models
from elasticity_lab.models import MODEL_REGISTRY, N_THREADS, build_model
from elasticity_lab.simulate import SyntheticConfig
from elasticity_lab.splits import RollingOriginSplit
from elasticity_lab.tuning import CausalScorer, tune

N_PRODUCTS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 20

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
fmt = lambda v: f"{v:+.4f}"                                             # noqa: E731

print(f"threads={N_THREADS} n_products={N_PRODUCTS} n_trials={N_TRIALS}", flush=True)
syn = synth_frame(N_PRODUCTS, confounding_price=0.6, confounding_demand=0.8)
real = real_frame(N_PRODUCTS)
sp = RollingOriginSplit(n_splits=4, horizon=8, min_train_weeks=30)
syn_scorer, real_scorer = CausalScorer(syn, sp, seed=0), CausalScorer(real, sp, seed=0)

BATTERY = ["name", "avg_elasticity", "blp_avg_elasticity", "eps_sd", "rmse",
           "r2", "rloss_skill",
           "rloss_vs_constant", "beta1", "beta1_t", "implied_spread",
           "heterogeneity_detected",
           "gates_monotone", "gates_spread", "gates_spread_t", "value_model",
           "value_best_uniform", "lift_over_uniform", "pct_priced_up"]
ORACLE = ["oracle_rmse", "oracle_bias", "oracle_corr", "share_of_mse_from_bias"]

tuned_params = {}


def tune_all(df, scorer, tag):
    chosen = {}
    for name in MODEL_REGISTRY:
        t0 = time.time()
        try:
            r = tune(name, df, objective="rloss", n_trials=N_TRIALS, scorer=scorer, seed=0)
        except Exception as ex:
            print(f"  {name:20s} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        chosen[name] = build_model(name, **r["best_params"])
        tuned_params[f"{tag}:{name}"] = r["best_params"]
        print(f"  {name:20s} rloss={r['best_metrics']['rloss']:.4f} "
              f"vs-const={r['best_metrics']['rloss_vs_constant']:+.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return chosen


# ============================================================== semi-synthetic: verifiable
print("\n=== tuning on the semi-synthetic panel (R-loss objective) ===", flush=True)
syn_models = tune_all(syn, syn_scorer, "synthetic")
syn_res = compare_models(syn_models, syn, syn_scorer, fold=-1, truth_col="true_elasticity")
syn_res.to_csv("experiments/out/12_synthetic_full_battery.csv", index=False)
if "error" in syn_res.columns and syn_res.error.notna().any():
    # A silently-NaN row reads as "this model is uninformative" when it actually means
    # "this model crashed".  Say which.
    print("\n!! models that failed to evaluate:")
    print(syn_res[syn_res.error.notna()][["name", "error"]].to_string(index=False))
print("\n" + "=" * 110)
print("SEMI-SYNTHETIC — every metric, including the oracle columns the real panel lacks")
print("=" * 110)
cols = [c for c in BATTERY + ORACLE if c in syn_res.columns]
print(syn_res[cols].sort_values("oracle_rmse").to_string(index=False, float_format=fmt))

# ------------------------------------- which observable metric would have picked correctly?
print("\n" + "=" * 110)
print("WHICH OBSERVABLE METRIC RANKS MODELS THE WAY THE TRUTH DOES?")
print("(Spearman with oracle_rmse, sign-flipped so that + means 'agrees with the truth')")
print("=" * 110)
OBS = {"rmse": +1, "r2": -1, "rloss_skill": -1, "rloss_vs_constant": -1,
       "beta1_t": +1, "gates_spread_t": -1, "lift_over_uniform": -1}
d = syn_res.dropna(subset=["oracle_rmse"])
rows = []
for c, s in OBS.items():
    dd = d.dropna(subset=[c])
    if len(dd) < 4 or dd[c].std() < 1e-12:
        continue
    rows.append({"observable metric": c, "n_models": len(dd),
                 "rho with oracle_rmse":
                     s * dd[[c, "oracle_rmse"]].corr("spearman").iloc[0, 1]})
sel = pd.DataFrame(rows).sort_values("rho with oracle_rmse", ascending=False)
sel.to_csv("experiments/out/12_metric_selection_power.csv", index=False)
print(sel.to_string(index=False, float_format=fmt))

# ======================================================= robustness: does the world matter?
print("\n" + "=" * 110)
print("DOES THE RANKING SURVIVE A DIFFERENT DATA-GENERATING PROCESS?")
print("=" * 110)
WORLDS = {
    "exogenous price": SyntheticConfig(confounding_price=0.0, confounding_demand=0.0),
    "mild confounding": SyntheticConfig(confounding_price=0.35, confounding_demand=0.8),
    "heavy confounding": SyntheticConfig(confounding_price=1.0, confounding_demand=1.2),
    "poisson outcome": SyntheticConfig(confounding_price=0.35, confounding_demand=0.8,
                                       outcome="poisson"),
}
rows = []
for wname, cfg in WORLDS.items():
    w = synth_frame(max(400, N_PRODUCTS // 3), config=cfg)
    for mname in MODEL_REGISTRY:
        try:
            m = build_model(mname).fit(w)
            e = np.asarray(m.elasticity(w), float)
        except Exception:
            continue
        t = w.true_elasticity.to_numpy(float)
        rows.append({"world": wname, "model": mname,
                     "bias": float(e.mean() - t.mean()),
                     "oracle_rmse": float(np.sqrt(np.mean((e - t) ** 2)))})
    print(f"  {wname} done", flush=True)
rob = pd.DataFrame(rows)
rob.to_csv("experiments/out/12_robustness_across_worlds.csv", index=False)
print("\nbias by world (untuned defaults):")
print(rob.pivot(index="model", columns="world", values="bias")
         .to_string(float_format=fmt))
print("\nrank by oracle RMSE within each world (1 = best):")
print(rob.pivot(index="model", columns="world", values="oracle_rmse")
         .rank().astype(int).to_string())

# ==================================================================== real panel: no oracle
print("\n=== tuning on the REAL panel (R-loss objective) ===", flush=True)
real_models = tune_all(real, real_scorer, "real")
real_res = compare_models(real_models, real, real_scorer, fold=-1)
real_res.to_csv("experiments/out/12_real_full_battery.csv", index=False)
if "error" in real_res.columns and real_res.error.notna().any():
    print("\n!! models that failed to evaluate:")
    print(real_res[real_res.error.notna()][["name", "error"]].to_string(index=False))
print("\n" + "=" * 110)
print("REAL PANEL — no ground truth; these are the numbers you would actually report")
print("=" * 110)
cols = [c for c in BATTERY if c in real_res.columns]
print(real_res[cols].sort_values("rloss_vs_constant", ascending=False)
      .to_string(index=False, float_format=fmt))

# ------------------------------------------------------------------- the six-step verdict
v = real_res.copy()
v["2_beats_constant"] = v.rloss_vs_constant > 0
v["3_blp_significant"] = v.beta1_t < -1.96
v["4_gates_monotone"] = v.gates_monotone.fillna(False).astype(bool)
v["5_beats_flat_policy"] = v.lift_over_uniform > 0
checks = ["2_beats_constant", "3_blp_significant", "4_gates_monotone",
          "5_beats_flat_policy"]
v["checks_passed"] = v[checks].sum(axis=1)
print("\n" + "=" * 110)
print("THE SELECTION RULE APPLIED (checks 2-5 of docs/ELASTICITY_MODELS.md section 5.1)")
print("A constant-elasticity model failing 3-5 is not a mark against it: it never claimed")
print("heterogeneity.  The checks decide between models that DO make the claim.")
print("=" * 110)
print(v[["name"] + checks + ["checks_passed", "avg_elasticity", "eps_sd"]]
      .sort_values("checks_passed", ascending=False)
      .to_string(index=False, float_format=fmt))

with open("experiments/out/12_tuned_params.json", "w") as f:
    json.dump(tuned_params, f, indent=2, default=str)
print("\nwrote experiments/out/12_*.csv|json")
