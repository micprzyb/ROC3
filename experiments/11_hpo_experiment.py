"""
The central HPO experiment: which objective should you tune an elasticity model on?

For every model in the zoo we run two independent Optuna searches with identical budgets —
one minimising held-out predictive RMSE, one minimising the orthogonal R-loss — and then
grade both winners against the synthetic ground truth, which neither search could see.

Outputs (experiments/out/):
    11_objective_comparison.csv   one row per (model, objective) with all metrics
    11_trial_histories.parquet    every trial of every search, all metrics recorded
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from elasticity_lab.bench import synth_frame, frame_summary
from elasticity_lab.models import MODEL_REGISTRY, N_THREADS
from elasticity_lab.splits import RollingOriginSplit
from elasticity_lab.tuning import CausalScorer, tune

N_PRODUCTS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OBJECTIVES = ("rmse", "rloss")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

print(f"threads={N_THREADS}  n_products={N_PRODUCTS}  n_trials={N_TRIALS}", flush=True)
syn = synth_frame(N_PRODUCTS, confounding_price=0.6, confounding_demand=0.8)
print(frame_summary(syn).to_string(), flush=True)

splitter = RollingOriginSplit(n_splits=4, horizon=8, min_train_weeks=30)
scorer = CausalScorer(syn, splitter, seed=0)
print(f"\n{len(scorer.folds_)} folds; sizes "
      f"{[(len(a), len(b)) for a, b in scorer.folds_]}", flush=True)

rows, hists = [], []
for name in MODEL_REGISTRY:
    for obj in OBJECTIVES:
        t0 = time.time()
        try:
            r = tune(name, syn, objective=obj, n_trials=N_TRIALS, scorer=scorer,
                     truth_col="true_elasticity", seed=0)
        except Exception as ex:
            print(f"{name:20s} {obj:6s} FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        el = time.time() - t0
        hists.append(r["history"].assign(model=name, tuned_on=obj))
        rows.append({"model": name, "tuned_on": obj, **r["best_metrics"],
                     "n_complete": r["n_complete"], "n_pruned": r["n_pruned"],
                     "n_failed": r["n_failed"],
                     "seconds": round(el, 1),
                     "best_params": json.dumps(r["best_params"], default=str)})
        m = r["best_metrics"]
        print(f"{name:20s} {obj:6s} rmse={m.get('rmse', np.nan):7.4f} "
              f"rloss={m.get('rloss', np.nan):8.4f} "
              f"oracle={m.get('oracle', np.nan):7.4f} "
              f"bias={m.get('oracle_bias', np.nan):+7.4f}  "
              f"[{el:5.0f}s, {r['n_complete']} kept / {r['n_pruned']} pruned / {r['n_failed']} failed]", flush=True)

out = pd.DataFrame(rows)
out.to_csv("experiments/out/11_objective_comparison.csv", index=False)
allh = pd.concat(hists, ignore_index=True)
allh.to_parquet("experiments/out/11_trial_histories.parquet", index=False)

# ---------------------------------------------------------------- the headline comparison
print("\n" + "=" * 100)
print("DOES THE TUNING OBJECTIVE CHANGE WHAT YOU GET?")
print("=" * 100)
piv = out.pivot(index="model", columns="tuned_on",
                values=["oracle", "oracle_bias", "rmse"])
piv.columns = [f"{a}[{b}]" for a, b in piv.columns]
piv["oracle_gain_from_rloss"] = piv["oracle[rmse]"] - piv["oracle[rloss]"]
piv["rmse_cost_of_rloss"] = piv["rmse[rloss]"] - piv["rmse[rmse]"]
print(piv.sort_values("oracle[rloss]").to_string(float_format=lambda v: f"{v:+.4f}"))

print("\nAcross the zoo:")
print(f"  models where tuning on R-loss recovered elasticity better : "
      f"{int((piv.oracle_gain_from_rloss > 0).sum())} / {len(piv)}")
print(f"  median improvement in oracle RMSE                         : "
      f"{piv.oracle_gain_from_rloss.median():+.4f}")
print(f"  median predictive RMSE given up to get it                 : "
      f"{piv.rmse_cost_of_rloss.median():+.4f}")

# ------------------------------------------------------- is R-loss a good proxy at all?
print("\n" + "=" * 100)
print("IS R-LOSS A USABLE PROXY FOR THE UNKNOWABLE ORACLE?")
print("(rank correlation across all trials of a search — the criterion HPO actually needs)")
print("=" * 100)
cc = []
for name, g in allh[allh.get("failed", 0) == 0].groupby("model"):
    g = g.dropna(subset=["oracle", "rloss", "rmse"])
    if len(g) < 8 or g.oracle.std() < 1e-9:
        continue
    cc.append({"model": name, "n_trials": len(g),
               "spearman(rloss, oracle)": g[["rloss", "oracle"]].corr("spearman").iloc[0, 1],
               "spearman(rmse, oracle)": g[["rmse", "oracle"]].corr("spearman").iloc[0, 1]})
cct = pd.DataFrame(cc)
print(cct.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
print("\nPositive = the objective ranks hyperparameters the same way the truth does.")
print("This is the property that matters: an objective can be a poor absolute proxy and")
print("still be a good selector, and vice versa.")
cct.to_csv("experiments/out/11_objective_rank_correlation.csv", index=False)
print("\nwrote experiments/out/11_*.csv|parquet")
