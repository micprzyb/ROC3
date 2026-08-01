"""Smoke-test the seven estimators: do they fit, predict, and recover a known truth?"""
import time
import numpy as np
import pandas as pd

from elasticity_lab.data import get_panel
from elasticity_lab.features import add_features, OUTCOME
from elasticity_lab.simulate import SyntheticConfig, make_synthetic_panel
from elasticity_lab.models import MODEL_REGISTRY, build_model, perturb_price

pd.set_option("display.width", 200)


def evaluate(df, truth_col=None, tag=""):
    rows = []
    for name in MODEL_REGISTRY:
        t0 = time.time()
        try:
            m = build_model(name).fit(df)
            yhat = m.predict_log_qty(df)
            e = m.elasticity(df)
            row = {
                "model": name,
                "avg_eps": m.average_elasticity,
                "eps_sd": float(np.std(e)),
                "in_rmse": float(np.sqrt(np.mean((yhat - df[OUTCOME]) ** 2))),
                "secs": round(time.time() - t0, 1),
            }
            if truth_col is not None:
                t = df[truth_col].to_numpy(float)
                row["true_mean"] = float(t.mean())
                row["bias"] = float(np.mean(e) - t.mean())
                row["eps_rmse"] = float(np.sqrt(np.mean((e - t) ** 2)))
                row["eps_corr"] = (float(np.corrcoef(e, t)[0, 1])
                                   if np.std(e) > 1e-9 else np.nan)
            rows.append(row)
        except Exception as ex:                      # keep going, report the failure
            rows.append({"model": name, "avg_eps": np.nan, "secs": round(time.time()-t0, 1),
                         "error": f"{type(ex).__name__}: {ex}"})
    t = pd.DataFrame(rows)
    print(f"\n=== {tag} ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return t


# ---------------------------------------------------------------------------- real data
panel = get_panel()
real = add_features(panel)
print(f"real panel {real.shape}, {real.stock_code.nunique()} products, "
      f"{real.week.nunique()} weeks")

# perturb_price must move every price-derived column together
chk = perturb_price(real.head(1000), 0.10)
assert np.allclose(chk.log_price - real.head(1000).log_price, 0.10)
assert np.allclose(chk.rel_price - real.head(1000).rel_price, 0.10)
assert np.allclose(chk.price_change - real.head(1000).price_change, 0.10)
print("perturb_price moves log_price, rel_price and price_change together: OK")

evaluate(real, tag="REAL DATA (no ground truth — only internal consistency)")

# ------------------------------------------------------------------- synthetic, gradeable
cfg = SyntheticConfig(confounding_price=0.6, confounding_demand=0.8, seed=0)
syn = make_synthetic_panel(panel, cfg)   # already a feature frame; do not re-engineer
print(f"\nsynthetic frame {syn.shape}; true mean elasticity "
      f"{syn.true_elasticity.mean():+.4f}")
tab = evaluate(syn, truth_col="true_elasticity", tag="SYNTHETIC (graded on recovery)")
tab.to_csv("experiments/out/10_model_zoo_smoke.csv", index=False)
