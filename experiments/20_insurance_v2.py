"""
Phase A-D of PLAN_INSURANCE_V2: is the GLM's win real, or was everything else neglected?

  A  the three diagnosed bugs (treatment feature, logit-space derivative, market position)
     plus a fair parametric baseline with interactions and splines
  B  the missing models: AIPW arm effects, an anchored T-learner
  C  hyperparameter optimisation, which nothing in experiments 18-19 had
  D  the same comparison under MISSPECIFIED demand, where the GLM loses its free lunch

Reported per data-generating process, never pooled: the whole point is that the answer
depends on whether the parametric form happens to be right.
"""
import json
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from elasticity_lab.insurance import (AIPWArmEffects, AnchoredTLearner,   # noqa: E402
                                      ConversionGBM, ConversionGBMv2, ConversionGLM,
                                      ConversionGLMv2, GLMThenGBM, MotorQuoteConfig,
                                      TLearnerConversion, lerner_price,
                                      make_quote_panel, policy_profit, profit_regret)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
f3 = lambda v: f"{v:,.3f}"                                                # noqa: E731
OUT = "experiments/out"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LINKS = ("logit", "probit", "kinked", "mixed")
WIDTH = 0.10                                   # the live test's arms


def build(link, seed=0, width=WIDTH, alloc=(0.1, 0.8, 0.1)):
    cfg = MotorQuoteConfig(n_quotes=N, seed=seed, link=link,
                           arm_multipliers=(1 - width, 1.0, 1 + width), arm_probs=alloc)
    d = make_quote_panel(cfg)
    a, b = int(0.55 * len(d)), int(0.75 * len(d))
    tr, va, te = d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()
    for f in (tr, va, te):
        f.attrs.update(d.attrs)
    return d, tr, va, te


def grade(m, frame):
    """Everything, on one frame.  Selection uses only observable quantities."""
    e = np.asarray(m.elasticity(frame), float)
    truth = frame["true_elasticity_control"].to_numpy(float)
    price0 = np.exp(frame["log_technical_premium"].to_numpy(float))
    cost = frame["claims_cost"].to_numpy(float)
    mult = np.asarray(frame.attrs["arm_multipliers"], float)
    reg = profit_regret(frame, e, price0=price0)
    arms = np.argmin(np.abs(np.log(lerner_price(e, cost, price0=price0))[:, None]
                            - np.log(price0)[:, None] - np.log(mult)[None, :]), axis=1)
    pp = policy_profit(frame, arms)
    out = {"eps_rmse": float(np.sqrt(np.mean((e - truth) ** 2))),
           "eps_bias": float(e.mean() - truth.mean()),
           "eps_corr": float(np.corrcoef(e, truth)[0, 1]) if e.std() > 1e-9 else np.nan,
           "regret": reg["regret_mean"], "pct_negative": float(np.mean(e < 0)),
           "policy_profit": pp["profit_per_quote"], "policy_se": pp["se"]}
    try:                                       # observable, so usable for selection
        s = np.clip(np.asarray(m.conversion(frame), float), 1e-9, 1 - 1e-9)
        y = frame["converted"].to_numpy(int)
        out["logloss"] = float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))
    except Exception:
        out["logloss"] = np.nan
    return out


# ------------------------------------------------------------------ search spaces (I5)
def sp_glm(t):
    return {"C": t.suggest_float("C", 1e-3, 1e3, log=True)}


def sp_glm_v2(t):
    return {"C": t.suggest_float("C", 1e-3, 1e3, log=True),
            "n_knots": t.suggest_int("n_knots", 3, 8)}


def sp_gbm(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 1200, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 255, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 20, 2000, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "delta": t.suggest_float("delta", 0.01, 0.5, log=True)}


def sp_gbm_v2(t):
    p = sp_gbm(t)
    p |= {"subsample": t.suggest_float("subsample", 0.5, 1.0),
          "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
          "monotone": t.suggest_categorical("monotone", [True, False]),
          "use_price_level": t.suggest_categorical("use_price_level", [True, False])}
    return p


def sp_aipw(t):
    return {"n_folds": t.suggest_int("n_folds", 2, 5),
            "n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 127, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 50, 2000, log=True),
            "final_num_leaves": t.suggest_int("final_num_leaves", 4, 63, log=True),
            "final_n_estimators": t.suggest_int("final_n_estimators", 50, 400, step=50),
            "monotone": t.suggest_categorical("monotone", [True, False])}


def sp_anchored(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 127, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 50, 2000, log=True),
            "side_n_estimators": t.suggest_int("side_n_estimators", 20, 400, step=20),
            "side_num_leaves": t.suggest_int("side_num_leaves", 2, 63, log=True),
            "side_learning_rate": t.suggest_float("side_learning_rate", 0.01, 0.2,
                                                  log=True),
            "monotone": t.suggest_categorical("monotone", [True, False])}


def sp_tlearner(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 4, 127, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 50, 3000, log=True)}


ZOO = {
    "glm (v1)": (ConversionGLM, sp_glm),
    "glm+interactions (v2)": (ConversionGLMv2, sp_glm_v2),
    "gbm (v1)": (ConversionGBM, sp_gbm),
    "gbm (v2: I1+I2+I3+I4)": (ConversionGBMv2, sp_gbm_v2),
    "tlearner (v1)": (TLearnerConversion, sp_tlearner),
    "tlearner (anchored)": (AnchoredTLearner, sp_anchored),
    "aipw arm effects": (AIPWArmEffects, sp_aipw),
}

#: Selection objective.  Held-out log-loss is the only fully observable score every model
#: exposes; regret is the GRADER and is never used to choose.  This mirrors the retail
#: study's discipline: tune on what you could actually see in production.
OBJECTIVE = "logloss"


def tune(cls, space, tr, va, n_trials, seed=0, objective=None):
    best = {"score": np.inf, "params": {}, "n_ok": 0}

    def obj(t):
        p = space(t)
        try:
            m = cls(**p).fit(tr)
            g = grade(m, va)
        except Exception:
            return float("inf")
        v = g[objective or OBJECTIVE]
        if not np.isfinite(v):
            return float("inf")
        best["n_ok"] += 1
        if v < best["score"]:
            best.update(score=v, params=p)
        return v

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    st.optimize(obj, n_trials=n_trials, catch=(Exception,))
    return best


# =======================================================================================
print("=" * 118)
print(f"TUNED COMPARISON ACROSS FOUR DEMAND SHAPES   (n={N:,}, {N_TRIALS} trials/model)")
print("=" * 118)
rows, untuned_rows, params_log = [], [], {}
for link in LINKS:
    d, tr, va, te = build(link)
    print(f"\n### link = {link}   conversion {d.converted.mean():.4f}   "
          f"true eps {te.true_elasticity_control.mean():+.3f} "
          f"(sd {te.true_elasticity_control.std():.3f})", flush=True)
    for name, (cls, space) in ZOO.items():
        t0 = time.time()
        try:
            m0 = cls().fit(tr)
            g0 = grade(m0, te)
        except Exception as ex:
            print(f"  {name:24s} UNTUNED FAILED {type(ex).__name__}: {ex}", flush=True)
            continue
        untuned_rows.append({"link": link, "model": name, "tuned": False,
                                 "select_on": "none", **g0})
        # Two searches, identical budgets.  `logloss` is what you could actually select on
        # in production; `regret` needs the oracle and is the unachievable BOUND.  The gap
        # between them says how much of the available gain an honest objective can reach --
        # the same discipline as the retail R-loss study.
        out = {}
        for obj, tag in (("logloss", "tuned"), ("regret", "oracle_tuned")):
            b = tune(cls, space, tr, va, N_TRIALS, objective=obj)
            try:
                g1 = grade(cls(**b["params"]).fit(tr), te)
            except Exception as ex:
                print(f"  {name:24s} {tag} FAILED {type(ex).__name__}: {ex}", flush=True)
                continue
            rows.append({"link": link, "model": name, "tuned": tag == "tuned",
                         "select_on": obj, **g1})
            params_log[f"{link}:{name}:{obj}"] = b["params"]
            out[obj] = g1
        if "logloss" in out:
            print(f"  {name:24s} untuned regret {g0['regret']:7.3f}  ->  "
                  f"tuned-on-logloss {out['logloss']['regret']:7.3f}  "
                  f"(oracle bound {out.get('regret', {}).get('regret', float('nan')):7.3f})"
                  f"   [{time.time()-t0:5.0f}s]", flush=True)

t = pd.DataFrame(rows + untuned_rows)
t.to_csv(f"{OUT}/20_insurance_v2.csv", index=False)
with open(f"{OUT}/20_best_params.json", "w") as f:
    json.dump(params_log, f, indent=2, default=str)

tt = t[t.select_on == "logloss"]        # the achievable one
print("\n" + "=" * 118)
print("ELASTICITY RMSE, TUNED, BY DEMAND SHAPE  (lower is better; the GLM is only")
print("correctly specified under 'logit')")
print("=" * 118)
print(tt.pivot(index="model", columns="link", values="eps_rmse").to_string(float_format=f3))
print("\nPROFIT REGRET, TUNED")
print(tt.pivot(index="model", columns="link", values="regret").to_string(float_format=f3))
print("\nRANK CORRELATION WITH THE TRUTH, TUNED")
print(tt.pivot(index="model", columns="link", values="eps_corr").to_string(float_format=f3))
print("\nNEGATIVE-ELASTICITY RATE, TUNED")
print(tt.pivot(index="model", columns="link", values="pct_negative").to_string(float_format=f3))

print("\n" + "=" * 118)
print("WHAT TUNING BOUGHT  (untuned regret - tuned regret; positive = tuning helped)")
print("=" * 118)
piv = t.pivot_table(index="model", columns=["link", "select_on"], values="regret")
gain = pd.DataFrame({lk: piv[(lk, "none")] - piv[(lk, "logloss")] for lk in LINKS
                     if (lk, "none") in piv.columns and (lk, "logloss") in piv.columns})
print(gain.to_string(float_format=f3))
print("\nAND WHAT AN ORACLE OBJECTIVE WOULD HAVE BOUGHT  (untuned - oracle-tuned)")
gain2 = pd.DataFrame({lk: piv[(lk, "none")] - piv[(lk, "regret")] for lk in LINKS
                      if (lk, "none") in piv.columns and (lk, "regret") in piv.columns})
print(gain2.to_string(float_format=f3))
print("\nThe gap between the two tables is what selecting on an observable objective costs.")

print("\n" + "=" * 118)
print("WINNER BY DEMAND SHAPE")
print("=" * 118)
for lk in LINKS:
    g = tt[tt.link == lk]
    if not len(g):
        continue
    print(f"  {lk:8s} best eps_rmse: {g.loc[g.eps_rmse.idxmin(),'model']:24s}"
          f"  best regret: {g.loc[g.regret.idxmin(),'model']:24s}"
          f"  best profit: {g.loc[g.policy_profit.idxmax(),'model']}")
print(f"\nwrote {OUT}/20_insurance_v2.csv and 20_best_params.json")
