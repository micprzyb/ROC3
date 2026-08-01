"""
The payoff question: can an OBSERVABLE objective reach the headroom that log-loss cannot?

Experiment 20 established the diagnosis.  Tuning the flexible models on held-out log-loss
made their profit regret *worse* -- gbm(v1) by 7-9, gbm(v2) by 5-7 -- while an oracle
objective would have improved them by 6-19.  So the capacity is there and the selection
criterion is throwing it away.  Exactly the retail R-loss finding, in a new setting.

The fix has to be an objective that is (a) computable without the truth and (b) aligned
with the decision.  Randomisation supplies one: with known ``pi`` the value of the model's
implied pricing policy can be estimated unbiasedly from held-out quotes.  Two forms:

``ipw_policy``   inverse-propensity value.  Unbiased, and noisy: a policy recommending the
                 side arms is scored on a tenth of the traffic.
``aipw_policy``  the augmented version.  Same target, an outcome model absorbing the
                 predictable part of the response, so much lower variance -- which is the
                 whole point at 0.1/0.8/0.1.

Four objectives, identical budgets, graded on profit regret which none of them can see
except the oracle.
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

import lightgbm as lgb                                                    # noqa: E402
from elasticity_lab.insurance import (AIPWArmEffects, AnchoredTLearner,   # noqa: E402
                                      ConversionGBM, ConversionGBMv2, ConversionGLM,
                                      MotorQuoteConfig, N_THREADS, QUOTE_FEATURES,
                                      TLearnerConversion, lerner_price, make_quote_panel,
                                      policy_profit, profit_regret)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
f3 = lambda v: f"{v:,.3f}"                                                # noqa: E731
OUT = "experiments/out"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LINKS = ("logit", "kinked")


def build(link, seed=0, width=0.10):
    cfg = MotorQuoteConfig(n_quotes=N, seed=seed, link=link,
                           arm_multipliers=(1 - width, 1.0, 1 + width),
                           arm_probs=(0.1, 0.8, 0.1))
    d = make_quote_panel(cfg)
    a, b = int(0.55 * len(d)), int(0.75 * len(d))
    fr = [d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy()]
    for f in fr:
        f.attrs.update(d.attrs)
    return (d, *fr)


def aipw_arm_values(frame, seed=0):
    """Cross-fitted AIPW estimate of each arm's conversion, per quote.

    ``mu_k = m_k(x) + 1[A=k]/pi_k * (Y - m_k(x))``.  Unbiased whatever ``m_k`` is, because
    the propensity is known exactly, and far lower variance than raw IPW because ``m_k``
    absorbs the predictable part of ``Y``.  Fitted once per frame and reused by every
    candidate, so no candidate can win by being paired with a lucky nuisance fit -- the
    same rule the retail R-loss study enforced.
    """
    X = frame[QUOTE_FEATURES].to_numpy(float)
    y = frame["converted"].to_numpy(float)
    arm = frame["arm"].to_numpy()
    pi = np.asarray(frame.attrs["arm_probs"], float)
    pi = pi / pi.sum()
    K = len(pi)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 4, len(frame))
    m = np.zeros((len(frame), K))
    for f in range(4):
        tr, te = fold != f, fold == f
        for k in range(K):
            sel = tr & (arm == k)
            if sel.sum() < 50:
                m[te, k] = y[tr].mean()
                continue
            g = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=N_THREADS,
                                   force_col_wise=True, random_state=seed,
                                   n_estimators=200, num_leaves=31,
                                   min_child_samples=200)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.fit(X[sel], y[sel].astype(int))
            m[te, k] = g.predict_proba(X[te])[:, 1]
    mu = m.copy()
    for k in range(K):
        hit = arm == k
        mu[hit, k] += (y[hit] - m[hit, k]) / pi[k]
    return mu


def implied_arms(m, frame):
    price0 = np.exp(frame["log_technical_premium"].to_numpy(float))
    cost = frame["claims_cost"].to_numpy(float)
    mult = np.asarray(frame.attrs["arm_multipliers"], float)
    e = np.asarray(m.elasticity(frame), float)
    p_star = lerner_price(e, cost, price0=price0)
    return np.argmin(np.abs(np.log(p_star)[:, None] - np.log(price0)[:, None]
                            - np.log(mult)[None, :]), axis=1), e


def score_all(m, frame, mu):
    """Every objective at once.  Only `regret` needs the truth."""
    arms, e = implied_arms(m, frame)
    price0 = np.exp(frame["log_technical_premium"].to_numpy(float))
    cost = frame["claims_cost"].to_numpy(float)
    mult = np.asarray(frame.attrs["arm_multipliers"], float)
    price = price0 * mult[arms]

    pp = policy_profit(frame, arms)
    # AIPW policy value: the model's chosen arm's AIPW conversion, priced and costed
    aipw_val = float(np.mean(mu[np.arange(len(frame)), arms] * (price - cost)))
    try:
        s = np.clip(np.asarray(m.conversion(frame), float), 1e-9, 1 - 1e-9)
        y = frame["converted"].to_numpy(int)
        ll = float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))
    except Exception:
        ll = np.nan
    return {"logloss": ll,
            "ipw_policy": -pp["profit_per_quote"],        # minimise
            "aipw_policy": -aipw_val,                     # minimise
            "regret": profit_regret(frame, e, price0=price0)["regret_mean"],
            "eps_rmse": float(np.sqrt(np.mean(
                (e - frame["true_elasticity_control"].to_numpy(float)) ** 2))),
            "pct_negative": float(np.mean(e < 0))}


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
          "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0)}
    # NB monotone is NOT offered to the search.  In experiment 20 it was, and a log-loss
    # objective switched it OFF -- unconstrained fits log-loss better -- after which 31-37%
    # of elasticities came out negative.  A constraint that encodes a law of demand is not
    # a hyperparameter to be traded against predictive fit.
    p["monotone"] = True
    return p


def sp_anchored(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 127, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 50, 2000, log=True),
            "side_n_estimators": t.suggest_int("side_n_estimators", 20, 400, step=20),
            "side_num_leaves": t.suggest_int("side_num_leaves", 2, 63, log=True),
            "side_learning_rate": t.suggest_float("side_learning_rate", 0.01, 0.2,
                                                  log=True),
            "monotone": True}


ZOO = {"gbm (v1)": (ConversionGBM, sp_gbm),
       "gbm (v2)": (ConversionGBMv2, sp_gbm_v2),
       "tlearner (anchored)": (AnchoredTLearner, sp_anchored)}
OBJECTIVES = ["logloss", "ipw_policy", "aipw_policy", "regret"]

rows = []
for link in LINKS:
    d, tr, va, te = build(link)
    mu_va, mu_te = aipw_arm_values(va), aipw_arm_values(te)
    print(f"\n### link = {link}", flush=True)
    for name, (cls, space) in ZOO.items():
        base = score_all(cls().fit(tr), te, mu_te)
        rows.append({"link": link, "model": name, "select_on": "untuned", **base})
        for obj in OBJECTIVES:
            t0 = time.time()
            best = {"v": np.inf, "p": {}}

            def f(t, cls=cls, space=space, obj=obj):
                p = space(t)
                try:
                    g = score_all(cls(**p).fit(tr), va, mu_va)
                except Exception:
                    return float("inf")
                v = g[obj]
                if np.isfinite(v) and v < best["v"]:
                    best.update(v=v, p=p)
                return v if np.isfinite(v) else float("inf")

            st = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=0, multivariate=True))
            st.optimize(f, n_trials=N_TRIALS, catch=(Exception,))
            try:
                g = score_all(cls(**best["p"]).fit(tr), te, mu_te)
            except Exception as ex:
                print(f"   {name} / {obj}: FAILED {type(ex).__name__}", flush=True)
                continue
            rows.append({"link": link, "model": name, "select_on": obj, **g})
            print(f"   {name:20s} select on {obj:12s} -> regret {g['regret']:7.3f}  "
                  f"eps_rmse {g['eps_rmse']:6.3f}  [{time.time()-t0:4.0f}s]", flush=True)

t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/21_observable_objective.csv", index=False)

print("\n" + "=" * 112)
print("PROFIT REGRET BY SELECTION OBJECTIVE  (lower is better; 'regret' is the oracle bound)")
print("=" * 112)
for link in LINKS:
    print(f"\n--- {link} ---")
    p = t[t.link == link].pivot(index="model", columns="select_on", values="regret")
    cols = [c for c in ["untuned", "logloss", "ipw_policy", "aipw_policy", "regret"]
            if c in p.columns]
    print(p[cols].to_string(float_format=f3))

print("\n" + "=" * 112)
print("HOW MUCH OF THE ORACLE HEADROOM DOES EACH OBSERVABLE OBJECTIVE RECOVER?")
print("=" * 112)
# Reported as DIFFERENCES in regret, not as a percentage of headroom.  The ratio version
# divides by (untuned - oracle), which is near zero or negative for some model/link pairs,
# and produced meaningless figures like 639% and -355%.  A ratio needs a denominator that
# is reliably positive and this one is not.
rec = []
for link in LINKS:
    p = t[t.link == link].pivot(index="model", columns="select_on", values="regret")
    for mdl in p.index:
        for obj in ("logloss", "ipw_policy", "aipw_policy"):
            if obj not in p.columns:
                continue
            rec.append({"link": link, "model": mdl, "objective": obj,
                        "vs_untuned": float(p.loc[mdl, "untuned"] - p.loc[mdl, obj]),
                        "vs_oracle": float(p.loc[mdl, obj] - p.loc[mdl, "regret"])})
r = pd.DataFrame(rec)
print("vs_untuned: regret saved against not tuning at all (positive = tuning helped)")
print(r.pivot_table(index="objective", columns="link", values="vs_untuned")
      .to_string(float_format=f3))
print("\nvs_oracle: regret still on the table against an oracle objective (lower is better)")
print(r.pivot_table(index="objective", columns="link", values="vs_oracle")
      .to_string(float_format=f3))
r.to_csv(f"{OUT}/21_headroom.csv", index=False)
print(f"\nwrote {OUT}/21_observable_objective.csv")
