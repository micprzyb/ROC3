"""
elasticity_lab.evaluation
=========================

Scoring elasticity models when the label does not exist.

Nobody observes a customer's elasticity, or a product's.  So "is this model good?" cannot
be answered by comparing a prediction to a target, and the usual regression metrics answer
a different question than the one being asked.  This module implements the three families
of answer that *are* available, in increasing order of how much they tell you:

**1. Predictive metrics** (:func:`predictive_metrics`) — RMSE, MAE, :math:`R^2` on held-out
``log q``.  Necessary but almost useless: a model can nail these while getting the price
derivative completely wrong, which is precisely the failure mode
:mod:`elasticity_lab.tuning` exists to detect.

**2. Grouped identification** (:func:`gates`, :func:`blp_calibration`) — the honest core.
You cannot check one row's elasticity, but you *can* check a group's, because a group has
price variation inside it.  Sort rows by predicted elasticity, cut into bins, and estimate
each bin's elasticity orthogonally from the data.  If the model's ranking is real, those
estimates increase across bins.  This is the continuous-treatment analogue of the
GATES/BLP machinery of Chernozhukov, Demirer, Duflo & Fernández-Val (2018), and it is the
only evidence about heterogeneity that does not assume the answer.

**3. Decision value** (:func:`policy_value`) — the metric a business actually cares about.
Turn the elasticity into the pricing decision it implies, and estimate what that decision
earns, using group-level identification for the counterfactual.  Always reported against
the best *uniform* price change, because a heterogeneous model that cannot beat one flat
number has bought nothing.

On synthetic data :func:`oracle_metrics` adds the comparison the others exist to
approximate, which is how the observable metrics above get validated in the first place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OUTCOME, TREATMENT
from .tuning import CausalScorer

__all__ = ["predictive_metrics", "gates", "blp_calibration", "policy_value",
           "oracle_metrics", "evaluate_model", "compare_models", "elasticity_summary"]


# =======================================================================================
# 1. predictive
# =======================================================================================
def predictive_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """Held-out fit on the log scale, plus the level scale that money lives on."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    ss = float(np.sum((y - y.mean()) ** 2))
    lvl, lvl_hat = np.exp(y), np.exp(yhat)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan,
            "level_mape": float(np.mean(np.abs(lvl_hat - lvl) / np.maximum(lvl, 1e-9))),
            "bias": float(err.mean())}


# =======================================================================================
# 2. grouped identification
# =======================================================================================
def _group_theta(ry, rt, mask):
    """Orthogonal slope inside a subgroup, with its standard error."""
    y, t = ry[mask], rt[mask]
    stt = float(np.dot(t, t))
    if stt <= 1e-12 or mask.sum() < 5:
        return np.nan, np.nan
    th = float(np.dot(t, y) / stt)
    psi = t * (y - th * t)
    return th, float(np.sqrt((psi ** 2).sum()) / stt)


def gates(ry: np.ndarray, rt: np.ndarray, eps_hat: np.ndarray, *,
          n_groups: int = 5) -> pd.DataFrame:
    """**G**\\ rouped **A**\\ verage **T**\\ reatment **E**\\ ffects, for a continuous price.

    Rows are sorted by predicted elasticity and cut into ``n_groups`` bins.  Within each
    bin the elasticity is re-estimated *from the data*, orthogonally::

        eps_g  =  − <rt, ry>_g / <rt, rt>_g

    The predicted column is the model's own average for the bin; the estimated column is
    what the data say.  A model whose ranking is real produces an estimated column that
    rises with the predicted one.  A model whose "heterogeneity" is noise produces a flat
    estimated column no matter how much its predictions vary — and that is the single most
    common way an elasticity model is wrong while looking fine.

    ``ry``/``rt`` must come from nuisance models fitted **without** the validation rows
    (:class:`elasticity_lab.tuning.CausalScorer` does this), or the bins inherit the
    nuisance model's overfitting.
    """
    eps_hat = np.asarray(eps_hat, float)
    if np.std(eps_hat) < 1e-12:
        # A constant-elasticity model has nothing to sort by, so there is one group.  The
        # heterogeneity attributes are left unset rather than faked: "not monotone" and
        # "monotone" are both wrong answers to a question this model never asked.
        th, se = _group_theta(ry, rt, np.ones(len(eps_hat), bool))
        t = pd.DataFrame([{"group": 1, "n": len(eps_hat),
                           "eps_predicted": float(eps_hat.mean()), "eps_estimated": -th,
                           "se": se, "lo": -th - 1.96 * se, "hi": -th + 1.96 * se}])
        t.attrs["monotone"] = None
        t.attrs["spread"] = None
        t.attrs["spread_se"] = None
        t.attrs["constant_model"] = True
        return t
    edges = np.quantile(eps_hat, np.linspace(0, 1, n_groups + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, eps_hat, side="right") - 1, 0, n_groups - 1)
    rows = []
    for g in range(n_groups):
        m = idx == g
        th, se = _group_theta(ry, rt, m)
        rows.append({"group": g + 1, "n": int(m.sum()),
                     "eps_predicted": float(eps_hat[m].mean()) if m.any() else np.nan,
                     "eps_estimated": -th, "se": se,
                     "lo": -th - 1.96 * se, "hi": -th + 1.96 * se})
    t = pd.DataFrame(rows)
    lo, hi = t.eps_estimated.iloc[0], t.eps_estimated.iloc[-1]
    t.attrs["monotone"] = bool(t.eps_estimated.is_monotonic_increasing)
    t.attrs["spread"] = float(hi - lo)
    t.attrs["spread_se"] = float(np.sqrt(t.se.iloc[0] ** 2 + t.se.iloc[-1] ** 2))
    return t


def blp_calibration(ry: np.ndarray, rt: np.ndarray, eps_hat: np.ndarray) -> dict:
    """**B**\\ est **L**\\ inear **P**\\ redictor test: is the *scale* of the heterogeneity right?

    Regress the residualised outcome on two terms::

        ry  =  beta0 · rt  +  beta1 · rt · (eps_hat − mean(eps_hat))  +  error

    ``−beta0`` estimates the average elasticity **from the data** and is returned as
    ``blp_avg_elasticity`` — deliberately not ``avg_elasticity``, which is the model's own
    mean prediction and a different quantity.  ``beta1`` is the interesting one:

    * ``beta1 ≈ −1``  the predicted spread is correctly scaled (the sign is negative
      because ``theta = −eps``);
    * ``beta1 ≈ 0``   the predictions carry no usable heterogeneity at all;
    * ``|beta1| < 1``  the model over-states how much elasticities differ — the usual
      outcome for a flexible learner, which reads noise as signal.

    The ``t`` statistic on ``beta1`` is a formal test that the model detects *any* real
    heterogeneity — but read it next to ``implied_spread`` = ``|beta1| * sd(eps_hat)``,
    which converts the test back into units of elasticity.  A significant ``beta1`` on a
    model whose own spread is ~0 is significant and useless.

    Chernozhukov et al. (2018), adapted from a binary treatment to a continuous one.
    """
    import statsmodels.api as sm

    eps_hat = np.asarray(eps_hat, float)
    centred = eps_hat - eps_hat.mean()
    if np.std(centred) < 1e-12:
        # A constant-elasticity model has no spread to calibrate, but beta0 is still
        # informative, so it is reported rather than dropped.
        th, _ = _group_theta(np.asarray(ry, float), np.asarray(rt, float),
                             np.ones(len(centred), bool))
        return {"beta0": th, "blp_avg_elasticity": -th, "beta1": np.nan,
                "beta1_se": np.nan, "beta1_t": np.nan, "implied_spread": 0.0,
                "heterogeneity_detected": False}
    X = np.column_stack([rt, rt * centred])
    m = sm.OLS(np.asarray(ry, float), X).fit(cov_type="HC1")
    b0, b1 = float(m.params[0]), float(m.params[1])
    se1 = float(m.bse[1])
    # NB the key is blp_avg_elasticity, not avg_elasticity: this is what the DATA say the
    # average is, which is a different quantity from the model's own mean prediction.
    # Naming them alike let this one silently overwrite the other in evaluate_model.
    #
    # implied_spread = |beta1| * sd(eps_hat) rescales the model's own spread by the factor
    # the data say it is off by, giving the data's estimate of the TRUE elasticity sd as
    # seen through this model's ranking.  It is the number to read when beta1 is extreme:
    # on the real panel a tuned ShrunkPerProduct returned beta1 = -21.0 with t = -2.2,
    # which looks like strong evidence until you notice its own spread was 0.003, so the
    # implied spread is 0.07 and the "detected heterogeneity" is nothing to act on.
    return {"beta0": b0, "blp_avg_elasticity": -b0, "beta1": b1, "beta1_se": se1,
            "beta1_t": float(m.tvalues[1]),
            "implied_spread": float(abs(b1) * np.std(eps_hat)),
            "heterogeneity_detected": bool(m.tvalues[1] < -1.96)}


# =======================================================================================
# 3. decision value
# =======================================================================================
def policy_value(ry: np.ndarray, rt: np.ndarray, eps_hat: np.ndarray, *,
                 delta: float = 0.05, n_groups: int = 5,
                 weights: np.ndarray | None = None) -> dict:
    """What the model's implied pricing decision is worth, in log-revenue.

    The decision rule is the textbook one: log revenue is ``log p + log q``, so a price
    nudge of ``d`` changes log revenue by ``d·(1 − eps)``.  Raise the price where the model
    says demand is inelastic (``eps < 1``), cut it where it says demand is elastic::

        d(x) = +delta  if eps_hat(x) < 1,   −delta  otherwise

    The value of that rule needs the *true* elasticity, which we do not have — so it is
    estimated group-wise, exactly as in :func:`gates`: bin by ``eps_hat``, identify each
    bin's elasticity from its own price variation, and average.

    Reported against two references:

    ``value_best_uniform``   the best single ``±delta`` applied to everyone.
    ``lift_over_uniform``    the difference.  **This is the number that justifies a
                             heterogeneous model.**  It is routinely zero or negative,
                             and reporting it is what stops a fancy model from being
                             adopted on the strength of a good AUC-like score alone.
    """
    eps_hat = np.asarray(eps_hat, float)
    w = np.ones(len(eps_hat)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()

    g = gates(ry, rt, eps_hat, n_groups=n_groups)
    if np.std(eps_hat) < 1e-12:
        idx = np.zeros(len(eps_hat), int)
    else:
        edges = np.quantile(eps_hat, np.linspace(0, 1, n_groups + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(edges, eps_hat, side="right") - 1, 0, n_groups - 1)

    eps_true_g = g.eps_estimated.to_numpy(float)
    eps_true_g = np.where(np.isfinite(eps_true_g), eps_true_g, np.nanmean(eps_true_g))
    eps_row = eps_true_g[idx]                       # the data's verdict for each row

    d_model = np.where(eps_hat < 1.0, delta, -delta)
    v_model = float(np.sum(w * d_model * (1 - eps_row)))
    v_up = float(np.sum(w * delta * (1 - eps_row)))
    v_down = -v_up
    v_flat = max(v_up, v_down, 0.0)                 # best uniform move, including "do nothing"
    return {"value_model": v_model, "value_uniform_up": v_up,
            "value_uniform_down": v_down, "value_best_uniform": v_flat,
            "lift_over_uniform": v_model - v_flat,
            "pct_priced_up": float(np.mean(d_model > 0))}


# =======================================================================================
# oracle (synthetic only)
# =======================================================================================
def oracle_metrics(eps_hat: np.ndarray, eps_true: np.ndarray) -> dict[str, float]:
    """The comparison every observable metric above is trying to approximate."""
    eps_hat, eps_true = np.asarray(eps_hat, float), np.asarray(eps_true, float)
    out = {"oracle_rmse": float(np.sqrt(np.mean((eps_hat - eps_true) ** 2))),
           "oracle_bias": float(np.mean(eps_hat - eps_true)),
           "oracle_mae": float(np.mean(np.abs(eps_hat - eps_true))),
           "true_mean": float(eps_true.mean()), "pred_mean": float(eps_hat.mean()),
           "true_sd": float(eps_true.std()), "pred_sd": float(eps_hat.std())}
    out["oracle_corr"] = (float(np.corrcoef(eps_hat, eps_true)[0, 1])
                          if eps_hat.std() > 1e-12 and eps_true.std() > 1e-12 else np.nan)
    # how much of the squared error is a level shift rather than a ranking failure
    out["share_of_mse_from_bias"] = (out["oracle_bias"] ** 2 / out["oracle_rmse"] ** 2
                                     if out["oracle_rmse"] > 0 else np.nan)
    return out


# =======================================================================================
# putting it together
# =======================================================================================
def evaluate_model(model, df: pd.DataFrame, scorer: CausalScorer, *,
                   fold: int = -1, truth_col: str | None = None,
                   n_groups: int = 5, delta: float = 0.05) -> dict:
    """Fit on a fold's training block, then run every diagnostic on its validation block.

    ``fold=-1`` uses the last (most recent) fold, which is the closest thing to "how would
    this have done if I had deployed it".
    """
    tr, va = scorer.folds_[fold]
    dtr, dva = df.iloc[tr], df.iloc[va]
    m = model.fit(dtr)
    eps = np.asarray(m.elasticity(dva), float)
    yhat = np.asarray(m.predict_log_qty(dva), float)
    res = scorer.residuals(fold if fold >= 0 else len(scorer.folds_) + fold)

    out = {"model": getattr(m, "name", type(m).__name__),
           "avg_elasticity": float(np.mean(eps)), "eps_sd": float(np.std(eps))}
    out |= predictive_metrics(dva[OUTCOME].to_numpy(float), yhat)
    rl = CausalScorer.r_loss(res, eps)
    out |= {"rloss": rl,
            "rloss_skill": 1 - rl / max(res["rloss_zero"], 1e-12),
            "rloss_vs_constant": 1 - rl / max(res["rloss_const"], 1e-12)}
    out |= blp_calibration(res["ry"], res["rt"], eps)
    g = gates(res["ry"], res["rt"], eps, n_groups=n_groups)
    out |= {"gates_monotone": g.attrs.get("monotone"),
            "gates_spread": g.attrs.get("spread"),
            "gates_spread_t": (g.attrs.get("spread") / g.attrs["spread_se"]
                               if g.attrs.get("spread_se") else np.nan)}
    out |= policy_value(res["ry"], res["rt"], eps, delta=delta, n_groups=n_groups)
    if truth_col is not None and truth_col in dva:
        out |= oracle_metrics(eps, dva[truth_col].to_numpy(float))
    out["_gates_table"] = g
    out["_elasticity"] = eps
    return out


def compare_models(models: dict, df: pd.DataFrame, scorer: CausalScorer, *,
                   fold: int = -1, truth_col: str | None = None) -> pd.DataFrame:
    """Run :func:`evaluate_model` over a dict of fitted-or-unfitted models."""
    rows = []
    for name, m in models.items():
        try:
            r = evaluate_model(m, df, scorer, fold=fold, truth_col=truth_col)
            r.pop("_gates_table", None)
            r.pop("_elasticity", None)
            rows.append({"name": name, **r})
        except Exception as ex:
            rows.append({"name": name, "error": f"{type(ex).__name__}: {ex}"})
    return pd.DataFrame(rows)


def elasticity_summary(eps: np.ndarray) -> dict[str, float]:
    """Distribution of predicted elasticities, with the sanity checks that matter."""
    eps = np.asarray(eps, float)
    return {"mean": float(eps.mean()), "sd": float(eps.std()),
            "p5": float(np.percentile(eps, 5)), "median": float(np.median(eps)),
            "p95": float(np.percentile(eps, 95)),
            "pct_negative": float(np.mean(eps < 0)),        # upward-sloping demand: wrong
            "pct_inelastic": float(np.mean(eps < 1))}       # price rise raises revenue
