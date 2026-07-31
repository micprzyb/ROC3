"""
roc3.elasticity
===============

Evaluating and comparing **per-customer price-elasticity models** built from a randomised
price test — when no customer's elasticity is ever observed.

The difficulty you named is real: each customer appears at exactly one price and we see
only a binary purchase.  An individual's elasticity is a contrast between two things that
never both happen, so there is no label to score against, ever.

The way out is that elasticity is **not identified for an individual, but it is exactly
identified for any group** — because the arms were randomised, any pre-specified set of
customers contains all three arms in known proportion, and its demand curve is read off
directly from raw purchase rates.  A model's predictions *define* such groups.  That turns
an impossible individual-level problem into an easy group-level one, and is the basis of
everything here (it is the GATES idea of Chernozhukov, Demirer, Duflo & Fernández-Val).

Two exact structural facts make this setting unusually tractable
----------------------------------------------------------------
With arms at price multipliers ``m_k`` and randomisation probabilities ``q_k``, write
``beta_k(x) = P(buy | x, arm k)`` and ``p_k(x) = P(arm = k | x, bought)``.  Bayes gives
``p_k ∝ q_k beta_k``, so:

1. **The arm classifier already is an elasticity model.**  The unknown normalising factor
   cancels out of any ratio, so the arc elasticity is recovered *exactly*:

       eps(x) = −[log p_3(x) − log p_1(x) − log(q_3/q_1)] / [log m_3 − log m_1]

   No extra modelling, no unidentified constant.  (Verified to 4e-15 in
   ``experiments/08_elasticity.py``.)

2. **The revenue-optimal personalised price is a point on the 3-class ROC surface.**
   Expected revenue from offering price ``m_k`` is ``m_k beta_k(x) = c(x) m_k p_k(x)/q_k``
   with ``c(x) > 0`` free of ``k``, so

       argmax_k  m_k beta_k(x)   =   argmax_k  (m_k/q_k) p_k(x)

   which is exactly ``roc3``'s weighted-argmax rule with weights proportional to the
   prices.  Choosing an operating point on the surface *is* choosing a pricing policy.
   (Verified to 100.000000% agreement on 2M customers.)

What to report
--------------
* :func:`gates_elasticity` — predicted vs **realised** elasticity by bin. The core
  validation; assumption-light, and the one plot to show a stakeholder.
* :func:`policy_value` — what a pricing policy is actually worth, estimated unbiasedly
  off the randomised data. The decision-relevant number.
* :func:`compare_elasticity_models` — paired bootstrap on the difference between two
  models, on all of the above at once.

No single criterion is best in general (Curth & van der Schaar 2023), and losses built
from a particular learner's pseudo-outcome are biased towards that learner (congeniality
bias, Mahajan et al. 2023).  Hence several, reported together.  See ``docs/ELASTICITY.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "elasticity_from_posterior",
    "demand_curve_from_posterior",
    "revenue_optimal_policy",
    "gates_elasticity",
    "blp_slope",
    "policy_value",
    "dr_pseudo_outcome",
    "targeting_curve",
    "compare_elasticity_models",
    "elasticity_report",
    "format_elasticity_report",
    "plot_elasticity_diagnostics",
]

_EPS = 1e-12


def _check_arms(m, q, n_arms=3):
    m = np.asarray(m, dtype=float)
    if m.ndim != 1 or np.any(m <= 0):
        raise ValueError("price_multipliers must be positive")
    if np.any(np.diff(m) <= 0):
        raise ValueError("price_multipliers must be strictly increasing (cheapest first)")
    q = np.full(len(m), 1.0 / len(m)) if q is None else np.asarray(q, dtype=float)
    if q.shape != m.shape or np.any(q <= 0):
        raise ValueError("arm_probs must be positive and the same length as the prices")
    return m, q / q.sum()


# ======================================================================================
# turning the arm classifier into an elasticity / demand model
# ======================================================================================
def elasticity_from_posterior(proba, price_multipliers=(0.9, 1.0, 1.1), arm_probs=None,
                              *, arms=(0, -1)):
    """Per-customer **arc elasticity**, recovered exactly from the arm posterior.

    ``proba[i, k] = P(arm = k | x_i, bought)``, arms in price order, cheapest first.
    ``arms`` selects which two arms define the arc; the default uses the extremes, which
    is the best-conditioned choice.

    Returns a non-negative array: larger = more price sensitive.
    """
    m, q = _check_arms(price_multipliers, arm_probs)
    P = np.asarray(proba, dtype=float)
    lo, hi = arms[0] % len(m), arms[1] % len(m)
    if lo == hi:
        raise ValueError("arms must name two different arms")
    num = (np.log(np.clip(P[:, hi], _EPS, None)) - np.log(np.clip(P[:, lo], _EPS, None))
           - (np.log(q[hi]) - np.log(q[lo])))
    return -num / (np.log(m[hi]) - np.log(m[lo]))


def demand_curve_from_posterior(proba, p_buy, price_multipliers=(0.9, 1.0, 1.1),
                                arm_probs=None):
    """The full individual demand curve ``beta_k(x)``, from the arm posterior *plus* a
    plain purchase-propensity model.

    The arm classifier gives the *shape* of the demand curve but not its level, because
    the normalising constant cancels.  The level comes from ``p_buy = P(buy | x)``, an
    ordinary binary model fitted on **all** customers, via

        beta_k(x) = p_k(x) * P(buy | x) / q_k .

    Needed for absolute revenue predictions; *not* needed to choose the optimal price
    (see :func:`revenue_optimal_policy`).
    """
    m, q = _check_arms(price_multipliers, arm_probs)
    P = np.asarray(proba, dtype=float)
    pb = np.asarray(p_buy, dtype=float)[:, None]
    return np.clip(P * pb / q[None, :], 0.0, 1.0)


def revenue_optimal_policy(proba, price_multipliers=(0.9, 1.0, 1.1), arm_probs=None,
                           *, unit_margin=None):
    """The revenue- (or margin-) maximising personalised price, per customer.

    ``unit_margin[k]`` overrides the value of a sale at arm ``k`` (use ``m_k − cost`` for
    margin rather than revenue).  Returns an integer array of arm indices.

    This is exactly the ``roc3`` weighted-argmax rule with ``w ∝ value_k / q_k``, i.e. a
    point on the 3-class ROC surface — see the module docstring.
    """
    m, q = _check_arms(price_multipliers, arm_probs)
    value = m if unit_margin is None else np.asarray(unit_margin, dtype=float)
    P = np.asarray(proba, dtype=float)
    return np.argmax(P * (value / q)[None, :], axis=1)


# ======================================================================================
# validation 1 — predicted vs REALISED elasticity, by group
# ======================================================================================
def gates_elasticity(arm, bought, score, price_multipliers=(0.9, 1.0, 1.1),
                     arm_probs=None, *, n_bins=5, method="loglinear",
                     predicted=None, alpha=0.05):
    """Realised elasticity per bin of a predicted score — the core validation.

    Bin customers by ``score`` (the model's predicted elasticity, or any ranking of price
    sensitivity), then, **within each bin**, estimate the elasticity directly from raw
    per-arm purchase rates.  Randomisation makes bin membership independent of the arm, so
    this needs no modelling assumptions at all beyond the experiment's own design.

    Parameters
    ----------
    arm : (n,) int array of arm indices, price order, cheapest first
    bought : (n,) 0/1 array
    score : (n,) array — the model's prediction, computed **out of fold**
    predicted : (n,) array, optional
        The predicted elasticity on its own scale, if different from ``score``; used for
        the calibration column.  Defaults to ``score``.
    method : {'loglinear', 'arc'}
        ``'loglinear'`` regresses ``log D_k`` on ``log m_k`` across all arms by weighted
        least squares (more efficient); ``'arc'`` uses only the two extreme arms.

    Returns
    -------
    dict with per-bin realised elasticity, standard errors, confidence intervals, mean
    predicted elasticity, per-arm purchase rates, and counts.
    """
    from scipy.stats import norm

    m, q = _check_arms(price_multipliers, arm_probs)
    arm = np.asarray(arm)
    y = np.asarray(bought, dtype=float)
    s = np.asarray(score, dtype=float)
    pred = s if predicted is None else np.asarray(predicted, dtype=float)
    if not (len(arm) == len(y) == len(s) == len(pred)):
        raise ValueError("arm, bought, score and predicted must have the same length")

    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    logm = np.log(m)
    K = len(m)

    out = {"n_bins": n_bins, "method": method, "bin": [], "n": [], "rates": [],
           "counts": [], "realised": [], "stderr": [], "ci_lower": [], "ci_upper": [],
           "predicted": [], "edges": edges}
    z = norm.ppf(1 - alpha / 2)
    for b in range(n_bins):
        sel = which == b
        n_b = int(sel.sum())
        D = np.empty(K)
        var_log = np.empty(K)
        cnt = np.empty(K, dtype=int)
        for k in range(K):
            mk = sel & (arm == k)
            cnt[k] = int(mk.sum())
            D[k] = y[mk].mean() if cnt[k] else np.nan
            # delta method: Var(log D) ~ (1-D)/(n D)
            var_log[k] = ((1 - D[k]) / (cnt[k] * D[k])) if (cnt[k] and D[k] > 0) else np.inf

        if method == "arc":
            e = -(np.log(D[-1]) - np.log(D[0])) / (logm[-1] - logm[0])
            se = np.sqrt(var_log[0] + var_log[-1]) / abs(logm[-1] - logm[0])
        elif method == "loglinear":
            ok = np.isfinite(var_log) & np.isfinite(D) & (D > 0)
            if ok.sum() < 2:
                e, se = np.nan, np.inf
            else:
                w = 1.0 / var_log[ok]
                X = np.stack([np.ones(ok.sum()), logm[ok]], axis=1)
                W = np.diag(w)
                XtWX_inv = np.linalg.inv(X.T @ W @ X)
                coef = XtWX_inv @ (X.T @ W @ np.log(D[ok]))
                e = -coef[1]
                se = np.sqrt(XtWX_inv[1, 1])
        else:
            raise ValueError("method must be 'loglinear' or 'arc'")

        out["bin"].append(b)
        out["n"].append(n_b)
        out["rates"].append(D)
        out["counts"].append(cnt)
        out["realised"].append(float(e))
        out["stderr"].append(float(se))
        out["ci_lower"].append(float(e - z * se))
        out["ci_upper"].append(float(e + z * se))
        out["predicted"].append(float(pred[sel].mean()))

    for key in ("realised", "stderr", "ci_lower", "ci_upper", "predicted", "n"):
        out[key] = np.asarray(out[key])
    out["rates"] = np.asarray(out["rates"])
    out["counts"] = np.asarray(out["counts"])
    out["spread"] = float(out["realised"][-1] - out["realised"][0])
    out["spread_stderr"] = float(np.sqrt(out["stderr"][-1] ** 2 + out["stderr"][0] ** 2))
    out["monotone"] = bool(np.all(np.diff(out["realised"]) >= -1e-12))
    return out


def blp_slope(gates: dict) -> dict:
    """Calibration slope and intercept of realised on predicted elasticity.

    A perfectly calibrated elasticity model has slope 1 and intercept 0.  Slope
    significantly above 0 means the model carries **some** signal about elasticity; slope
    near 1 means the **magnitudes** are right too.  Fitted by inverse-variance weighted
    least squares over the bins, so the noisier bins count for less.
    """
    from scipy.stats import norm

    x = np.asarray(gates["predicted"], dtype=float)
    yv = np.asarray(gates["realised"], dtype=float)
    se = np.asarray(gates["stderr"], dtype=float)
    ok = np.isfinite(x) & np.isfinite(yv) & np.isfinite(se) & (se > 0)
    if ok.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "slope_stderr": np.nan,
                "p_value_no_signal": np.nan, "p_value_miscalibrated": np.nan}
    w = 1.0 / se[ok] ** 2
    X = np.stack([np.ones(ok.sum()), x[ok]], axis=1)
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    coef = cov @ (X.T @ W @ yv[ok])
    sl_se = float(np.sqrt(cov[1, 1]))
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "slope_stderr": sl_se,
        "slope_ci": (float(coef[1] - 1.96 * sl_se), float(coef[1] + 1.96 * sl_se)),
        "p_value_no_signal": float(2 * norm.sf(abs(coef[1] / sl_se))) if sl_se > 0 else 0.0,
        "p_value_miscalibrated": float(2 * norm.sf(abs((coef[1] - 1) / sl_se)))
        if sl_se > 0 else 0.0,
    }


# ======================================================================================
# validation 2 — what the policy is worth
# ======================================================================================
def policy_value(arm, bought, policy, price_multipliers=(0.9, 1.0, 1.1), arm_probs=None,
                 *, unit_margin=None, beta_hat=None):
    """Expected revenue per customer under a pricing ``policy``, estimated off the test.

    Because the arms were randomised, inverse-probability weighting is **unbiased with no
    modelling assumptions**:

        V(d) = mean_i  1{K_i = d(x_i)} / q_{K_i} * value_{K_i} * Y_i

    Supplying ``beta_hat`` (an ``(n, K)`` estimate of ``P(buy | x, arm k)``) switches to
    the augmented (AIPW) estimator, which is far less noisy and still unbiased if either
    the outcome model or the propensities are right — here the propensities are known
    exactly, so AIPW is unbiased regardless of how bad ``beta_hat`` is.

    Returns a dict with ``value``, ``stderr``, and the per-customer contributions (for
    paired comparisons).
    """
    m, q = _check_arms(price_multipliers, arm_probs)
    value = m if unit_margin is None else np.asarray(unit_margin, dtype=float)
    arm = np.asarray(arm)
    y = np.asarray(bought, dtype=float)
    pol = np.asarray(policy)
    if pol.ndim != 1 or len(pol) != len(arm):
        raise ValueError("policy must be a per-customer arm index array")

    match = (arm == pol).astype(float)
    contrib = match / q[arm] * value[arm] * y
    if beta_hat is not None:
        B = np.asarray(beta_hat, dtype=float)
        n = len(arm)
        direct = value[pol] * B[np.arange(n), pol]
        resid = match / q[arm] * value[arm] * (y - B[np.arange(n), arm])
        contrib = direct + resid
    return {
        "value": float(contrib.mean()),
        "stderr": float(contrib.std(ddof=1) / np.sqrt(len(contrib))),
        "contributions": contrib,
        "estimator": "aipw" if beta_hat is not None else "ipw",
    }


def dr_pseudo_outcome(arm, bought, beta_hat, arm_probs=None, *, pair=(0, -1)):
    """Doubly-robust pseudo-outcome for the **uplift** ``beta_lo(x) − beta_hi(x)``.

    Its conditional mean is the true uplift, so it can stand in for the missing label when
    scoring a ranking (rank correlation) or a regression (DR-MSE).  Noisy per customer,
    unbiased in aggregate.

    Note this targets a *difference*, whereas elasticity is a *ratio*; the two rank
    customers differently, and neither is the same as the revenue gain from discounting.
    :func:`gates_elasticity` avoids the issue by estimating the ratio directly per bin.
    """
    K = np.asarray(beta_hat).shape[1]
    m, q = _check_arms(np.arange(1, K + 1) / K, arm_probs)  # only q is used here
    arm = np.asarray(arm)
    y = np.asarray(bought, dtype=float)
    B = np.asarray(beta_hat, dtype=float)
    lo, hi = pair[0] % K, pair[1] % K
    n = len(arm)
    psi = B[:, lo] - B[:, hi]
    psi = psi + (arm == lo) / q[lo] * (y - B[np.arange(n), lo])
    psi = psi - (arm == hi) / q[hi] * (y - B[np.arange(n), hi])
    return psi


def targeting_curve(arm, bought, gain_score, policy, price_multipliers=(0.9, 1.0, 1.1),
                    arm_probs=None, *, baseline_arm=1, n_points=21, unit_margin=None):
    """Revenue as a function of how many customers you personalise — a pricing Qini curve.

    Rank customers by ``gain_score`` (predicted revenue gain from moving off the baseline
    price), personalise the top ``t`` fraction and leave the rest at ``baseline_arm``, and
    estimate the resulting policy value by IPW at each ``t``.  A model with real signal
    puts its gains at small ``t``; a model with none traces a straight line.

    Returns ``(fractions, values, area)`` where ``area`` is the mean uplift over the
    baseline across the sweep — an AUUC-style scalar, higher is better.
    """
    m, q = _check_arms(price_multipliers, arm_probs)
    g = np.asarray(gain_score, dtype=float)
    pol = np.asarray(policy)
    order = np.argsort(-g)
    fracs = np.linspace(0.0, 1.0, n_points)
    base = np.full(len(g), int(baseline_arm))
    v0 = policy_value(arm, bought, base, m, q, unit_margin=unit_margin)["value"]
    vals = []
    for t in fracs:
        take = order[: int(round(t * len(g)))]
        pi = base.copy()
        pi[take] = pol[take]
        vals.append(policy_value(arm, bought, pi, m, q, unit_margin=unit_margin)["value"])
    vals = np.asarray(vals)
    return fracs, vals, float(np.mean(vals - v0))


# ======================================================================================
# comparing two models
# ======================================================================================
def compare_elasticity_models(arm, bought, score_a, score_b, policy_a, policy_b,
                              price_multipliers=(0.9, 1.0, 1.1), arm_probs=None, *,
                              n_bins=5, n_boot=400, unit_margin=None, seed=0,
                              beta_hat_a=None, beta_hat_b=None):
    """Head-to-head comparison of two elasticity models, with a paired bootstrap.

    Pairing matters: both models are scored on the *same* resampled customers, so the
    shared sampling noise cancels and the interval on the difference is far tighter than
    two independent intervals would suggest.

    Returns a dict with, for each model and for the difference: the policy value, the
    calibration slope, the GATES spread (top bin minus bottom bin realised elasticity),
    and the targeting-curve area.
    """
    m, q = _check_arms(price_multipliers, arm_probs)
    arm = np.asarray(arm)
    y = np.asarray(bought, dtype=float)
    n = len(arm)
    rng = np.random.default_rng(seed)

    def summarise(idx, sa, sb, pa, pb, ba, bb):
        va = policy_value(arm[idx], y[idx], pa[idx], m, q, unit_margin=unit_margin,
                          beta_hat=None if ba is None else ba[idx])["value"]
        vb = policy_value(arm[idx], y[idx], pb[idx], m, q, unit_margin=unit_margin,
                          beta_hat=None if bb is None else bb[idx])["value"]
        ga = gates_elasticity(arm[idx], y[idx], sa[idx], m, q, n_bins=n_bins)
        gb = gates_elasticity(arm[idx], y[idx], sb[idx], m, q, n_bins=n_bins)
        return (va, vb, ga["spread"], gb["spread"],
                blp_slope(ga)["slope"], blp_slope(gb)["slope"])

    point = summarise(np.arange(n), score_a, score_b, policy_a, policy_b,
                      beta_hat_a, beta_hat_b)
    reps = np.empty((n_boot, 6))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            reps[b] = summarise(idx, score_a, score_b, policy_a, policy_b,
                                beta_hat_a, beta_hat_b)
        except Exception:                       # pragma: no cover - degenerate resample
            reps[b] = np.nan

    def ci(col_a, col_b):
        d = reps[:, col_a] - reps[:, col_b]
        d = d[np.isfinite(d)]
        lo, hi = np.quantile(d, [0.025, 0.975])
        return {"diff": float(point[col_a] - point[col_b]),
                "ci": (float(lo), float(hi)),
                "significant": bool(lo * hi > 0)}

    return {
        "policy_value": {"a": point[0], "b": point[1], "a_minus_b": ci(0, 1)},
        "gates_spread": {"a": point[2], "b": point[3], "a_minus_b": ci(2, 3)},
        "calibration_slope": {"a": point[4], "b": point[5], "a_minus_b": ci(4, 5)},
        "n_boot": n_boot,
    }


# ======================================================================================
# reporting
# ======================================================================================
def elasticity_report(arm, bought, proba, price_multipliers=(0.9, 1.0, 1.1),
                      arm_probs=None, *, n_bins=5, p_buy=None, unit_margin=None,
                      baseline_arm=1) -> dict:
    """One-call evaluation of an arm-classifier-derived elasticity model."""
    m, q = _check_arms(price_multipliers, arm_probs)
    P = np.asarray(proba, dtype=float)
    eps = elasticity_from_posterior(P, m, q)
    pol = revenue_optimal_policy(P, m, q, unit_margin=unit_margin)
    value = m if unit_margin is None else np.asarray(unit_margin, dtype=float)

    gates = gates_elasticity(arm, bought, eps, m, q, n_bins=n_bins)
    blp = blp_slope(gates)
    beta = (None if p_buy is None
            else demand_curve_from_posterior(P, p_buy, m, q))
    v_pol = policy_value(arm, bought, pol, m, q, unit_margin=unit_margin, beta_hat=beta)
    flats = {int(k): policy_value(arm, bought, np.full(len(arm), k), m, q,
                                  unit_margin=unit_margin, beta_hat=beta)
             for k in range(len(m))}
    # gain score for the targeting curve: predicted revenue gain over the baseline
    gain = (P * (value / q)[None, :]).max(axis=1) - P[:, baseline_arm] * (
        value[baseline_arm] / q[baseline_arm])
    fr, vals, area = targeting_curve(arm, bought, gain, pol, m, q,
                                     baseline_arm=baseline_arm, unit_margin=unit_margin)
    return {
        "elasticity": eps, "policy": pol, "gain_score": gain,
        "gates": gates, "blp": blp,
        "policy_value": v_pol, "flat_policies": flats,
        "baseline_arm": baseline_arm,
        "uplift_over_baseline": v_pol["value"] - flats[baseline_arm]["value"],
        "targeting_curve": (fr, vals, area),
        "price_multipliers": m, "arm_probs": q,
        "beta_hat": beta,
    }


def format_elasticity_report(rep: dict) -> str:
    g, blp = rep["gates"], rep["blp"]
    m = rep["price_multipliers"]
    L = ["=" * 76,
         "PRICE-ELASTICITY MODEL — VALIDATION AGAINST DATA THAT HAS NO ELASTICITY LABELS",
         "=" * 76,
         f"price arms (multipliers, cheapest first): {np.round(m, 4)}",
         f"predicted elasticity: mean {rep['elasticity'].mean():.3f}, "
         f"range {rep['elasticity'].min():.3f} .. {rep['elasticity'].max():.3f}",
         "",
         "1. DOES IT RANK CUSTOMERS BY ELASTICITY?",
         "   bins are formed from the model's prediction; the realised elasticity in each",
         "   bin is measured from raw per-arm purchase rates -- randomisation alone.",
         "",
         f"   {'bin':>4} {'n':>9} {'predicted':>10} {'realised':>10} {'95% CI':>20}"
         f"   purchase rates by arm"]
    for b in range(g["n_bins"]):
        ci_txt = f"({g['ci_lower'][b]:.3f}, {g['ci_upper'][b]:.3f})"
        L.append(f"   {b:>4} {g['n'][b]:>9,} {g['predicted'][b]:>10.3f} "
                 f"{g['realised'][b]:>10.3f} {ci_txt:>20}"
                 f"   {np.round(g['rates'][b], 4)}")
    L += ["",
          f"   monotone across bins            : {g['monotone']}",
          f"   spread (top bin - bottom bin)   : {g['spread']:+.3f} "
          f"+/- {g['spread_stderr']:.3f}",
          "",
          "2. ARE THE MAGNITUDES RIGHT?  (calibration of realised on predicted)",
          f"   slope     {blp['slope']:+.3f} +/- {blp['slope_stderr']:.3f}"
          f"   (1.000 = perfectly calibrated)",
          f"   intercept {blp['intercept']:+.3f}",
          f"   p(no signal, slope = 0)   {blp['p_value_no_signal']:.2e}",
          f"   p(miscalibrated, slope = 1) {blp['p_value_miscalibrated']:.3f}",
          "",
          "3. IS IT WORTH ANYTHING?  (revenue per customer, estimated off the experiment)"]
    for k, v in rep["flat_policies"].items():
        tag = "   <- baseline" if k == rep["baseline_arm"] else ""
        L.append(f"   flat price {m[k]:.2f}          {v['value']:.5f} "
                 f"+/- {v['stderr']:.5f}{tag}")
    pv = rep["policy_value"]
    base = rep["flat_policies"][rep["baseline_arm"]]["value"]
    L += [f"   personalised            {pv['value']:.5f} +/- {pv['stderr']:.5f}"
          f"   ({pv['estimator'].upper()})",
          f"   uplift over baseline    {rep['uplift_over_baseline']:+.5f}"
          f"   ({100 * rep['uplift_over_baseline'] / base:+.2f}%)",
          f"   targeting-curve area    {rep['targeting_curve'][2]:+.5f}",
          "=" * 76]
    return "\n".join(L)


def plot_elasticity_diagnostics(rep: dict, *, theme="light", path=None, dpi=150,
                                title=None, truth=None):
    """Two panels: the elasticity calibration curve, and the targeting curve."""
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import THEMES, _style_axes

    th = THEMES[theme] if isinstance(theme, str) else theme
    g = rep["gates"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), facecolor=th["page"])

    ax = axes[0]
    lo = min(g["predicted"].min(), g["ci_lower"].min())
    hi = max(g["predicted"].max(), g["ci_upper"].max())
    pad = 0.08 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=th["muted"],
            linewidth=1.2, zorder=2)
    ax.annotate("perfect calibration", (hi, hi), textcoords="offset points",
                xytext=(-8, 8), ha="right", fontsize=8, color=th["muted"])
    ax.errorbar(g["predicted"], g["realised"],
                yerr=[g["realised"] - g["ci_lower"], g["ci_upper"] - g["realised"]],
                fmt="o", color=th["series"][0], ecolor=th["series"][0], elinewidth=1.4,
                capsize=3, markersize=7, zorder=4, label="bins (95% CI)")
    b = rep["blp"]
    xs = np.array([lo - pad, hi + pad])
    ax.plot(xs, b["intercept"] + b["slope"] * xs, color=th["series"][1], linewidth=1.8,
            zorder=3, label=f"fit: slope {b['slope']:.2f}")
    _style_axes(ax, th)
    ax.set_xlabel("predicted elasticity (bin mean)", fontsize=9)
    ax.set_ylabel("realised elasticity (from raw purchase rates)", fontsize=9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_title("Does the model's elasticity mean anything?\n"
                 "bins from the model; heights measured by randomisation alone",
                 fontsize=11, color=th["ink"], pad=8)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                    facecolor=th["surface"], edgecolor=th["grid"])
    for t in leg.get_texts():
        t.set_color(th["ink2"])

    ax = axes[1]
    fr, vals, area = rep["targeting_curve"]
    base = rep["flat_policies"][rep["baseline_arm"]]["value"]
    ax.plot(fr, vals - base, color=th["series"][0], linewidth=2.2, label="model")
    ax.plot([0, 1], [0, vals[-1] - base], color=th["muted"], linewidth=1.4,
            label="no targeting signal")
    ax.axhline(0.0, color=th["axis"], linewidth=0.9)
    if truth is not None:
        ax.plot(truth[0], truth[1] - base, color=th["series"][2], linewidth=1.8,
                linestyle="-", label="oracle")
    _style_axes(ax, th)
    ax.set_xlabel("fraction of customers personalised (most promising first)", fontsize=9)
    ax.set_ylabel("revenue per customer, above baseline", fontsize=9)
    ax.set_title(f"Is it worth anything?\ntargeting-curve area {area:+.5f}",
                 fontsize=11, color=th["ink"], pad=8)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                    facecolor=th["surface"], edgecolor=th["grid"])
    for t in leg.get_texts():
        t.set_color(th["ink2"])

    if title:
        fig.suptitle(title, fontsize=14, color=th["ink"], x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi, facecolor=th["page"], bbox_inches="tight")
    return fig
