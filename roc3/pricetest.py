"""
roc3.pricetest
==============

ROC/VUS analysis for **"which price arm was this buyer in?"**

The setting.  You run a randomised price test with arms at, say, −10% / 0% / +10%.
For each customer you observe covariates ``x``, the arm they were assigned, and whether
they bought.  Restricting to **buyers**, you fit a model for ``P(arm = k | x)`` and want
an ROC-style analysis: thresholds to pick, and a VUS to report.

Why this is not an ordinary 3-class problem
-------------------------------------------
Write ``beta_k(x) = P(buy | customer x, arm k)`` for the individual demand curve, and
``q_k`` for the (known, randomised) arm probabilities.  Conditioning on purchase,

    P(arm = k | x, bought)  =  q_k beta_k(x) / sum_j q_j beta_j(x).

**Monotone demand** — a given customer is at least as likely to buy at a lower price —
says ``beta_cheap(x) >= beta_mid(x) >= beta_expensive(x)`` for every ``x``.  With equal
randomisation this gives, for every buyer,

    p_cheap(x)  >=  p_mid(x)  >=  p_expensive(x).

So the ordering constraint is **derived, not assumed**, and it constrains the *true
posterior*.  That is "Reading A" of :mod:`roc3.constrained`: it binds every classifier,
and it caps the attainable VUS strictly below 1.

The result that matters
-----------------------
Averaging the display above over customers, the prevalence of each arm **among buyers** is

    pi_k  =  q_k D_k / sum_j q_j D_j ,      D_k = the purchase rate observed in arm k

— quantities you read straight off the experiment's topline, before fitting anything.
And the VUS ceiling depends only on ``pi``.  So:

* **The maximum attainable VUS is a function of the measured demand response.**
  For a −10/0/+10 test with unit price elasticity the ceiling is ``0.219`` against a
  chance level of ``0.167``.  The entire achievable band is about five points wide.
* A model reporting a VUS above the ceiling has a **bug or a leak** — see :func:`audit`.
* Reporting VUS on the usual ``[1/6, 1]`` scale makes every honest model look worthless.
  Report it against the ceiling instead.

See ``docs/PRICETEST.md`` for the derivations and the measurements.
"""

from __future__ import annotations

import numpy as np

from .constrained import (pairwise_auc_ceiling, scalar_ceilings, vus_ceiling)
from .core import CHANCE_VUS

__all__ = [
    "posterior_prevalences",
    "purchase_rates_from_elasticity",
    "check_monotone_demand",
    "ceiling_from_purchase_rates",
    "ceiling_from_elasticity",
    "audit",
    "pricetest_report",
    "format_pricetest_report",
    "plot_ceiling_vs_elasticity",
]


def posterior_prevalences(purchase_rates, arm_probs=None) -> np.ndarray:
    """Arm prevalence **among buyers**, from the experiment's topline numbers.

    Parameters
    ----------
    purchase_rates : array-like, shape (3,)
        ``D_k`` — the fraction of customers in arm ``k`` who bought.  Arms in **price
        order**, cheapest first.
    arm_probs : array-like, shape (3,), optional
        ``q_k`` — the randomisation probabilities.  Defaults to equal allocation.

    Returns
    -------
    ndarray, shape (3,), summing to 1, in the same (price) order as the input.
    """
    D = np.asarray(purchase_rates, dtype=float)
    if D.shape != (3,) or np.any(D < 0):
        raise ValueError("purchase_rates must be 3 non-negative numbers, cheapest first")
    q = np.full(3, 1 / 3) if arm_probs is None else np.asarray(arm_probs, dtype=float)
    q = q / q.sum()
    w = q * D
    if w.sum() <= 0:
        raise ValueError("no purchases in any arm")
    return w / w.sum()


def purchase_rates_from_elasticity(elasticity, price_multipliers=(0.9, 1.0, 1.1),
                                   base_rate=0.5) -> np.ndarray:
    """Constant-elasticity demand ``D(p) ∝ p^(−ε)``, normalised to ``base_rate`` at p = 1.

    A planning tool: it lets you compute the ceiling *before* running the test, from an
    assumed elasticity, and decide whether the arm-prediction exercise is worth doing.
    """
    m = np.asarray(price_multipliers, dtype=float)
    if m.shape != (3,) or np.any(m <= 0):
        raise ValueError("price_multipliers must be 3 positive numbers, cheapest first")
    if np.any(np.diff(m) <= 0):
        raise ValueError("price_multipliers must be strictly increasing (cheapest first)")
    return float(base_rate) * m ** (-float(elasticity))


def check_monotone_demand(scores, *, score_type="proba") -> dict:
    """Does the model's output respect ``p_cheap >= p_mid >= p_expensive`` row by row?

    Columns must be in **price order, cheapest first**.  A violation means the model is
    predicting that some customer was *more* likely to have bought at a higher price,
    which monotone demand forbids.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("scores must have shape (n, 3), arms in price order")
    if score_type == "proba":
        logp = np.log(np.clip(s / s.sum(axis=1, keepdims=True), 1e-300, None))
    else:
        logp = s
    d1 = logp[:, 0] - logp[:, 1]          # want >= 0
    d2 = logp[:, 1] - logp[:, 2]          # want >= 0
    ok = (d1 >= 0) & (d2 >= 0)
    return {
        "satisfied": bool(ok.all()),
        "fraction_satisfied": float(ok.mean()),
        "n_violations": int((~ok).sum()),
        "worst_violation_log": float(min(d1.min(), d2.min())),
    }


def ceiling_from_purchase_rates(purchase_rates, arm_probs=None) -> dict:
    """Everything the demand response implies, before any model is fitted.

    Returns a dict with the buyer-side prevalences, the VUS ceiling, the pairwise AUC
    ceilings, and the ceilings on accuracy / balanced accuracy / worst-class sensitivity.
    """
    pi = posterior_prevalences(purchase_rates, arm_probs)
    # the polytope machinery wants classes in ascending prevalence; the ceiling is a
    # volume, so it is invariant to relabelling the axes
    pi_asc = np.sort(pi)
    D = np.asarray(purchase_rates, dtype=float)
    return {
        "purchase_rates": D,
        "prevalence_among_buyers": pi,          # price order, cheapest first
        "prevalence_ascending": pi_asc,
        "demand_ratios": {"cheap/mid": float(D[0] / D[1]),
                          "mid/expensive": float(D[1] / D[2]),
                          "cheap/expensive": float(D[0] / D[2])},
        "vus_ceiling": vus_ceiling(pi_asc),
        "chance": CHANCE_VUS,
        "pairwise_auc_ceiling": pairwise_auc_ceiling(pi_asc),
        "scalar_ceilings": scalar_ceilings(pi_asc),
    }


def ceiling_from_elasticity(elasticity, price_multipliers=(0.9, 1.0, 1.1),
                            arm_probs=None) -> dict:
    """:func:`ceiling_from_purchase_rates` for an assumed constant elasticity."""
    D = purchase_rates_from_elasticity(elasticity, price_multipliers)
    out = ceiling_from_purchase_rates(D, arm_probs)
    out["elasticity"] = float(elasticity)
    out["price_multipliers"] = np.asarray(price_multipliers, dtype=float)
    return out


def audit(vus, purchase_rates, arm_probs=None, *, tolerance=0.01) -> dict:
    """Is a reported VUS even possible, given the observed demand response?

    The ceiling depends only on the topline purchase rates, so this is a cheap and quite
    powerful sanity check.  A VUS above the ceiling cannot come from a correct evaluation
    of an honest model on this task: the usual causes are target leakage (a feature that
    encodes the arm, e.g. the price paid, the discount code, the order total), evaluating
    on the training split, or scoring non-buyers as well as buyers.

    Returns a dict with ``verdict`` in ``{'impossible', 'suspicious', 'plausible',
    'at or below chance'}``.
    """
    info = ceiling_from_purchase_rates(purchase_rates, arm_probs)
    ceil = info["vus_ceiling"]
    v = float(vus)
    if v > ceil + tolerance:
        verdict = "impossible"
    elif v > ceil - 1e-12:
        verdict = "suspicious"
    elif v <= CHANCE_VUS:
        verdict = "at or below chance"
    else:
        verdict = "plausible"
    span = ceil - CHANCE_VUS
    return {
        "vus": v,
        "ceiling": ceil,
        "chance": CHANCE_VUS,
        "verdict": verdict,
        "fraction_of_attainable": float((v - CHANCE_VUS) / span) if span > 0 else float("nan"),
        "excess_over_ceiling": float(v - ceil),
        **{k: info[k] for k in ("prevalence_among_buyers", "demand_ratios")},
    }


def pricetest_report(y, scores, purchase_rates, *, arm_probs=None, classes=None,
                     score_type="proba", resolution=240, marker=None) -> dict:
    """Full analysis of a fitted arm-prediction model against its logical ceiling.

    ``y`` and the columns of ``scores`` must both be in **price order, cheapest first**.
    ``marker``, if given, is a single price-sensitivity index; when the problem is
    one-dimensional this is the better object to threshold on (see ``docs/PRICETEST.md``).
    """
    from .core import roc_surface
    from .ordinal import ordinal_roc_surface, vus_ordinal
    from .vus import forced_choice_profile

    surf = roc_surface(y, scores, classes=classes, score_type=score_type,
                       resolution=resolution)
    fc = forced_choice_profile(y, scores, classes=classes, score_type=score_type)
    info = ceiling_from_purchase_rates(purchase_rates, arm_probs)
    ceil = info["vus_ceiling"]
    span = ceil - CHANCE_VUS

    out = {
        "surface": surf,
        "VUS": surf.vus,
        "VUS_3AFC": fc.vus,
        "monotone_demand": check_monotone_demand(scores, score_type=score_type),
        "conventional_scale": (surf.vus - CHANCE_VUS) / (1 - CHANCE_VUS),
        "fraction_of_attainable": (surf.vus - CHANCE_VUS) / span if span > 0 else np.nan,
        "audit": audit(surf.vus, purchase_rates, arm_probs),
        **info,
    }
    if marker is not None:
        out["ordinal_surface"] = ordinal_roc_surface(
            y, marker, classes=classes, resolution=resolution)
        out["VUS_ordinal_rank"] = vus_ordinal(y, marker, classes=classes)
    return out


def format_pricetest_report(rep: dict) -> str:
    pi = rep["prevalence_among_buyers"]
    dr = rep["demand_ratios"]
    sc = rep["scalar_ceilings"]
    md = rep["monotone_demand"]
    L = ["=" * 74,
         "PRICE-TEST ARM PREDICTION — VUS AGAINST ITS LOGICAL CEILING",
         "=" * 74,
         "measured demand response (arms in price order, cheapest first)",
         f"  purchase rate per arm        : {np.round(rep['purchase_rates'], 4)}",
         f"  demand ratios                : cheap/mid {dr['cheap/mid']:.3f}   "
         f"mid/exp {dr['mid/expensive']:.3f}   cheap/exp {dr['cheap/expensive']:.3f}",
         f"  prevalence among buyers pi   : {np.round(pi, 4)}",
         "",
         "model output respects monotone demand (p_cheap >= p_mid >= p_exp)?",
         f"  {md['fraction_satisfied']:.1%} of rows"
         + ("" if md["satisfied"] else
            f"   ** {md['n_violations']} violations **"),
         "",
         "volume under the ROC surface",
         f"  VUS  (geometric)             : {rep['VUS']:.4f}",
         f"  VUS  (3AFC)                  : {rep['VUS_3AFC']:.4f}"]
    if "VUS_ordinal_rank" in rep:
        L.append(f"  VUS  (ordinal, P(s1<s2<s3))  : {rep['VUS_ordinal_rank']:.4f}")
    L += [f"  chance                       : {CHANCE_VUS:.4f}",
          f"  CEILING set by the demand    : {rep['vus_ceiling']:.4f}",
          "",
          "how to read it",
          f"  on the conventional [1/6, 1] scale : {rep['conventional_scale']:+.4f}"
          "   <- misleading here",
          f"  as a fraction of what is attainable: "
          f"{rep['fraction_of_attainable']:.1%}   <- the honest number",
          f"  audit verdict                      : {rep['audit']['verdict'].upper()}",
          "",
          "no classifier on this experiment can exceed",
          f"  accuracy            {sc['accuracy']:.4f}   "
          f"(the constant rule 'always the cheap arm' attains it)",
          f"  balanced accuracy   {sc['balanced_accuracy']:.4f}",
          f"  worst-class Se      {sc['min_sensitivity']:.4f}",
          "=" * 74]
    return "\n".join(L)


def plot_ceiling_vs_elasticity(elasticities=None, price_multipliers=(0.9, 1.0, 1.1), *,
                               theme="light", path=None, dpi=150, observed=None):
    """The headline picture: how much signal the experiment could possibly contain.

    ``observed`` is an optional ``(elasticity, vus)`` pair to mark.
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import THEMES, _style_axes

    th = THEMES[theme] if isinstance(theme, str) else theme
    eps = np.linspace(0.1, 6.0, 40) if elasticities is None else np.asarray(elasticities)
    ceil, pw = [], []
    for e in eps:
        info = ceiling_from_elasticity(e, price_multipliers)
        ceil.append(info["vus_ceiling"])
        pw.append(min(info["pairwise_auc_ceiling"].values()))
    ceil, pw = np.array(ceil), np.array(pw)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), facecolor=th["page"])

    ax = axes[0]
    ax.fill_between(eps, CHANCE_VUS, ceil, color=th["series"][0], alpha=0.18,
                    label="attainable band")
    ax.plot(eps, ceil, color=th["series"][0], linewidth=2.2, label="VUS ceiling")
    ax.axhline(CHANCE_VUS, color=th["muted"], linewidth=1.2)
    ax.annotate("chance = 1/6", (eps[-1], CHANCE_VUS), textcoords="offset points",
                xytext=(-6, 6), ha="right", fontsize=8, color=th["muted"])
    if observed is not None:
        ax.scatter([observed[0]], [observed[1]], s=90, marker="X", color=th["ink"],
                   edgecolor=th["surface"], linewidth=1.4, zorder=6)
        ax.annotate("observed", observed, textcoords="offset points", xytext=(8, 6),
                    fontsize=8.5, color=th["ink"])
    _style_axes(ax, th)
    ax.set_xlabel("price elasticity of demand  |ε|", fontsize=9)
    ax.set_ylabel("VUS", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Everything an arm-prediction model could possibly achieve\n"
                 f"price arms {tuple(price_multipliers)}", fontsize=11,
                 color=th["ink"], pad=8)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                    facecolor=th["surface"], edgecolor=th["grid"])
    for t in leg.get_texts():
        t.set_color(th["ink2"])

    ax = axes[1]
    ax.plot(eps, pw, color=th["series"][1], linewidth=2.2,
            label="tightest pairwise AUC ceiling")
    ax.axhline(0.5, color=th["muted"], linewidth=1.2)
    ax.annotate("chance = 1/2", (eps[-1], 0.5), textcoords="offset points",
                xytext=(-6, 6), ha="right", fontsize=8, color=th["muted"])
    _style_axes(ax, th)
    ax.set_xlabel("price elasticity of demand  |ε|", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_ylim(0.4, 1)
    ax.set_title("...and for any one pair of arms\n"
                 "$1 - \\pi_i/(2\\pi_j)$", fontsize=11, color=th["ink"], pad=8)
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                    facecolor=th["surface"], edgecolor=th["grid"])
    for t in leg.get_texts():
        t.set_color(th["ink2"])

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi, facecolor=th["page"], bbox_inches="tight")
    return fig
