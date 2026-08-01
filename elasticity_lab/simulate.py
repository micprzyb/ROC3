"""
elasticity_lab.simulate
=======================

A semi-synthetic benchmark with a **known** elasticity.

Why this module exists
----------------------
On the real panel there is no elasticity label, so no amount of cross-validation can tell
you whether a tuned model estimates elasticity *well* — only whether it predicts quantity
well, which is a different question and, as
:mod:`elasticity_lab.tuning` demonstrates, frequently a conflicting one.  Any claim that a
hyperparameter search produced a "high quality" elasticity model is untestable on
observational data alone.

So we build a world that keeps everything real except the outcome:

* the **products**, the **weeks**, the **observed prices** and every **control feature**
  are taken unchanged from the UCI panel, so the covariate structure, the seasonality and
  the price process are those of a real business;
* the **quantity** is generated from a known equation, so each row carries a
  ``true_elasticity`` column;
* the **confounding is real but removable** — see below.

That last point is the design decision that makes the benchmark useful rather than a toy.

The confounding design
----------------------
Let ``X`` be the control features and ``h`` a *nonlinear* function of them.  Write
``u = h(X)`` for a demand shock that a retailer can see (weather, a merchandising push,
the product's position in a catalogue) but that the analyst has not encoded linearly.
Then

    log price  =  observed real log price  +  theta * u  +  small noise
    log qty    =  alpha_product + gamma_week + g(X)  −  eps_product * log price  +  lam * u  +  noise

Both price and quantity move with ``u``, so a naive regression of quantity on price is
biased — by an amount that is known in closed form and reported by
:func:`theoretical_ols_bias`.  Crucially ``u`` is a *deterministic function of observed
controls*, so the confounding is removable **in principle** by any method flexible enough
to learn ``h``.  It is therefore a fair test of tuning: a well-specified, well-tuned
nuisance model recovers the truth; a linear one, or a badly regularised one, does not.

Making ``u`` unobservable instead would be more pessimistic and completely uninformative —
no method could succeed, and every hyperparameter would look equally bad.

Heterogeneity
-------------
``eps_product`` varies across products as a function of product-level characteristics
(price level, how long the product has been selling, its typical order size), plus noise.
Cheaper, longer-established products are made more elastic, which matches the usual
retail finding and gives the heterogeneous-elasticity models something real to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .features import CONTROL_FEATURES, add_features

__all__ = ["SyntheticConfig", "make_synthetic_panel", "theoretical_ols_bias",
           "naive_bias_decomposition", "elasticity_recovery_metrics"]


@dataclass
class SyntheticConfig:
    """Knobs of the data-generating process.  All defaults are documented in the module."""

    #: strength of the confounder's effect on price (theta).  0 = price is exogenous.
    confounding_price: float = 0.35
    #: strength of the confounder's effect on demand (lambda).
    confounding_demand: float = 0.80
    #: mean of the true elasticity distribution across products.
    elasticity_mean: float = 1.60
    #: spread of true elasticity across products (sd of the product-level component).
    elasticity_sd: float = 0.55
    #: how strongly elasticity depends on the product's price level (negative = cheap
    #: products are more elastic).
    elasticity_price_slope: float = -0.45
    #: residual noise on log quantity.
    noise_sd: float = 0.45
    #: multiplier on the nonlinear part of baseline demand g(X).
    signal_strength: float = 1.0
    seed: int = 0
    #: 'lognormal' gives log q directly (the estimand is then exactly the OLS target);
    #: 'poisson' draws counts, which is more realistic and makes the Poisson GLM apt.
    outcome: str = "lognormal"

    def to_dict(self) -> dict:
        return asdict(self)


def _standardise(a: np.ndarray) -> np.ndarray:
    s = np.std(a)
    return (a - np.mean(a)) / (s if s > 1e-12 else 1.0)


def _nonlinear_confounder(d: pd.DataFrame, rng) -> np.ndarray:
    """u = h(X): smooth, genuinely nonlinear, and a function of observed controls only."""
    z = lambda c: _standardise(d[c].to_numpy(float))  # noqa: E731
    u = (0.9 * np.tanh(1.3 * z("roll4_log_qty"))
         + 0.7 * z("is_q4") * z("lag_n_customers")
         + 0.6 * np.sin(2.0 * z("woy_sin"))
         + 0.5 * np.maximum(z("weeks_since_launch"), 0.0) ** 1.5 / 3.0
         + 0.4 * z("market_log_qty") * z("prod_log_price_mean"))
    return _standardise(u)


def _nonlinear_baseline(d: pd.DataFrame) -> np.ndarray:
    """g(X): the part of demand that has nothing to do with price or with u."""
    z = lambda c: _standardise(d[c].to_numpy(float))  # noqa: E731
    g = (0.8 * z("lag_log_qty")
         + 0.5 * z("roll12_log_qty")
         + 0.4 * np.tanh(z("market_log_qty"))
         + 0.3 * z("woy_cos")
         + 0.25 * z("lag_n_lines"))
    return _standardise(g)


def make_synthetic_panel(panel: pd.DataFrame, cfg: SyntheticConfig | None = None,
                         *, price_col: str = "price_modal") -> pd.DataFrame:
    """Build the benchmark frame.

    Takes a **raw panel** and returns a ready **feature frame** — it calls
    :func:`~elasticity_lab.features.add_features` internally, overwrites ``log_price`` and
    ``log_qty`` with simulated values, and repairs the price-derived columns.  Do not run
    ``add_features`` over the result; every engineered column already exists and the merge
    would collide.

    One deliberate simplification: the lagged-quantity controls (``lag_log_qty``,
    ``roll4_log_qty``, …) are lags of the *real* quantities, not of the simulated ones.
    They are still legitimate observed controls — the simulated demand is built as a
    function of them, via ``g(X)`` and ``u = h(X)``, so the world is self-consistent and
    the bias decomposition stays exact.  What it costs is realism in one respect: the
    synthetic series is not autoregressive in its own history, so predictive :math:`R^2`
    is lower here than on the real panel.  Regenerating the lags from simulated quantities
    would restore that at the price of making ``g(X)`` circular, and the benchmark exists
    to grade elasticity recovery, not forecasting.

    Returns the feature frame with synthetic ``log_price`` and ``log_qty``, plus

    ``true_elasticity``
        the product's true elasticity, constant within product, varying across.
    ``u_confounder``
        the realised confounder, for diagnostics only — a model may never use it.
    ``log_price_exogenous``
        the price stripped of its confounded component, i.e. what the price would have
        been in a randomised experiment.  Used to compute the oracle benchmark.

    The DGP parameters are attached to ``df.attrs['dgp']``.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    d = add_features(panel, price_col=price_col).copy()

    # ---- true heterogeneous elasticity, one value per product --------------------------
    prod = d.groupby("stock_code", observed=True).agg(
        pm=("prod_log_price_mean", "first"),
        wk=("prod_n_weeks", "first"),
        sz=("lag_n_lines", "mean"))
    eps = (cfg.elasticity_mean
           + cfg.elasticity_price_slope * _standardise(prod.pm.to_numpy(float))
           + 0.20 * _standardise(np.log1p(prod.sz.to_numpy(float)))
           + cfg.elasticity_sd * rng.standard_normal(len(prod)))
    eps = np.clip(eps, 0.15, 4.0)
    prod_eps = pd.Series(eps, index=prod.index, name="true_elasticity")
    d["true_elasticity"] = d["stock_code"].map(prod_eps).to_numpy(float)

    # ---- confounder, endogenous price --------------------------------------------------
    u = _nonlinear_confounder(d, rng)
    d["u_confounder"] = u
    d["log_price_exogenous"] = d["log_price"].to_numpy(float)
    d["log_price"] = (d["log_price_exogenous"]
                      + cfg.confounding_price * u
                      + 0.05 * rng.standard_normal(len(d)))

    # ---- outcome -----------------------------------------------------------------------
    alpha = pd.Series(rng.normal(0.0, 0.8, len(prod)), index=prod.index)
    weeks = np.array(sorted(pd.unique(d["week"])))
    gamma = pd.Series(rng.normal(0.0, 0.25, len(weeks)), index=weeks)
    # keep every additive component, so the naive bias can be decomposed exactly
    d["_dgp_alpha"] = d["stock_code"].map(alpha).to_numpy(float)
    d["_dgp_gamma"] = d["week"].map(gamma).to_numpy(float)
    d["_dgp_g"] = cfg.signal_strength * _nonlinear_baseline(d)
    d["_dgp_u_term"] = cfg.confounding_demand * u
    d["_dgp_noise"] = cfg.noise_sd * rng.standard_normal(len(d))
    eta = (3.0 + d["_dgp_alpha"] + d["_dgp_gamma"] + d["_dgp_g"]
           - d["true_elasticity"].to_numpy(float) * d["log_price"].to_numpy(float)
           + d["_dgp_u_term"])

    if cfg.outcome == "lognormal":
        d["log_qty"] = eta + d["_dgp_noise"]
        d["qty"] = np.exp(d["log_qty"])
    elif cfg.outcome == "poisson":
        lam = np.exp(np.clip(eta, -5, 12))
        d["qty"] = np.maximum(rng.poisson(lam), 1)
        d["log_qty"] = np.log(d["qty"])
    else:
        raise ValueError("outcome must be 'lognormal' or 'poisson'")

    # recompute the price-derived features, which changed when the price did
    d["rel_price"] = d["log_price"] - d["roll4_log_price"]
    d["price_change"] = d["log_price"] - d["lag_log_price"]
    d["is_discounted"] = (d["rel_price"] < -0.02).astype(float)

    d.attrs["dgp"] = cfg.to_dict()
    # stored as a plain dict, not a Series: pandas compares ``attrs`` element-wise when
    # concatenating frames, and ``Series == Series`` raises "truth value is ambiguous".
    # A Series here makes every downstream ``pd.concat`` of two slices blow up.
    d.attrs["true_elasticity_by_product"] = {k: float(v) for k, v in prod_eps.items()}
    return d


def naive_bias_decomposition(d: pd.DataFrame) -> pd.DataFrame:
    """Exactly where the naive estimate's error comes from.

    A pooled regression of ``log q`` on ``log p`` recovers
    ``Cov(p, y) / Var(p)``.  Substituting the data-generating equation and using the
    linearity of covariance splits that into one contribution per omitted component:

        estimated elasticity  =  mean true elasticity
                                 −  sum over omitted components of Cov(p, comp)/Var(p)
                                 −  the heterogeneity-weighting term

    The heterogeneity term appears because a pooled regression recovers a *variance
    weighted* average of the product-level elasticities, not the plain average: products
    whose price moves most get the most say.  Every channel below is computed from the
    realised sample, and the total is checked against the actual OLS coefficient — so the
    table is an identity, not an approximation.
    """
    import statsmodels.api as sm

    p = d["log_price"].to_numpy(float)
    v = float(np.var(p, ddof=1))   # match np.cov, which uses ddof=1
    eps = d["true_elasticity"].to_numpy(float)
    contrib = {
        "product effect (alpha)": -np.cov(p, d["_dgp_alpha"])[0, 1] / v,
        "week effect (gamma)": -np.cov(p, d["_dgp_gamma"])[0, 1] / v,
        "baseline demand g(X)": -np.cov(p, d["_dgp_g"])[0, 1] / v,
        "confounder u": -np.cov(p, d["_dgp_u_term"])[0, 1] / v,
        "residual noise": -np.cov(p, d["_dgp_noise"])[0, 1] / v,
        # a pooled regression recovers Cov(p, eps*p)/Var(p), a VARIANCE-WEIGHTED
        # average of the product elasticities, not the plain mean.  This channel is
        # the gap between the two, and it is present even with zero confounding.
        "elasticity heterogeneity": float(np.cov(p, eps * p)[0, 1] / v) - eps.mean(),
    }
    truth = float(eps.mean())
    total = truth + sum(contrib.values())
    actual = float(-sm.OLS(d["log_qty"].to_numpy(float), sm.add_constant(p)).fit().params[1])
    rows = [{"channel": "TRUE mean elasticity", "contribution": truth}]
    rows += [{"channel": k, "contribution": float(x)} for k, x in contrib.items()]
    rows += [{"channel": "= implied naive estimate", "contribution": total},
             {"channel": "  actual naive OLS estimate", "contribution": actual},
             {"channel": "  identity check (should be 0)", "contribution": total - actual}]
    out = pd.DataFrame(rows)
    out.attrs["exact"] = abs(total - actual) < 1e-8
    return out


def theoretical_ols_bias(d: pd.DataFrame) -> dict:
    """Headline numbers from :func:`naive_bias_decomposition`."""
    t = naive_bias_decomposition(d).set_index("channel")["contribution"]
    return {
        "true_mean_elasticity": float(t["TRUE mean elasticity"]),
        "naive_estimate": float(t["  actual naive OLS estimate"]),
        "total_bias": float(t["  actual naive OLS estimate"] - t["TRUE mean elasticity"]),
        "largest_channel": str(naive_bias_decomposition(d).iloc[1:7]
                               .set_index("channel")["contribution"].abs().idxmax()),
        "identity_exact": bool(t["  identity check (should be 0)"] < 1e-8),
    }


def elasticity_recovery_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Grade an elasticity estimate against the truth.

    ``bias`` and ``rmse`` are on the elasticity scale.  ``calibration_slope`` is the slope
    of true on predicted — 1 means the spread of predicted heterogeneity is right, below 1
    means the model over-states how much customers differ.  ``spearman`` is ranking only,
    which is what matters if the model is used for targeting rather than for pricing.
    """
    from scipy.stats import spearmanr

    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[ok], true[ok]
    if len(pred) < 3:
        return {k: float("nan") for k in
                ("bias", "rmse", "mae", "calibration_slope", "spearman", "n")}
    err = pred - true
    if np.std(pred) > 1e-9:
        slope = float(np.polyfit(pred, true, 1)[0])
    else:
        slope = float("nan")
    return {
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "calibration_slope": slope,
        "spearman": float(spearmanr(pred, true).statistic) if np.std(pred) > 1e-9 else float("nan"),
        "n": int(len(pred)),
    }
