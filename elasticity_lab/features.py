"""
elasticity_lab.features
=======================

Feature engineering for the product-week panel.

Two rules govern every feature here, and both exist to keep the downstream elasticity
estimates honest rather than merely to raise :math:`R^2`.

**Rule 1 — nothing may look forward.**  Every lag, rolling mean and expanding statistic is
computed with ``shift(1)`` inside the product, so a row's features use only weeks strictly
before it.  A rolling mean that includes the current week leaks the outcome and will make
a validation score look wonderful and a forecast look terrible.

**Rule 2 — nothing may be a function of the current week's quantity.**  That is the same
division-bias trap as the unit-value price (:mod:`elasticity_lab.data`).  ``revenue`` and
``n_lines`` for the *current* week are therefore never features; only their lags are.

The engineered block splits into three groups, and the split matters for the causal models
in :mod:`elasticity_lab.models`, which need to know which columns are legitimate controls:

``CONTROL_FEATURES``
    Confounder proxies — things that plausibly move both price and quantity (season,
    product identity, recent trading intensity).  These go into the nuisance models.
``PRICE_FEATURES``
    Functions of the *current* price.  These are the treatment, not controls, and must
    never be fed to a nuisance model that is supposed to be price-free.
``ID_COLUMNS``
    Keys, kept for grouping and splitting, never fed to a model as numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_features", "CONTROL_FEATURES", "PRICE_FEATURES", "ID_COLUMNS",
           "OUTCOME", "TREATMENT", "feature_glossary"]

ID_COLUMNS = ["stock_code", "week"]
OUTCOME = "log_qty"
TREATMENT = "log_price"

PRICE_FEATURES = ["log_price", "rel_price", "price_change", "is_discounted"]

CONTROL_FEATURES = [
    "week_idx", "woy_sin", "woy_cos", "is_q4", "weeks_since_launch",
    "prod_log_price_mean", "prod_log_price_sd", "prod_log_qty_mean", "prod_n_weeks",
    "lag_log_qty", "lag2_log_qty", "roll4_log_qty", "roll12_log_qty",
    "lag_log_price", "roll4_log_price",
    "lag_n_customers", "lag_n_lines",
    "market_log_qty", "market_log_price",
]

feature_glossary = {
    "log_qty": "log units sold this week — the outcome",
    "log_price": "log modal posted price this week — the treatment",
    "rel_price": "log price minus the product's trailing 12-week mean log price; "
                 "'how deep is this week's discount'",
    "price_change": "log price minus last week's log price",
    "is_discounted": "1 if rel_price < −0.02",
    "week_idx": "weeks since the start of the panel — a linear trend",
    "woy_sin/woy_cos": "sine and cosine of week-of-year; smooth annual seasonality",
    "is_q4": "1 in October–December, when this giftware business does most of its trade",
    "weeks_since_launch": "weeks since the product's first observed sale",
    "prod_*": "product-level summaries computed on the training period only",
    "lag_*/roll*": "strictly past values, shifted by one week inside the product",
    "market_log_qty": "log total units sold across all products that week — a control "
                      "for aggregate demand shocks (excludes the product itself)",
    "market_log_price": "log mean price across all products that week — a control for "
                        "cost shocks common to the catalogue",
}


def add_features(panel: pd.DataFrame, *, price_col: str = "price_modal",
                 train_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Engineer the modelling frame.

    Parameters
    ----------
    price_col :
        Which price measure to treat as *the* price.  Defaults to the modal price; pass
        ``'price_unitvalue'`` only to reproduce the division-bias demonstration.
    train_end :
        If given, product-level summary statistics (``prod_*``) are computed using only
        weeks strictly before this date.  Leaving it ``None`` computes them on everything,
        which leaks a little and is fine for exploration but **not** for honest scoring —
        the splitters in :mod:`elasticity_lab.splits` pass it for you.
    """
    d = panel.sort_values(["stock_code", "week"]).copy()
    d["log_qty"] = np.log(d["qty"])
    d["log_price"] = np.log(d[price_col])

    g = d.groupby("stock_code", observed=True)

    # ---- strictly-past product history -------------------------------------------------
    d["lag_log_qty"] = g["log_qty"].shift(1)
    d["lag2_log_qty"] = g["log_qty"].shift(2)
    # a product's second week has no 2-week lag; fall back to the 1-week lag
    d["lag2_log_qty"] = d["lag2_log_qty"].fillna(g["log_qty"].shift(1))
    d["lag_log_price"] = g["log_price"].shift(1)
    d["lag_n_customers"] = g["n_customers"].shift(1)
    d["lag_n_lines"] = g["n_lines"].shift(1)
    for w in (4, 12):
        d[f"roll{w}_log_qty"] = (g["log_qty"].shift(1)
                                 .groupby(d.stock_code, observed=True)
                                 .rolling(w, min_periods=1).mean()
                                 .reset_index(level=0, drop=True))
    d["roll4_log_price"] = (g["log_price"].shift(1)
                            .groupby(d.stock_code, observed=True)
                            .rolling(4, min_periods=1).mean()
                            .reset_index(level=0, drop=True))
    ref = (g["log_price"].shift(1).groupby(d.stock_code, observed=True)
           .rolling(12, min_periods=1).mean().reset_index(level=0, drop=True))
    d["rel_price"] = d["log_price"] - ref
    d["price_change"] = d["log_price"] - d["lag_log_price"]
    d["is_discounted"] = (d["rel_price"] < -0.02).astype(float)

    # ---- calendar ----------------------------------------------------------------------
    wk = pd.to_datetime(d["week"])
    d["week_idx"] = ((wk - wk.min()).dt.days // 7).astype(float)
    woy = wk.dt.isocalendar().week.astype(float)
    d["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    d["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    d["is_q4"] = wk.dt.month.isin([10, 11, 12]).astype(float)
    d["weeks_since_launch"] = (wk - g["week"].transform("min")).dt.days / 7.0

    # ---- market-level controls (leave-one-out, so a product cannot control for itself) --
    tot_q = d.groupby("week", observed=True)["qty"].transform("sum")
    cnt = d.groupby("week", observed=True)["qty"].transform("size")
    d["market_log_qty"] = np.log(np.maximum(tot_q - d["qty"], 1.0))
    sum_lp = d.groupby("week", observed=True)["log_price"].transform("sum")
    d["market_log_price"] = (sum_lp - d["log_price"]) / np.maximum(cnt - 1, 1)

    # ---- product summaries, optionally train-only --------------------------------------
    hist = d if train_end is None else d[d["week"] < train_end]
    ph = hist.groupby("stock_code", observed=True).agg(
        prod_log_price_mean=("log_price", "mean"),
        prod_log_price_sd=("log_price", "std"),
        prod_log_qty_mean=("log_qty", "mean"),
        prod_n_weeks=("log_qty", "size"))
    d = d.merge(ph, on="stock_code", how="left")
    d["prod_log_price_sd"] = d["prod_log_price_sd"].fillna(0.0)
    for c in ("prod_log_price_mean", "prod_log_qty_mean"):
        d[c] = d[c].fillna(d[c].mean())
    d["prod_n_weeks"] = d["prod_n_weeks"].fillna(0.0)

    # rows whose lags are undefined (a product's first week) cannot be modelled
    d = d.dropna(subset=["lag_log_qty", "lag_log_price"]).reset_index(drop=True)
    assert d[CONTROL_FEATURES + PRICE_FEATURES].isna().sum().sum() == 0, \
        "engineered features contain NaN; a lag fallback is missing"
    return d


def design_matrix(df: pd.DataFrame, *, with_price: bool = True) -> pd.DataFrame:
    """The numeric block a model sees.  ``with_price=False`` gives the nuisance design."""
    cols = list(CONTROL_FEATURES) + (list(PRICE_FEATURES) if with_price else [])
    return df[cols].astype(float)
