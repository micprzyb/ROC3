"""
elasticity_lab.bench
====================

The two standard frames every notebook and experiment starts from, so that all of them
demonstrably run on the same data.

``real_frame()``
    The UCI panel with features attached.  No ground truth: elasticity is unknowable here,
    which is the honest situation.
``synth_frame()``
    The same panel with quantities *replaced* by a simulated demand system whose elasticity
    is known per product.  This is where methods get graded.

Both accept ``n_products`` to subsample.  Subsampling is by whole product, never by row —
dropping random rows would tear holes in the lag structure and quietly break every
time-series feature.  The notebooks use a subsample so they execute in minutes; the
experiment scripts in ``experiments/`` run the same code at full scale, and the two agree
on every qualitative conclusion (the full-scale numbers are in
``docs/ELASTICITY_MODELS.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import get_panel, panel_fingerprint
from .features import add_features
from .simulate import SyntheticConfig, make_synthetic_panel

__all__ = ["real_frame", "synth_frame", "subsample_products", "frame_summary"]


def subsample_products(panel: pd.DataFrame, n_products: int | None,
                       seed: int = 0) -> pd.DataFrame:
    """Keep ``n_products`` whole products, chosen at random with a fixed seed."""
    if n_products is None:
        return panel
    codes = pd.unique(panel["stock_code"])
    if n_products >= len(codes):
        return panel
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(codes, size=n_products, replace=False))
    return panel[panel["stock_code"].isin(keep)].reset_index(drop=True)


def real_frame(n_products: int | None = None, *, seed: int = 0,
               min_weeks: int = 20) -> pd.DataFrame:
    """UCI Online Retail II, cleaned, aggregated and feature-engineered."""
    panel = subsample_products(get_panel(min_weeks=min_weeks), n_products, seed)
    df = add_features(panel)
    df.attrs["fingerprint"] = panel_fingerprint(panel)
    df.attrs["source"] = "UCI Online Retail II"
    return df


def synth_frame(n_products: int | None = None, *, seed: int = 0,
                confounding_price: float = 0.35, confounding_demand: float = 0.80,
                min_weeks: int = 20,
                config: SyntheticConfig | None = None) -> pd.DataFrame:
    """The gradeable twin: real prices and covariates, simulated quantities.

    ``confounding_price`` (theta) and ``confounding_demand`` (lambda) set how strongly the
    hidden factor moves price and quantity respectively.  Their product is roughly what a
    naive regression mistakes for a price effect.
    """
    panel = subsample_products(get_panel(min_weeks=min_weeks), n_products, seed)
    cfg = config or SyntheticConfig(confounding_price=confounding_price,
                                    confounding_demand=confounding_demand, seed=seed)
    # make_synthetic_panel already returns a *feature frame*: it engineers the controls
    # from the real panel, then overwrites price and quantity and repairs the
    # price-derived columns.  Running add_features over it again would collide on every
    # engineered column, so do not.
    df = make_synthetic_panel(panel, cfg)
    df.attrs["config"] = cfg
    df.attrs["source"] = "semi-synthetic (UCI prices, simulated demand)"
    return df


def frame_summary(df: pd.DataFrame) -> pd.Series:
    """One-glance description, for the header cell of a notebook."""
    s = {
        "rows": len(df),
        "products": df["stock_code"].nunique(),
        "weeks": df["week"].nunique(),
        "first_week": str(pd.Timestamp(df["week"].min()).date()),
        "last_week": str(pd.Timestamp(df["week"].max()).date()),
        "median_weeks_per_product": float(
            df.groupby("stock_code", observed=True).size().median()),
        "sd_log_price_within_product": float(
            df.groupby("stock_code", observed=True)["log_price"].std().median()),
        "source": df.attrs.get("source", "?"),
    }
    if "true_elasticity" in df:
        s["true_mean_elasticity"] = round(float(df["true_elasticity"].mean()), 4)
        s["true_sd_elasticity"] = round(float(df["true_elasticity"].std()), 4)
    return pd.Series(s)
