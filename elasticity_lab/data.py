"""
elasticity_lab.data
===================

From the raw UCI *Online Retail II* workbook to a modelling-ready product-week panel.

Source
------
Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/502/online+retail+ii> — 1,067,371 transaction lines
from a UK online giftware wholesaler, 2009-12-01 to 2011-12-09.  Public, no registration.

Why this dataset
----------------
Estimating a price elasticity needs price *variation*.  This one has a great deal of it:
after cleaning, 81% of the 3,218 retained products show a within-product standard
deviation of log price above 0.10, and only 10% are effectively fixed-price.  That variation is observational, not
experimental, which is the whole difficulty — see :mod:`elasticity_lab.simulate` for how
we grade methods when the truth is unknowable.

The price-measurement trap
--------------------------
The obvious price for a product-week is ``revenue / quantity``.  **Do not use it.**  In
this data a bigger line item gets a lower unit price — regressing within product-week,

    d(log unit price) / d(log line quantity) = −0.1212   (t = −500, n = 1,037,081)

so any week containing a large order mechanically shows a *low* average price and a *high*
quantity.  That is **division bias**: it manufactures a negative price-quantity
relationship out of nothing.  Measured on this panel it inflates the fixed-effects
elasticity from **1.571 to 2.004**, a 27.6% overstatement of price sensitivity.

The panel therefore carries three price columns, and the default is the **modal** line
price, which is not a function of the week's quantity:

===================  ==========================================================
``price_modal``      most common posted line price that week — **the default**
``price_median``     median line price; robust, still not quantity-weighted
``price_unitvalue``  revenue / quantity — kept only to *demonstrate* the bias
===================  ==========================================================
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DATA_URL", "download", "load_transactions", "clean_transactions",
           "build_panel", "get_panel", "PANEL_SCHEMA"]

DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
_ROOT = Path(__file__).resolve().parents[1] / "data"

#: Stock codes that are not products (postage, fees, manual adjustments, samples).
NON_PRODUCT_CODES = {
    "POST", "D", "M", "DOT", "CRUK", "BANK CHARGES", "S", "AMAZONFEE",
    "ADJUST", "ADJUST2", "TEST001", "TEST002", "PADS", "GIFT", "B", "C2",
}

PANEL_SCHEMA = {
    "stock_code": "product identifier",
    "week": "Monday of the ISO week",
    "qty": "units sold that week (the outcome)",
    "revenue": "total revenue that week",
    "n_lines": "number of order lines",
    "n_customers": "distinct customers who bought",
    "price_modal": "modal posted line price — the default price measure",
    "price_median": "median line price",
    "price_unitvalue": "revenue / qty — biased, kept for demonstration only",
}


def download(dest: Path | None = None, *, force: bool = False) -> Path:
    """Fetch and unzip the workbook.  Returns the path to the .xlsx."""
    dest = Path(dest) if dest else _ROOT
    dest.mkdir(parents=True, exist_ok=True)
    xlsx = dest / "online_retail_II.xlsx"
    if xlsx.exists() and not force:
        return xlsx
    zpath = dest / "online_retail_ii.zip"
    if not zpath.exists() or force:
        urllib.request.urlretrieve(DATA_URL, zpath)
    import zipfile
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    return xlsx


def load_transactions(path: Path | None = None, *, cache: bool = True) -> pd.DataFrame:
    """Load all transaction lines.  Caches to parquet, because the xlsx takes ~50 s."""
    root = _ROOT if path is None else Path(path).parent
    pq = root / "online_retail_ii.parquet"
    if cache and pq.exists():
        return pd.read_parquet(pq)
    xlsx = Path(path) if path else download()
    xl = pd.ExcelFile(xlsx)
    df = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    for c in ("Invoice", "StockCode", "Description", "Country"):
        df[c] = df[c].astype(str)
    if cache:
        df.to_parquet(pq, index=False)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep genuine, positively-priced sales of real products.

    Drops returns (``Quantity <= 0``), credit notes (invoice starting ``C``), zero or
    negative prices, and the non-product stock codes listed in :data:`NON_PRODUCT_CODES`.
    """
    d = df[(df.Quantity > 0) & (df.Price > 0)].copy()
    d = d[~d.Invoice.str.startswith("C")]
    d = d[~d.StockCode.str.upper().str.strip().isin(NON_PRODUCT_CODES)]
    # a handful of stock codes are free-text notes rather than SKUs
    d = d[d.StockCode.str.len().between(4, 12)]
    d["week"] = d.InvoiceDate.dt.to_period("W").dt.start_time
    return d


def build_panel(d: pd.DataFrame, *, min_weeks: int = 20,
                min_weekly_units: int = 1) -> pd.DataFrame:
    """Aggregate cleaned transactions to a product-week panel.

    ``min_weeks`` keeps only products active in at least that many distinct weeks, so
    every retained product carries enough within-product price variation to identify
    anything at all.
    """
    key = d.groupby(["StockCode", "week"], observed=True)
    g = key.agg(qty=("Quantity", "sum"),
                n_lines=("Quantity", "size"),
                n_customers=("CustomerID", "nunique")).reset_index()
    g["revenue"] = (d.assign(_r=d.Quantity * d.Price)
                     .groupby(["StockCode", "week"], observed=True)["_r"].sum().values)
    g["price_median"] = key.Price.median().values
    g["price_modal"] = key.Price.agg(lambda s: s.mode().iloc[0]).values
    g["price_unitvalue"] = g.revenue / g.qty
    g = g.rename(columns={"StockCode": "stock_code"})
    g = g[g.qty >= min_weekly_units]
    n = g.groupby("stock_code").size()
    g = g[g.stock_code.isin(n[n >= min_weeks].index)].copy()
    g = g.sort_values(["stock_code", "week"]).reset_index(drop=True)
    return g[list(PANEL_SCHEMA)]


def get_panel(*, min_weeks: int = 20, cache: bool = True,
              rebuild: bool = False) -> pd.DataFrame:
    """One-call reproducible panel.  Cached on disk keyed by the build parameters."""
    pq = _ROOT / f"panel_minweeks{min_weeks}.parquet"
    if pq.exists() and cache and not rebuild:
        return pd.read_parquet(pq)
    panel = build_panel(clean_transactions(load_transactions()), min_weeks=min_weeks)
    if cache:
        _ROOT.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(pq, index=False)
    return panel


def panel_fingerprint(panel: pd.DataFrame) -> str:
    """Short hash of the panel, so a notebook can assert it ran on the same data."""
    h = hashlib.sha256()
    h.update(str(panel.shape).encode())
    h.update(np.ascontiguousarray(panel.qty.values).tobytes())
    h.update(np.ascontiguousarray(panel.price_modal.values).tobytes())
    return h.hexdigest()[:12]


def division_bias_diagnostic(d: pd.DataFrame) -> dict:
    """Quantify the mechanical price-quantity link that makes unit value unusable.

    Regresses log line price on log line quantity *within* product-week.  A negative
    slope means bigger orders are discounted, so ``revenue / quantity`` moves with the
    week's quantity for reasons that have nothing to do with demand.
    """
    import statsmodels.api as sm

    x = d[["StockCode", "week", "Quantity", "Price"]].copy()
    x["lq"] = np.log(x.Quantity)
    x["lp"] = np.log(x.Price)
    k = x.groupby(["StockCode", "week"], observed=True)
    x["lp_dm"] = x.lp - k.lp.transform("mean")
    x["lq_dm"] = x.lq - k.lq.transform("mean")
    m = sm.OLS(x.lp_dm.values, sm.add_constant(x.lq_dm.values)).fit()
    return {"slope": float(m.params[1]), "t_stat": float(m.tvalues[1]), "n": int(len(x))}
