"""
elasticity_lab.splits
=====================

Cross-validation designs for a **product × week panel**.

This module exists because the default choice — ``KFold(shuffle=True)`` — is wrong here in
two independent ways, and each inflates the validation score enough to corrupt a
hyperparameter search.

**Leak 1: time.**  The features include lags and rolling means.  A random fold puts week
40 in training and week 39 in validation, so the model is asked to interpolate a series it
has already seen either side of.  Worse, the product-level summaries (``prod_*``) and the
market controls are computed across the whole panel, so a random split lets the model see
the future in aggregate even when it cannot see it directly.

**Leak 2: product identity.**  A random fold puts the same product in training and
validation.  With ~3,200 products over 104 weeks, a flexible model can memorise the
product's level and score well without learning anything transferable.

Which leak matters depends on the question being asked, so this module provides three
designs rather than one:

:class:`RollingOriginSplit`
    Train on weeks ``< t``, validate on the block ``[t, t+h)``, roll forward.  This is the
    honest analogue of deployment: you always predict forward.  Use it for anything whose
    purpose is forecasting, and as the default HPO substrate.
:class:`GroupedProductSplit`
    Hold out whole products.  Answers "does this generalise to a SKU I have never seen?"
:class:`PurgedRollingSplit`
    Rolling origin with an **embargo** of ``gap`` weeks between train and validation, so
    that a validation row's lag window cannot overlap the training period.  With 12-week
    rolling features an embargo of 12 removes the last route by which the past leaks
    forward.

Every splitter yields ``(train_idx, valid_idx)`` positional arrays and exposes
``train_end`` for each fold, which the feature builder needs so that ``prod_*`` statistics
are computed on training weeks only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = ["RollingOriginSplit", "GroupedProductSplit", "PurgedRollingSplit",
           "NaiveKFold", "describe_split", "split_report"]


@dataclass
class RollingOriginSplit:
    """Expanding- (or sliding-) window splits over the week axis.

    Parameters
    ----------
    n_splits : number of folds
    horizon : validation block length, in weeks
    min_train_weeks : weeks required before the first fold
    window : ``None`` for an expanding window, or an integer for a sliding one
    """

    n_splits: int = 5
    horizon: int = 8
    min_train_weeks: int = 26
    window: int | None = None

    def _weeks(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(sorted(pd.unique(df["week"])))

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        weeks = self._weeks(df)
        n = len(weeks)
        last_start = n - self.horizon
        starts = np.linspace(self.min_train_weeks, last_start, self.n_splits).astype(int)
        starts = np.unique(starts)
        wk = df["week"].values
        for s in starts:
            t0 = weeks[s]
            t1 = weeks[min(s + self.horizon, n - 1)]
            lo = weeks[max(0, s - self.window)] if self.window else weeks[0]
            tr = np.flatnonzero((wk >= lo) & (wk < t0))
            va = np.flatnonzero((wk >= t0) & (wk < t1))
            if len(tr) and len(va):
                yield tr, va

    def train_ends(self, df: pd.DataFrame) -> list[pd.Timestamp]:
        return [pd.Timestamp(df["week"].values[tr].max()) for tr, _ in self.split(df)]

    def get_n_splits(self, df: pd.DataFrame) -> int:
        return sum(1 for _ in self.split(df))


@dataclass
class PurgedRollingSplit(RollingOriginSplit):
    """Rolling origin with an embargo, so lag windows cannot straddle the boundary."""

    gap: int = 12

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        weeks = np.array(sorted(pd.unique(df["week"])))
        n = len(weeks)
        starts = np.unique(np.linspace(self.min_train_weeks + self.gap,
                                       n - self.horizon, self.n_splits).astype(int))
        wk = df["week"].values
        for s in starts:
            t0 = weeks[s]
            t1 = weeks[min(s + self.horizon, n - 1)]
            tr_end = weeks[max(0, s - self.gap)]
            lo = weeks[max(0, s - self.gap - self.window)] if self.window else weeks[0]
            tr = np.flatnonzero((wk >= lo) & (wk < tr_end))
            va = np.flatnonzero((wk >= t0) & (wk < t1))
            if len(tr) and len(va):
                yield tr, va


@dataclass
class GroupedProductSplit:
    """Hold out whole products; nothing about a validation SKU is ever seen in training."""

    n_splits: int = 5
    seed: int = 0

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        codes = pd.unique(df["stock_code"])
        rng = np.random.default_rng(self.seed)
        folds = rng.permutation(len(codes)) % self.n_splits
        assign = dict(zip(codes, folds))
        f = df["stock_code"].map(assign).values
        for k in range(self.n_splits):
            yield np.flatnonzero(f != k), np.flatnonzero(f == k)

    def get_n_splits(self, df: pd.DataFrame) -> int:
        return self.n_splits


@dataclass
class NaiveKFold:
    """Shuffled K-fold — **wrong here**, provided so the leak can be measured."""

    n_splits: int = 5
    seed: int = 0

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self.seed)
        f = rng.permutation(len(df)) % self.n_splits
        for k in range(self.n_splits):
            yield np.flatnonzero(f != k), np.flatnonzero(f == k)

    def get_n_splits(self, df: pd.DataFrame) -> int:
        return self.n_splits


def describe_split(df: pd.DataFrame, splitter) -> pd.DataFrame:
    """Per-fold sizes, date ranges, and how much the folds actually overlap."""
    rows = []
    for i, (tr, va) in enumerate(splitter.split(df)):
        w_tr, w_va = df["week"].values[tr], df["week"].values[va]
        p_tr = set(df["stock_code"].values[tr])
        p_va = set(df["stock_code"].values[va])
        rows.append({
            "fold": i, "n_train": len(tr), "n_valid": len(va),
            "train_from": pd.Timestamp(w_tr.min()).date(),
            "train_to": pd.Timestamp(w_tr.max()).date(),
            "valid_from": pd.Timestamp(w_va.min()).date(),
            "valid_to": pd.Timestamp(w_va.max()).date(),
            "weeks_shared": len(set(w_tr) & set(w_va)),
            "products_shared_pct": 100 * len(p_tr & p_va) / max(len(p_va), 1),
        })
    return pd.DataFrame(rows)


def split_report(df: pd.DataFrame, splitters: dict) -> pd.DataFrame:
    """Compare designs on the two leak channels at once."""
    out = []
    for name, sp in splitters.items():
        t = describe_split(df, sp)
        out.append({"design": name, "folds": len(t),
                    "median_n_train": int(t.n_train.median()),
                    "median_n_valid": int(t.n_valid.median()),
                    "weeks_shared_train_valid": int(t.weeks_shared.sum()),
                    "pct_valid_products_also_in_train": round(
                        float(t.products_shared_pct.mean()), 1)})
    return pd.DataFrame(out)
