"""Leakage-resistant splitting for repeated insurance quotes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import StratifiedGroupKFold
from .core import as_arm_index
IntArray = NDArray[np.int64]

@dataclass(frozen=True)
class TemporalSplit:
    development_index: IntArray
    holdout_index: IntArray
    cutoff: pd.Timestamp

def make_forward_time_holdout(timestamps: ArrayLike, groups: ArrayLike, *, holdout_fraction: float=0.2) -> TemporalSplit:
    if not 0.05 <= holdout_fraction <= 0.5:
        raise ValueError('holdout_fraction must be between 0.05 and 0.50')
    ts = pd.to_datetime(np.asarray(timestamps), errors='raise', utc=True)
    g = np.asarray(groups)
    if len(ts) != len(g):
        raise ValueError('timestamps and groups must align')
    frame = pd.DataFrame({'group': g, 'timestamp': ts})
    last = frame.groupby('group', sort=False)['timestamp'].max().sort_values()
    n_hold = max(1, int(np.ceil(len(last) * holdout_fraction)))
    hold = set(last.iloc[-n_hold:].index.tolist())
    mask = np.array([x in hold for x in g], bool)
    cutoff = last.iloc[-n_hold]
    return TemporalSplit(np.flatnonzero(~mask).astype(np.int64), np.flatnonzero(mask).astype(np.int64), pd.Timestamp(cutoff))

def assert_no_group_overlap(train_index: ArrayLike, valid_index: ArrayLike, groups: ArrayLike) -> None:
    tr = np.asarray(train_index, int)
    va = np.asarray(valid_index, int)
    g = np.asarray(groups)
    overlap = np.intersect1d(np.unique(g[tr]), np.unique(g[va]))
    if len(overlap):
        raise ValueError(f'group leakage: {len(overlap)} groups occur in train and validation')

def make_buyer_group_folds(arm: ArrayLike, groups: ArrayLike, *, n_splits: int=5, random_state: int=2026) -> list[tuple[IntArray, IntArray]]:
    t = as_arm_index(arm)
    g = np.asarray(groups)
    if len(t) != len(g):
        raise ValueError('arm and groups must align')
    if len(np.unique(g)) < n_splits:
        raise ValueError('fewer unique groups than requested folds')
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy = np.zeros((len(t), 1))
    out = []
    for tr, va in splitter.split(dummy, t, groups=g):
        assert_no_group_overlap(tr, va, g)
        out.append((tr.astype(np.int64), va.astype(np.int64)))
    return out

def rolling_time_splits(timestamps: ArrayLike, groups: ArrayLike, *, n_splits: int=3, min_train_fraction: float=0.5, validation_fraction: float=0.15) -> list[tuple[IntArray, IntArray]]:
    ts = pd.to_datetime(np.asarray(timestamps), errors='raise', utc=True)
    g = np.asarray(groups)
    if len(ts) != len(g):
        raise ValueError('timestamps and groups must align')
    gt = pd.DataFrame({'group': g, 'time': ts}).groupby('group')['time'].max().sort_values()
    vals = gt.index.to_numpy()
    n = len(vals)
    min_train = max(1, int(np.floor(n * min_train_fraction)))
    val_size = max(1, int(np.floor(n * validation_fraction)))
    endpoints = np.linspace(min_train, n - val_size, n_splits, dtype=int)
    out = []
    for end in np.unique(endpoints):
        tg = set(vals[:end].tolist())
        vg = set(vals[end:end + val_size].tolist())
        tr = np.flatnonzero([x in tg for x in g])
        va = np.flatnonzero([x in vg for x in g])
        if len(tr) and len(va):
            assert_no_group_overlap(tr, va, g)
            out.append((tr.astype(np.int64), va.astype(np.int64)))
    return out
