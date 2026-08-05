"""Shared real-data helpers for the examples. No synthetic fallback is provided."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from elasticity_only.preprocessing import FeatureSpecification
from elasticity_only.schema import ExperimentColumns


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_frame(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["data"]["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Real randomized experiment table not found: {path}. "
            "This package deliberately has no synthetic fallback."
        )
    fmt = config["data"].get("format", path.suffix.lstrip(".")).lower()
    if fmt in {"parquet", "pq"}:
        return pd.read_parquet(path)
    if fmt in {"csv", "csv.gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {fmt}")


def columns_from_config(config: dict[str, Any]) -> ExperimentColumns:
    c = config["columns"]
    return ExperimentColumns(
        quote_id=c["quote_id"],
        customer_id=c["customer_id"],
        timestamp=c["timestamp"],
        eligible=c["eligible"],
        arm=c["arm"],
        purchase=c["purchase"],
        pi_high=c["pi_high"],
        pi_control=c["pi_control"],
        pi_low=c["pi_low"],
        assigned_multiplier=c.get("assigned_multiplier", "assigned_multiplier"),
        base_premium=c.get("base_premium", "base_premium"),
        displayed_premium=c.get("displayed_premium", "displayed_premium"),
        tariff_version=c.get("tariff_version", "tariff_version"),
        contribution_high=c.get("contribution_high", "contribution_H"),
        contribution_control=c.get("contribution_control", "contribution_C"),
        contribution_low=c.get("contribution_low", "contribution_L"),
    )


def features_from_config(config: dict[str, Any]) -> FeatureSpecification:
    return FeatureSpecification(
        numeric=tuple(config["features"].get("numeric", [])),
        categorical=tuple(config["features"].get("categorical", [])),
    )
