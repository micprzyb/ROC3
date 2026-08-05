"""Validation of a real randomized insurance price-test table."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
import pandas as pd
from .core import as_arm_index

@dataclass(frozen=True)
class ExperimentColumns:
    quote_id: str = 'quote_id'
    customer_id: str = 'customer_id'
    timestamp: str = 'quote_timestamp'
    eligible: str = 'eligible'
    arm: str = 'arm'
    purchase: str = 'purchase'
    pi_high: str = 'pi_H'
    pi_control: str = 'pi_C'
    pi_low: str = 'pi_L'
    assigned_multiplier: str = 'assigned_multiplier'
    base_premium: str = 'base_premium'
    displayed_premium: str = 'displayed_premium'
    tariff_version: str = 'tariff_version'
    contribution_high: str = 'contribution_H'
    contribution_control: str = 'contribution_C'
    contribution_low: str = 'contribution_L'

    def required_for_shape(self) -> list[str]:
        return [self.quote_id, self.customer_id, self.timestamp, self.eligible, self.arm, self.purchase, self.pi_high, self.pi_control, self.pi_low]

    def required_for_policy(self) -> list[str]:
        return self.required_for_shape() + [self.contribution_high, self.contribution_control, self.contribution_low]

@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str

@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    n_rows: int = 0
    n_eligible: int = 0
    n_buyers: int = 0
    n_customers: int = 0
    arm_quotes: dict[str, int] = field(default_factory=dict)
    arm_buyers: dict[str, int] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any((i.severity == 'error' for i in self.issues))

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, code, message))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.__dict__ for i in self.issues])
FORBIDDEN_FEATURE_NAMES = {'arm', 'treatment', 'assigned_multiplier', 'displayed_premium', 'experimental_price', 'purchase', 'converted', 'bought', 'policy_id', 'purchase_timestamp', 'payment_method_after_purchase'}

def validate_feature_list(feature_columns: Iterable[str], *, explicit_allowed_post_assignment: Iterable[str]=()) -> None:
    allowed = {str(x).lower() for x in explicit_allowed_post_assignment}
    bad = []
    for column in feature_columns:
        low = str(column).lower()
        if low in FORBIDDEN_FEATURE_NAMES and low not in allowed:
            bad.append(str(column))
        if any((tok in low for tok in ('after_purchase', 'post_treatment', 'post_assignment'))) and low not in allowed:
            bad.append(str(column))
    if bad:
        raise ValueError('feature list contains likely treatment/outcome leakage: ' + ', '.join(sorted(set(bad))))

def validate_experiment_table(df: pd.DataFrame, columns: ExperimentColumns=ExperimentColumns(), *, require_policy_columns: bool=False, propensity_tolerance: float=1e-08) -> ValidationReport:
    report = ValidationReport(n_rows=len(df))
    required = columns.required_for_policy() if require_policy_columns else columns.required_for_shape()
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add('error', 'missing_columns', f'Missing required columns: {missing}')
        return report
    if df[columns.quote_id].isna().any():
        report.add('error', 'missing_quote_id', 'quote_id contains missing values')
    dup = int(df[columns.quote_id].duplicated().sum())
    if dup:
        report.add('error', 'duplicate_quote_id', f'Found {dup} duplicate quote_id rows')
    eligible = pd.to_numeric(df[columns.eligible], errors='coerce')
    if eligible.isna().any() or not eligible.dropna().isin([0, 1]).all():
        report.add('error', 'invalid_eligible', 'eligible must contain only 0/1')
        eligible_mask = np.ones(len(df), bool)
    else:
        eligible_mask = eligible.to_numpy(int) == 1
    report.n_eligible = int(eligible_mask.sum())
    purchase = pd.to_numeric(df[columns.purchase], errors='coerce')
    if purchase.isna().any() or not purchase.dropna().isin([0, 1]).all():
        report.add('error', 'invalid_purchase', 'purchase must contain only 0/1')
        purchase_arr = np.zeros(len(df), int)
    else:
        purchase_arr = purchase.to_numpy(int)
    report.n_buyers = int(np.sum(eligible_mask & (purchase_arr == 1)))
    try:
        arm = as_arm_index(df[columns.arm].to_numpy())
    except ValueError as exc:
        report.add('error', 'invalid_arm', str(exc))
        arm = np.zeros(len(df), int)
    pi = df[[columns.pi_high, columns.pi_control, columns.pi_low]].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    if np.any(~np.isfinite(pi)):
        report.add('error', 'invalid_propensity', 'propensities contain missing/non-finite values')
    else:
        if np.any(pi <= 0):
            report.add('error', 'nonpositive_propensity', 'all logged propensities must be positive')
        if not np.allclose(pi.sum(1), 1, atol=propensity_tolerance, rtol=propensity_tolerance):
            report.add('error', 'propensity_sum', 'each propensity row must sum to one')
        assigned = pi[np.arange(len(df)), arm]
        if np.any(assigned <= 0):
            report.add('error', 'impossible_assignment', 'observed arm has zero logged propensity')
    parsed_time = pd.to_datetime(df[columns.timestamp], errors='coerce', utc=True)
    if parsed_time.isna().any():
        report.add('error', 'invalid_timestamp', 'quote_timestamp contains unparseable values')
    if df[columns.customer_id].isna().any():
        report.add('error', 'missing_customer_id', 'customer_id contains missing values')
    report.n_customers = int(df[columns.customer_id].nunique(dropna=True))
    for idx, label in enumerate(('H', 'C', 'L')):
        mask = eligible_mask & (arm == idx)
        report.arm_quotes[label] = int(mask.sum())
        report.arm_buyers[label] = int(np.sum(mask & (purchase_arr == 1)))
        if report.arm_quotes[label] == 0:
            report.add('error', 'empty_arm', f'No eligible observations in arm {label}')
        if report.arm_buyers[label] == 0:
            report.add('error', 'empty_buyer_arm', f'No buyers in arm {label}')
    if report.n_buyers < 500:
        report.add('warning', 'low_buyer_count', 'Fewer than 500 buyers; only global or very strongly pooled elasticity models are credible.')
    if require_policy_columns:
        contribution = df[[columns.contribution_high, columns.contribution_control, columns.contribution_low]].apply(pd.to_numeric, errors='coerce')
        if contribution.isna().any().any():
            report.add('error', 'invalid_contribution', 'contribution columns contain missing/non-numeric values')
        if (contribution[columns.contribution_control] <= 0).any():
            report.add('warning', 'nonpositive_control_margin', 'Some control contributions are nonpositive')
    repeated = int(df.loc[eligible_mask, columns.customer_id].duplicated().sum())
    if repeated:
        report.add('warning', 'repeated_customers', f'{repeated} eligible rows repeat a customer; folds and bootstraps must cluster by customer/household.')
    return report

def eligible_rows(df: pd.DataFrame, columns: ExperimentColumns=ExperimentColumns()) -> pd.DataFrame:
    return df.loc[pd.to_numeric(df[columns.eligible], errors='raise').astype(int) == 1].copy()

def buyer_rows(df: pd.DataFrame, columns: ExperimentColumns=ExperimentColumns()) -> pd.DataFrame:
    e = eligible_rows(df, columns)
    return e.loc[pd.to_numeric(e[columns.purchase], errors='raise').astype(int) == 1].copy()

def propensity_matrix(df: pd.DataFrame, columns: ExperimentColumns=ExperimentColumns()) -> np.ndarray:
    return df[[columns.pi_high, columns.pi_control, columns.pi_low]].to_numpy(float)

def contribution_matrix(df: pd.DataFrame, columns: ExperimentColumns=ExperimentColumns()) -> np.ndarray:
    return df[[columns.contribution_high, columns.contribution_control, columns.contribution_low]].to_numpy(float)
