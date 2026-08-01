"""
elasticity_lab.insurance
========================

Price testing for **mandatory motor insurance**: one product, everyone must buy, and the
only question is whether they buy from us.  See [`docs/INSURANCE.md`](../docs/INSURANCE.md)
for why this is a different problem rather than the same one with different data.

The four structural facts the models here are built around
----------------------------------------------------------
1. The outcome is **binary conversion**, not a count.  Category demand is fixed; the price
   moves share.
2. Demand is **bounded**, so with a logit conversion curve
   ``s = sigmoid(a(x) - beta(x) log p)`` the elasticity is

       eps(x, p) = beta(x) * (1 - s(x, p))

   — mechanically tied to the conversion rate.  Elasticity is therefore heterogeneous
   **even when beta is constant**, and it varies along the price ladder for one customer.
3. Allocation is **unequal** (0.1 / 0.8 / 0.1).  The arm posterior among converters is
   ``b_k ∝ pi_k s_k``, so the 80% arm dominates and ``b`` is **not** monotone even though
   demand is.  The constraint that survives is ``b_1/pi_1 >= b_2/pi_2 >= b_3/pi_3``, which
   is checkable because ``pi`` is known by design.
4. The objective is **profit**, ``s(p)(p - c)``, whose optimum is the Lerner condition
   ``(p-c)/p = 1/eps``.  Accuracy in ``eps`` matters near 1 and hardly at all when large.

Contents
--------
``MotorQuoteConfig`` / ``make_quote_panel``
    A quote-level simulator with a known ``true_elasticity`` per row.
``ConversionGLM``, ``ConversionGBM``, ``ArmPosteriorModel``, ``TLearnerConversion``
    The model set.  ``ArmPosteriorModel`` is the price-test-first construction with the
    known allocation substituted for an estimated propensity.
``profit_curve``, ``optimal_price``, ``policy_profit``
    The decision layer: what the elasticity is *for*.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .models import N_THREADS
from .pricelevels import isotonic_decreasing

__all__ = ["MotorQuoteConfig", "make_quote_panel", "QUOTE_FEATURES",
           "ConversionModel", "ConversionGLM", "ConversionGBM", "TLearnerConversion",
           "ArmPosteriorModel", "profit_curve", "optimal_price", "policy_profit",
           "realised_elasticity_by_bin", "arm_posterior_constraint_check"]


# =======================================================================================
# simulator
# =======================================================================================
@dataclass
class MotorQuoteConfig:
    """A motor book and its price test.

    The defaults describe a plausible mid-market motor account: about a third of quotes
    convert at the control price, and the competitive response is strong enough that the
    average conversion elasticity sits above 1 (so a finite profit optimum exists).
    """

    n_quotes: int = 200_000
    #: arm price multipliers, cheapest first
    arm_multipliers: tuple = (0.9, 1.0, 1.1)
    #: allocation.  The 0.1/0.8/0.1 of a live test: most traffic stays on control.
    arm_probs: tuple = (0.1, 0.8, 0.1)
    #: mean of the logit slope beta.  NB the ELASTICITY is beta*(1-s), so it is smaller.
    beta_mean: float = 3.2
    #: across-customer sd of beta, the part of elasticity heterogeneity that is not
    #: mechanically induced by the conversion rate
    beta_sd: float = 0.6
    #: how strongly beta depends on observed features (shoppers are more price-sensitive)
    beta_shopper_slope: float = 0.9
    #: target mean conversion at the control price
    base_conversion: float = 0.33
    #: expected claims cost as a fraction of the control premium, on average
    loss_ratio: float = 0.68
    #: sd of the log technical premium across customers
    premium_sd: float = 0.45
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


#: The rating and behavioural features a quote carries.  Deliberately small and readable:
#: this is a simulator for studying estimators, not a rating model.
QUOTE_FEATURES = [
    "driver_age", "vehicle_age", "vehicle_group", "region_risk", "ncd_years",
    "prior_claims", "annual_mileage", "is_renewal", "tenure_years",
    "shopping_intensity", "channel_aggregator", "log_technical_premium",
    "log_competitor_index",
]


def make_quote_panel(cfg: MotorQuoteConfig | None = None) -> pd.DataFrame:
    """Simulate a quote-level price test with a known elasticity on every row.

    Returns one row per quote with :data:`QUOTE_FEATURES`, plus

    ``arm``            0-based, cheapest first
    ``arm_prob``       the **known** assignment probability — what randomisation buys
    ``log_price``      log of the premium actually quoted
    ``converted``      the binary outcome
    ``claims_cost``    expected claims for this risk (known to the insurer, not the customer)
    ``true_beta``      the customer's logit slope
    ``true_s``         the true conversion probability at the price quoted
    ``true_elasticity``       ``beta * (1 - s)`` at the price quoted
    ``true_elasticity_control``  the same at the CONTROL price — **use this for anything
                              grouped**, see the note where it is defined
    """
    cfg = cfg or MotorQuoteConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_quotes
    mult = np.asarray(cfg.arm_multipliers, float)
    pi = np.asarray(cfg.arm_probs, float)
    pi = pi / pi.sum()
    K = len(mult)

    z = lambda a: (a - a.mean()) / max(a.std(), 1e-12)            # noqa: E731

    # ---- the book ----------------------------------------------------------------------
    driver_age = np.clip(rng.gamma(9.0, 4.2, n) + 17, 17, 90)
    vehicle_age = np.clip(rng.gamma(2.2, 3.0, n), 0, 25)
    vehicle_group = np.clip(rng.normal(22, 9, n), 1, 50)
    region_risk = rng.normal(0, 1, n)
    ncd_years = np.clip(rng.poisson(4.5, n), 0, 15)
    prior_claims = rng.poisson(0.22, n)
    annual_mileage = np.clip(rng.lognormal(9.1, 0.45, n), 1000, 60000)
    is_renewal = (rng.random(n) < 0.62).astype(float)
    tenure_years = np.where(is_renewal > 0, np.clip(rng.gamma(1.8, 2.0, n), 0, 25), 0.0)
    channel_aggregator = (rng.random(n) < 0.55).astype(float)
    # shoppers: aggregator users, new business, low tenure, young
    shopping_intensity = (0.9 * channel_aggregator + 0.6 * (1 - is_renewal)
                          - 0.35 * z(tenure_years) - 0.30 * z(driver_age)
                          + 0.5 * rng.normal(0, 1, n))

    # ---- the technical premium (what a rating model would charge) -----------------------
    log_tech = (np.log(420.0)
                + 0.55 * z(-np.log(driver_age)) + 0.30 * z(vehicle_group)
                + 0.35 * z(region_risk) + 0.28 * z(np.log(annual_mileage))
                + 0.40 * z(prior_claims) - 0.25 * z(ncd_years)
                + cfg.premium_sd * 0.35 * rng.normal(0, 1, n))
    # where the market sits relative to us, which drives conversion at any given price
    log_comp = log_tech + 0.22 * rng.normal(0, 1, n) - 0.10 * z(shopping_intensity)

    d = pd.DataFrame({
        "driver_age": driver_age, "vehicle_age": vehicle_age,
        "vehicle_group": vehicle_group, "region_risk": region_risk,
        "ncd_years": ncd_years.astype(float), "prior_claims": prior_claims.astype(float),
        "annual_mileage": annual_mileage, "is_renewal": is_renewal,
        "tenure_years": tenure_years, "shopping_intensity": shopping_intensity,
        "channel_aggregator": channel_aggregator,
        "log_technical_premium": log_tech, "log_competitor_index": log_comp,
    })

    # ---- price sensitivity ---------------------------------------------------------------
    beta = np.clip(cfg.beta_mean
                   + cfg.beta_shopper_slope * z(shopping_intensity)
                   + cfg.beta_sd * rng.normal(0, 1, n), 0.3, 12.0)
    d["true_beta"] = beta

    # ---- the randomisation ----------------------------------------------------------------
    arm = rng.choice(K, size=n, p=pi)
    d["arm"] = arm
    d["arm_prob"] = pi[arm]
    d["arm_log_multiplier"] = np.log(mult)[arm]
    d["log_price"] = log_tech + np.log(mult)[arm]

    # ---- conversion ------------------------------------------------------------------------
    # a(x) is calibrated so the mean conversion at the CONTROL price hits base_conversion
    a_raw = (1.3 * (log_comp - log_tech) * 4.0 - 0.25 * z(vehicle_age)
             + 0.30 * is_renewal + 0.20 * z(tenure_years))
    target = np.log(cfg.base_conversion / (1 - cfg.base_conversion))
    a = a_raw - a_raw.mean() + target + beta * log_tech          # so eta is O(1) at mult=1
    d["_a"] = a
    eta = a - beta * d["log_price"].to_numpy()
    s = 1.0 / (1.0 + np.exp(-eta))
    d["true_s"] = s
    d["converted"] = (rng.random(n) < s).astype(int)
    #: the estimand AT THE PRICE ACTUALLY QUOTED.  It depends on the arm through s -- the
    #: same customer has a different elasticity at -10% and at +10%.
    d["true_elasticity"] = beta * (1.0 - s)
    #: the estimand AT THE CONTROL PRICE, the same for a customer whichever arm they got.
    #: Group-level diagnostics must use this one.  Binning on the arm-dependent column
    #: silently selects a different population into each arm -- for a given eps, a customer
    #: in the dear arm has a lower beta than one in the cheap arm -- and the realised arc
    #: elasticity then comes out badly biased (measured: 0.31 against a true 0.52 in the
    #: least-elastic bin).  The trap is not specific to the simulator: any model whose
    #: `elasticity()` is evaluated at the price the customer happened to be quoted has it.
    s_ctrl = 1.0 / (1.0 + np.exp(-(a - beta * log_tech)))
    d["true_s_control"] = s_ctrl
    d["true_elasticity_control"] = beta * (1.0 - s_ctrl)

    # ---- economics -------------------------------------------------------------------------
    d["claims_cost"] = np.exp(log_tech) * cfg.loss_ratio * np.exp(
        0.25 * rng.normal(0, 1, n) - 0.5 * 0.25 ** 2)

    d.attrs["cfg"] = cfg.to_dict()
    d.attrs["arm_probs"] = pi
    d.attrs["arm_multipliers"] = mult
    d.attrs["K"] = K
    return d


def true_conversion_matrix(d: pd.DataFrame) -> np.ndarray:
    """``(n, K)`` of the TRUE conversion probability of each quote in each arm.

    Only the simulator knows this.  It is the oracle for the counterfactual questions —
    what would this customer have done at the other two prices — which is exactly what no
    amount of data supplies for any single row.
    """
    mult = np.asarray(d.attrs["arm_multipliers"], float)
    a = d["_a"].to_numpy()[:, None]
    beta = d["true_beta"].to_numpy()[:, None]
    logp = d["log_technical_premium"].to_numpy()[:, None] + np.log(mult)[None, :]
    return 1.0 / (1.0 + np.exp(-(a - beta * logp)))


# =======================================================================================
# the decision layer — what the elasticity is FOR
# =======================================================================================
def profit_curve(s: np.ndarray, price: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Expected profit per quote: ``s * (price - cost)``."""
    return np.asarray(s, float) * (np.asarray(price, float) - np.asarray(cost, float))


def optimal_price(elasticity_fn, price0: np.ndarray, cost: np.ndarray, *,
                  n_iter: int = 25, lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    """Solve the Lerner condition ``(p - c)/p = 1/eps(p)`` by fixed-point iteration.

    ``eps`` depends on ``p`` (§1.2 of ``docs/INSURANCE.md``), so this is implicit and has
    no closed form.  ``elasticity_fn(price)`` must return the elasticity at that price.

    Where ``eps <= 1`` there is no interior optimum — profit rises with price without
    bound in the model — so the price is capped at ``hi * price0`` and the caller should
    treat those rows as "the model says charge more than the test explored", which is a
    statement about extrapolation rather than a recommendation.
    """
    p = np.asarray(price0, float).copy()
    c = np.asarray(cost, float)
    for _ in range(n_iter):
        eps = np.asarray(elasticity_fn(p), float)
        with np.errstate(divide="ignore", invalid="ignore"):
            target = np.where(eps > 1.0 + 1e-6, c / (1.0 - 1.0 / eps), hi * price0)
        target = np.clip(target, lo * np.asarray(price0, float),
                         hi * np.asarray(price0, float))
        p = 0.5 * p + 0.5 * target                     # damped, for stability
    return p


def policy_profit(d: pd.DataFrame, chosen_arm: np.ndarray) -> dict:
    """Estimate the profit of an arm-choosing policy, by inverse-propensity weighting.

    A policy that picks arm ``k(x)`` for each quote can only be evaluated on the quotes that
    happened to receive that arm — so each contributes with weight ``1/pi_k``.  This is the
    standard off-policy value estimate and it is **unbiased here**, because the allocation
    is known and every arm has positive probability for every ``x``.

    With allocation 0.1/0.8/0.1 the side arms carry weight 10, so a policy that mostly
    recommends them is evaluated on a tenth of the data; ``ess`` reports the damage.
    """
    arm = d["arm"].to_numpy()
    pi = d["arm_prob"].to_numpy()
    mult = np.asarray(d.attrs["arm_multipliers"], float)
    price = np.exp(d["log_technical_premium"].to_numpy()) * mult[np.asarray(chosen_arm)]
    match = (arm == np.asarray(chosen_arm))
    w = np.where(match, 1.0 / pi, 0.0)
    profit = d["converted"].to_numpy() * (price - d["claims_cost"].to_numpy())
    tot = w.sum()
    return {"profit_per_quote": float((w * profit).sum() / tot) if tot > 0 else np.nan,
            "conversion": float((w * d["converted"].to_numpy()).sum() / tot)
            if tot > 0 else np.nan,
            "matched_quotes": int(match.sum()),
            "ess": float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12)),
            "pct_arm_cheap": float(np.mean(np.asarray(chosen_arm) == 0)),
            "pct_arm_dear": float(np.mean(np.asarray(chosen_arm) == len(mult) - 1))}


def realised_elasticity_by_bin(d: pd.DataFrame, eps_hat: np.ndarray, *,
                               n_bins: int = 5) -> pd.DataFrame:
    """GATES for a randomised price test: bin by predicted elasticity, measure the truth.

    Because the arms are randomised, the realised conversion rate in each arm **within a
    bin** is an unbiased estimate of that bin's demand curve — no modelling required.  The
    arc elasticity between the cheap and dear arms is then a direct, assumption-free check
    on the model's ranking.  This is the strongest diagnostic available in this setting and
    it exists only because the test is randomised.

    **``eps_hat`` must be evaluated at a common reference price, not at the price each
    customer was actually quoted.**  Because the elasticity of a bounded demand curve
    depends on the price (``eps = beta(1-s)``), binning on the quoted-price elasticity puts
    a systematically different population into each arm within a bin, and the realised arc
    comes out biased low — 0.31 against a true 0.52 in the least-elastic bin when this was
    measured.  Compare ``eps_realised`` against a model's prediction *at control*.
    """
    eps_hat = np.asarray(eps_hat, float)
    mult = np.asarray(d.attrs["arm_multipliers"], float)
    if np.std(eps_hat) < 1e-12:
        idx = np.zeros(len(d), int)
        n_bins = 1
    else:
        edges = np.quantile(eps_hat, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(edges, eps_hat, side="right") - 1, 0, n_bins - 1)

    arm = d["arm"].to_numpy()
    conv = d["converted"].to_numpy()
    rows = []
    for b in range(n_bins):
        m = idx == b
        rates, ns = [], []
        for k in range(len(mult)):
            sel = m & (arm == k)
            ns.append(int(sel.sum()))
            rates.append(float(conv[sel].mean()) if sel.sum() > 0 else np.nan)
        rates = np.array(rates)
        # arc elasticity between the extreme arms, straight from the realised rates
        with np.errstate(divide="ignore", invalid="ignore"):
            arc = -(np.log(rates[-1]) - np.log(rates[0])) / (
                np.log(mult[-1]) - np.log(mult[0]))
        # binomial standard error, propagated through the logs
        se = np.sqrt(sum((1 - rates[k]) / max(rates[k] * ns[k], 1e-9)
                         for k in (0, len(mult) - 1))) / (
            np.log(mult[-1]) - np.log(mult[0]))
        rows.append({"bin": b + 1, "n": int(m.sum()),
                     "eps_predicted": float(eps_hat[m].mean()) if m.any() else np.nan,
                     "eps_realised": float(arc), "se": float(se),
                     **{f"conv_arm{k}": rates[k] for k in range(len(mult))},
                     **{f"n_arm{k}": ns[k] for k in range(len(mult))}})
    t = pd.DataFrame(rows)
    t.attrs["monotone"] = bool(t.eps_realised.is_monotonic_increasing) if n_bins > 1 else None
    return t


def arm_posterior_constraint_check(b: np.ndarray, arm_probs) -> dict:
    """Is ``b_k / pi_k`` decreasing?  The constraint that survives unequal allocation.

    Checking ``b_1 >= b_2 >= b_3`` instead — the form the constraint is usually written in —
    is **wrong at 0.1/0.8/0.1**: the control arm carries eight times the traffic, so its
    posterior dominates however elastic demand is.  This function reports both, so the gap
    between them is visible rather than assumed away.
    """
    b = np.asarray(b, float)
    pi = np.asarray(arm_probs, float)
    pi = pi / pi.sum()
    s = b / pi[None, :]
    dec_s = np.all(np.diff(s, axis=1) <= 1e-12, axis=1)
    dec_b = np.all(np.diff(b, axis=1) <= 1e-12, axis=1)
    return {"pct_demand_monotone": float(dec_s.mean()),
            "pct_posterior_monotone": float(dec_b.mean()),
            "note": "the first is the real constraint; the second is what it degenerates "
                    "to only when allocation is equal"}
