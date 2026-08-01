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
           "realised_elasticity_by_bin", "arm_posterior_constraint_check",
           "to_balanced_gauge", "constraint_triangle_vertices", "GLMThenGBM",
           "MOTOR_MODELS", "true_conversion_matrix", "profit_regret",
           "profit_weight", "lerner_price", "experiment_cost", "ConversionGBMv2",
           "ConversionGLMv2", "AIPWArmEffects", "AnchoredTLearner", "MOTOR_MODELS_V2",
           "ENGINEERED", "add_engineered"]


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
    #: shape of the demand curve.  ``"logit"`` is the world ConversionGLM is exactly right
    #: about, so any benchmark run only there is rigged in its favour.  The others break
    #: that free lunch (PLAN_INSURANCE_V2 I12):
    #:   ``"probit"``  same index, a different link -- mild misspecification
    #:   ``"kinked"``  a shopping-behaviour threshold in the index: below a competitive
    #:                 position the customer does not shop at all, above it they do
    #:   ``"mixed"``   the link's curvature varies across customers, so no single
    #:                 parametric form fits everyone
    link: str = "logit"
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


def _apply_link(eta, link, d=None):
    """Map the linear index to a conversion probability.

    Only ``"logit"`` makes :class:`ConversionGLM` correctly specified; the others exist so
    that "a flexible model beats a GLM" can be a finding rather than an artefact of the
    simulator agreeing with the GLM's functional form.
    """
    eta = np.asarray(eta, float)
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    if link == "probit":
        from scipy.stats import norm
        return norm.cdf(eta * 0.5875)                 # scaled so the two links roughly agree
    if link == "kinked":
        # a shopping threshold: below it the customer barely reacts, above it they do.
        # Piecewise-linear in the index, which no single-index GLM can represent.
        kink = np.where(eta > 0.0, eta, 0.45 * eta)
        return 1.0 / (1.0 + np.exp(-kink))
    if link == "mixed":
        # curvature varies by customer, so no one link fits the book
        if d is None or "shopping_intensity" not in d:
            raise ValueError("the 'mixed' link needs the covariate frame")
        z = d["shopping_intensity"].to_numpy(float)
        t = 0.6 + 0.8 / (1.0 + np.exp(-(z - z.mean()) / max(z.std(), 1e-9)))
        return 1.0 / (1.0 + np.exp(-eta * t))
    raise ValueError(f"unknown link {link!r}")


def _numeric_elasticity(a, beta, log_price, link, d, h=1e-4):
    """``-d log s / d log p`` by central difference — identical treatment for every link."""
    lp = np.asarray(log_price, float)
    up = _apply_link(a - beta * (lp + h), link, d)
    dn = _apply_link(a - beta * (lp - h), link, d)
    return -(np.log(np.maximum(up, 1e-300)) - np.log(np.maximum(dn, 1e-300))) / (2 * h)


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
    d["_link"] = cfg.link
    eta = a - beta * d["log_price"].to_numpy()
    s = _apply_link(eta, cfg.link, d)
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
    # The elasticity is -d log s / d log p, which for a general link is
    # (dS/deta)(-beta)/S -- computed numerically so every link is handled identically and
    # the closed-form logit case is not silently special-cased.
    d["true_s_control"] = _apply_link(a - beta * log_tech, cfg.link, d)
    d["true_elasticity_control"] = _numeric_elasticity(a, beta, log_tech, cfg.link, d)
    d["true_elasticity"] = _numeric_elasticity(
        a, beta, d["log_price"].to_numpy(float), cfg.link, d)

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
    link = d["_link"].iloc[0] if "_link" in d else "logit"
    return np.column_stack([_apply_link(a[:, 0] - beta[:, 0] * logp[:, k], link, d)
                            for k in range(logp.shape[1])])


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
    if tot <= 0:
        return {"profit_per_quote": np.nan, "se": np.nan, "conversion": np.nan,
                "matched_quotes": 0, "ess": 0.0, "pct_arm_cheap": np.nan,
                "pct_arm_dear": np.nan}
    val = float((w * profit).sum() / tot)
    # Standard error of the weighted mean.  Reporting it is not optional here: with only
    # 10% of traffic in each side arm, a policy that recommends them is evaluated on ~9,000
    # effective quotes, and two policies half a currency unit apart are indistinguishable.
    # A ranked table without this invites reading noise as a result.
    se = float(np.sqrt((w ** 2 * (profit - val) ** 2).sum()) / tot)
    return {"profit_per_quote": val, "se": se,
            "conversion": float((w * d["converted"].to_numpy()).sum() / tot),
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


# =======================================================================================
# restoring the familiar constraint under unequal allocation
# =======================================================================================
def to_balanced_gauge(b: np.ndarray, arm_probs) -> np.ndarray:
    """Map the arm posterior to what it would have been under **equal** allocation.

        b  |-->  (b_k / pi_k) / sum_j (b_j / pi_j)

    Two facts make this the right way to handle ``pi = (0.1, 0.8, 0.1)``.

    **It is the gauge map of** ``CONSTRAINED.md`` with ``c_k = 1/pi_k``.  So by the gauge
    theorem the ROC surface and every VUS estimator are **bit-for-bit unchanged** by it:
    whatever a model's VUS was on the raw posterior, it is the same here.  Nothing about
    discrimination is gained or lost; only the coordinates change.

    **It carries the constraint region onto the standard ordered chamber.**  The set
    ``{b in simplex : b_1/pi_1 >= b_2/pi_2 >= b_3/pi_3}`` is cut out of the simplex by two
    homogeneous linear inequalities, so it is itself a **triangle** — with vertices
    ``e_1``, ``(pi_1, pi_2, 0)/(pi_1+pi_2)`` and ``pi`` — and this map sends it to the
    familiar ``{v_1 >= v_2 >= v_3}``, whose vertices are ``e_1``, ``(1,1,0)/2``,
    ``(1,1,1)/3``.  :func:`constraint_triangle_vertices` returns both and
    ``experiments/18_insurance_models.py`` checks the correspondence numerically.

    So there are two equivalent routes to the usual ``p_1 < p_2 < p_3`` machinery, and they
    are **not** the same thing in finite samples:

    ``balance="weights"`` (or ``"oversample"``)
        change the *training objective* so the fitted posterior is already in balanced
        coordinates.  A different model comes out, because a weighted log-loss is a
        different loss.
    ``balance="none"`` + this function
        leave the fit alone and change *coordinates afterwards*.  Same model, relabelled,
        and — by the gauge theorem — necessarily the same VUS.

    The first spends effective sample size (the 10% arms get weight 10); the second spends
    none.  Which wins is an empirical question and is measured, not assumed.
    """
    b = np.asarray(b, float)
    pi = np.asarray(arm_probs, float)
    pi = pi / pi.sum()
    v = b / pi[None, :]
    return v / v.sum(axis=1, keepdims=True)


def constraint_triangle_vertices(arm_probs) -> dict:
    """The constraint region is a triangle; return its vertices, and the standard one's.

    Under unequal allocation the set of *admissible* posteriors is not the usual ordered
    chamber but its image under the gauge — a triangle with a different shape and a
    different area.  Seeing the two side by side is the quickest way to understand why
    enforcing ``b_1 >= b_2 >= b_3`` at 0.1/0.8/0.1 is enforcing the wrong thing.
    """
    pi = np.asarray(arm_probs, float)
    pi = pi / pi.sum()
    K = len(pi)
    std = [np.eye(K)[0],
           np.array([1.0] * 2 + [0.0] * (K - 2)) / 2,
           np.ones(K) / K]
    mapped = []
    for v in std:
        w = pi * v
        mapped.append(w / w.sum())
    return {"standard_chamber": np.array(std), "admissible_region": np.array(mapped),
            "arm_probs": pi}


# =======================================================================================
# the model set
# =======================================================================================
class ConversionModel:
    """Common interface for every motor model.

    ``elasticity(df)`` is evaluated **at the control price by default**, not at the price
    the customer happened to be quoted.  That is not a detail: because ``eps = beta(1-s)``
    depends on the price, grouping on the quoted-price elasticity selects a different
    population into each arm and biases every grouped diagnostic (``docs/INSURANCE.md``
    §2.1 — the least-elastic bin came out at 0.31 against a true 0.53).
    """

    name = "base"

    def __init__(self, **params):
        self.params = params

    def fit(self, d):                                        # pragma: no cover
        raise NotImplementedError

    def conversion(self, d, log_price=None):                 # pragma: no cover
        raise NotImplementedError

    def elasticity(self, d, at="control"):                   # pragma: no cover
        raise NotImplementedError

    def _log_price(self, d, at):
        lp = d["log_technical_premium"].to_numpy(float)
        if at == "control":
            return lp
        if at == "quoted":
            return d["log_price"].to_numpy(float)
        return lp + float(np.log(at))

    def __repr__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


def _design(d, log_price):
    X = d[QUOTE_FEATURES].to_numpy(float)
    return np.column_stack([X, log_price])


class ConversionGLM(ConversionModel):
    """Logistic regression of conversion on log price and the rating features.

    ``s = sigmoid(a'x - beta log p)`` and therefore ``eps = beta (1 - s)`` in closed form —
    no finite differences, no step-size choice, and the elasticity is guaranteed positive
    and correctly shaped along the price ladder.  It pools all three arms, so it uses the
    80% control traffic as well as the 20% that carries the price variation, which is why
    the crudest model in the set is hard to beat when only 10% sits in each side arm.
    """

    name = "conversion_glm"

    def __init__(self, C: float = 1.0):
        super().__init__(C=C)

    def fit(self, d):
        from sklearn.linear_model import LogisticRegression
        lp = d["log_price"].to_numpy(float)
        X = _design(d, lp)
        self.mu_, self.sd_ = X.mean(0), np.where(X.std(0) > 1e-12, X.std(0), 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m_ = LogisticRegression(C=self.params["C"], max_iter=2000).fit(
                (X - self.mu_) / self.sd_, d["converted"].to_numpy(int))
        # de-standardise the price coefficient: beta is minus it
        self.beta_ = float(-self.m_.coef_[0, -1] / self.sd_[-1])
        return self

    def conversion(self, d, log_price=None):
        lp = d["log_price"].to_numpy(float) if log_price is None else np.asarray(log_price)
        X = (_design(d, lp) - self.mu_) / self.sd_
        return self.m_.predict_proba(X)[:, 1]

    def elasticity(self, d, at="control"):
        s = self.conversion(d, self._log_price(d, at))
        return self.beta_ * (1.0 - s)


class ConversionGBM(ConversionModel):
    """Gradient-boosted conversion (an S-learner), elasticity by finite difference.

    Flexible in the covariates and, unlike the GLM, free to let the price effect interact
    with them.  The cost is the one from ``MODELS.md`` §3.4: a boosted tree is piecewise
    constant in log price, so the finite-difference step ``delta`` is part of the estimator
    and a step smaller than the distance to a price split returns noise.  ``delta``
    defaults to the arm half-width, which is the only step the data actually supports.
    """

    name = "conversion_gbm"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_child_samples: int = 100,
                 reg_lambda: float = 1.0, delta: float = 0.10, seed: int = 0):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         reg_lambda=reg_lambda, delta=delta, seed=seed)

    def fit(self, d):
        import lightgbm as lgb
        p = {k: v for k, v in self.params.items() if k not in ("delta", "seed")}
        self.m_ = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=N_THREADS,
                                     random_state=self.params["seed"],
                                     force_col_wise=True, **p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m_.fit(_design(d, d["log_price"].to_numpy(float)),
                        d["converted"].to_numpy(int))
        return self

    def conversion(self, d, log_price=None):
        lp = d["log_price"].to_numpy(float) if log_price is None else np.asarray(log_price)
        return self.m_.predict_proba(_design(d, lp))[:, 1]

    def elasticity(self, d, at="control"):
        lp = self._log_price(d, at)
        h = self.params["delta"]
        up = self.conversion(d, lp + h)
        dn = self.conversion(d, lp - h)
        # -d log s / d log p, central difference on the log scale
        return -(np.log(np.maximum(up, 1e-12)) - np.log(np.maximum(dn, 1e-12))) / (2 * h)


class TLearnerConversion(ConversionModel):
    """One conversion model per arm — and a demonstration of what 0.1/0.8/0.1 costs.

    The T-learner is the textbook way to use a randomised experiment: fit ``s_k(x)``
    separately in each arm and difference.  It makes no functional-form assumption linking
    the arms, which is its appeal.  With this allocation it also throws away the 80%
    control traffic when estimating the side arms, so each of the two models that actually
    carry the price signal sees a tenth of the data.  Expect it to be the noisiest member
    of the set; that is the point of including it.
    """

    name = "tlearner_conversion"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 15, min_child_samples: int = 200, seed: int = 0):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         seed=seed)

    def fit(self, d):
        import lightgbm as lgb
        self.mult_ = np.asarray(d.attrs["arm_multipliers"], float)
        self.models_ = []
        X = d[QUOTE_FEATURES].to_numpy(float)
        arm = d["arm"].to_numpy()
        y = d["converted"].to_numpy(int)
        for k in range(len(self.mult_)):
            m = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=N_THREADS,
                                   random_state=self.params["seed"], force_col_wise=True,
                                   **{k2: v for k2, v in self.params.items()
                                      if k2 != "seed"})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[arm == k], y[arm == k])
            self.models_.append(m)
        return self

    def _curve(self, d):
        X = d[QUOTE_FEATURES].to_numpy(float)
        return np.column_stack([np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
                                for m in self.models_])

    def conversion(self, d, log_price=None):
        return self._curve(d)[np.arange(len(d)), d["arm"].to_numpy()]

    def elasticity(self, d, at="control"):
        s = self._curve(d)
        lm = np.log(self.mult_)
        # slope of log s on log multiplier, least squares across arms
        x = lm - lm.mean()
        ls = np.log(s)
        return -(ls * x[None, :]).sum(1) / np.sum(x ** 2)


class ArmPosteriorModel(ConversionModel):
    """The **price-test-first** model: predict the arm from a converter, read off demand.

    Fitted on converters only — a choice-based sample in the sense of Manski & Lerman
    (1977) — so the posterior is ``b_k(x) = pi_k s_k(x) / sum_j pi_j s_j(x)`` and the demand
    curve is recovered by dividing out the **known** allocation.  Because ``pi`` is known by
    design rather than estimated, this is the one setting where the construction carries no
    propensity-model risk at all.

    ``balance`` decides how the unequal allocation is handled, and the two routes are
    genuinely different (see :func:`to_balanced_gauge`):

    ``"weights"``     weight each converter by ``1/pi_k``, so the fitted posterior is
                      already the balanced one and the ordinary ``p_1 >= p_2 >= p_3``
                      constraint applies directly.  Changes the loss, so changes the fit.
    ``"oversample"``  the same idea by resampling rather than weighting.
    ``"none"``        fit raw, then move to balanced coordinates afterwards.  Same fit,
                      and by the gauge theorem necessarily the same VUS.

    ``constraint`` is applied in the balanced coordinates, where it is the familiar one.
    """

    name = "arm_posterior"

    def __init__(self, balance: str = "weights", constraint: str = "isotonic",
                 n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_child_samples: int = 100,
                 reg_lambda: float = 1.0, seed: int = 0):
        super().__init__(balance=balance, constraint=constraint,
                         n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         reg_lambda=reg_lambda, seed=seed)

    def fit(self, d):
        import lightgbm as lgb
        self.mult_ = np.asarray(d.attrs["arm_multipliers"], float)
        self.pi_ = np.asarray(d.attrs["arm_probs"], float)
        self.pi_ = self.pi_ / self.pi_.sum()
        K = len(self.mult_)

        conv = d[d["converted"] == 1]
        X = conv[QUOTE_FEATURES].to_numpy(float)
        arm = conv["arm"].to_numpy()
        w = None
        bal = self.params["balance"]
        if bal == "weights":
            w = 1.0 / self.pi_[arm]
        elif bal == "oversample":
            rng = np.random.default_rng(self.params["seed"])
            reps = 1.0 / self.pi_[arm]
            reps = reps / reps.min()
            idx = np.repeat(np.arange(len(conv)), np.floor(reps).astype(int))
            frac = reps - np.floor(reps)
            idx = np.concatenate([idx, np.flatnonzero(rng.random(len(conv)) < frac)])
            X, arm = X[idx], arm[idx]
        elif bal != "none":
            raise ValueError(f"unknown balance={bal!r}")
        self.balanced_fit_ = bal in ("weights", "oversample")
        self.ess_ = (float(w.sum() ** 2 / (w ** 2).sum()) / len(conv)) if w is not None \
            else 1.0
        self.n_converters_ = len(conv)

        p = {k: v for k, v in self.params.items()
             if k in ("n_estimators", "learning_rate", "num_leaves",
                      "min_child_samples", "reg_lambda")}
        self.m_ = lgb.LGBMClassifier(objective="multiclass", num_class=K, verbosity=-1,
                                     n_jobs=N_THREADS, random_state=self.params["seed"],
                                     force_col_wise=True, **p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m_.fit(X, arm, sample_weight=w)
        return self

    def posterior(self, d, balanced=False):
        """``b_k(x)``.  ``balanced=True`` returns it in the equal-allocation gauge."""
        b = np.clip(self.m_.predict_proba(d[QUOTE_FEATURES].to_numpy(float)), 1e-12, None)
        b = b / b.sum(1, keepdims=True)
        if balanced and not self.balanced_fit_:
            b = to_balanced_gauge(b, self.pi_)
        if not balanced and self.balanced_fit_:
            # undo: the fitted object is already balanced, so raw = gauge by pi
            w = b * self.pi_[None, :]
            b = w / w.sum(1, keepdims=True)
        return b

    def demand_curve(self, d):
        """``log s_k(x)`` up to an additive constant, in balanced coordinates."""
        v = self.posterior(d, balanced=True)
        ls = np.log(np.clip(v, 1e-12, None))
        if self.params["constraint"] == "isotonic":
            ls = isotonic_decreasing(ls)
        elif self.params["constraint"] not in ("none",):
            raise ValueError(f"unknown constraint={self.params['constraint']!r}")
        return ls - ls.mean(axis=1, keepdims=True)

    def conversion(self, d, log_price=None):
        """Not available, deliberately.

        The arm posterior identifies the **shape** of the demand curve — the ratios
        ``s_k/s_j`` — and says nothing about its level, because the level cancels out of
        Lemma 1.  Returning ``exp(log_s)`` anyway would look like a conversion probability
        and score a log-loss of 8-9 against a real one of 0.52.  Raising is the honest
        response; pair this model with any conversion model if a level is needed.
        """
        raise NotImplementedError(
            "ArmPosteriorModel estimates the demand curve up to a constant and has no "
            "conversion level; use .demand_curve() or .elasticity(), or pair it with a "
            "ConversionGLM for the level.")

    def elasticity(self, d, at="control"):
        """Slope of the log demand curve against the log arm multiplier.

        Note this is an **arc** elasticity across the tested prices, so it does not depend
        on ``at`` — the construction never sees a price other than the three that were run.
        That is a real limitation next to the GLM, which extrapolates by assumption.
        """
        ls = self.demand_curve(d)
        x = np.log(self.mult_) - np.log(self.mult_).mean()
        return -(ls * x[None, :]).sum(1) / np.sum(x ** 2)


class GLMThenGBM(ConversionModel):
    """Two-stage: a logistic GLM, then a GBM on what it leaves behind.

    The actuarial **offset** construction (Yan et al. 2009; Wüthrich & Merz's CANN): stage
    one fits an interpretable GLM, stage two boosts on the residual with the GLM's linear
    predictor as ``init_score``.  Fitting on the residual and offsetting are the same thing.

    ``stage2_sees_price`` is the interesting switch.  With it **off**, the price effect is
    entirely the GLM's single coefficient — the predictions improve, the elasticity stays a
    clean interpretable number, and the two concerns are decoupled.  With it **on**, the
    GBM can correct the price effect and also re-imports the flattening problem onto the
    correction.
    """

    name = "glm_then_gbm"

    def __init__(self, stage2_sees_price: bool = False, C: float = 1.0,
                 n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_child_samples: int = 100,
                 delta: float = 0.10, seed: int = 0):
        super().__init__(stage2_sees_price=stage2_sees_price, C=C,
                         n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         delta=delta, seed=seed)

    def _stage2_X(self, d, lp):
        return _design(d, lp) if self.params["stage2_sees_price"] \
            else d[QUOTE_FEATURES].to_numpy(float)

    def fit(self, d):
        import lightgbm as lgb
        self.glm_ = ConversionGLM(C=self.params["C"]).fit(d)
        lp = d["log_price"].to_numpy(float)
        s1 = np.clip(self.glm_.conversion(d, lp), 1e-6, 1 - 1e-6)
        init = np.log(s1 / (1 - s1))                    # the GLM's linear predictor
        p = {k: v for k, v in self.params.items()
             if k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")}
        self.gbm_ = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=N_THREADS,
                                       random_state=self.params["seed"],
                                       force_col_wise=True, **p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gbm_.fit(self._stage2_X(d, lp), d["converted"].to_numpy(int),
                          init_score=init)
        return self

    def conversion(self, d, log_price=None):
        lp = d["log_price"].to_numpy(float) if log_price is None else np.asarray(log_price)
        s1 = np.clip(self.glm_.conversion(d, lp), 1e-6, 1 - 1e-6)
        init = np.log(s1 / (1 - s1))
        raw = self.gbm_.predict(self._stage2_X(d, lp), raw_score=True)
        return 1.0 / (1.0 + np.exp(-(init + raw)))

    def elasticity(self, d, at="control"):
        lp = self._log_price(d, at)
        if not self.params["stage2_sees_price"]:
            # stage 2 cannot touch the price derivative, so the elasticity is exactly the
            # GLM's -- beta*(1-s) at the TWO-STAGE conversion estimate, which is better
            # calibrated than the GLM's own
            return self.glm_.beta_ * (1.0 - self.conversion(d, lp))
        h = self.params["delta"]
        up = self.conversion(d, lp + h)
        dn = self.conversion(d, lp - h)
        return -(np.log(np.maximum(up, 1e-12)) - np.log(np.maximum(dn, 1e-12))) / (2 * h)


MOTOR_MODELS = {m.name: m for m in (ConversionGLM, ConversionGBM, TLearnerConversion,
                                    ArmPosteriorModel, GLMThenGBM)}


# =======================================================================================
# the profit-weighted objective
# =======================================================================================
def lerner_price(eps: np.ndarray, cost: np.ndarray, *, price0=None,
                 lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    """``p* = c * eps / (eps - 1)`` — the Lerner price for a *given* elasticity.

    Treats ``eps`` as fixed rather than re-solving the implicit equation, which is what a
    practitioner does when the model reports one elasticity per customer.  Where
    ``eps <= 1`` there is no interior optimum and the price is capped at ``hi * price0``.
    """
    eps = np.asarray(eps, float)
    c = np.asarray(cost, float)
    p0 = c if price0 is None else np.asarray(price0, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(eps > 1.0 + 1e-9, c * eps / (eps - 1.0), hi * p0)
    return np.clip(np.nan_to_num(p, nan=hi * p0), lo * p0, hi * p0)


def profit_weight(eps: np.ndarray) -> np.ndarray:
    """How much a unit of elasticity error costs, up to a common factor: ``(eps-1)^-4``.

    The optimal price moves with the elasticity at rate

        dp*/d eps  =  -c / (eps - 1)^2

    and profit is locally quadratic around its optimum, so the regret from an error
    ``d eps`` scales as ``(d eps)^2 / (eps - 1)^4``.  The exponent is not a typo, and the
    consequence is severe: an error at ``eps = 1.5`` costs **256 times** what the same error
    costs at ``eps = 3``.

    This is why squared error in the elasticity is the wrong loss for a pricing model.  It
    spends its effort where the elasticity is large and the answer barely matters, and
    ignores the near-unit-elasticity customers where the price recommendation swings
    wildly.  Use :func:`profit_regret` to grade, and the IPW policy value to tune.
    """
    eps = np.asarray(eps, float)
    return 1.0 / np.maximum(np.abs(eps - 1.0), 1e-3) ** 4


def profit_regret(d: pd.DataFrame, eps_hat: np.ndarray, *, price0=None,
                  reduce: str = "mean") -> dict:
    """Oracle regret: the profit given up by pricing on ``eps_hat`` instead of the truth.

        regret_i  =  pi_i(p*(eps_i))  -  pi_i(p*(eps_hat_i))

    with the true conversion curve for ``pi_i``, so this is exact rather than estimated.
    It is the **grader** — available only in simulation — and the thing every observable
    objective is trying to approximate.

    Unlike squared error it is asymmetric and scale-aware: it is near zero wherever the
    price recommendation is insensitive to the elasticity, and large wherever it is not.
    """
    a = d["_a"].to_numpy(float)
    beta = d["true_beta"].to_numpy(float)
    c = d["claims_cost"].to_numpy(float)
    p0 = np.exp(d["log_technical_premium"].to_numpy(float)) if price0 is None \
        else np.asarray(price0, float)

    link = d["_link"].iloc[0] if "_link" in d else "logit"

    def true_profit(p):
        s = _apply_link(a - beta * np.log(p), link, d)
        return s * (p - c)

    eps_true = d["true_elasticity_control"].to_numpy(float)
    p_opt = lerner_price(eps_true, c, price0=p0)
    p_hat = lerner_price(np.asarray(eps_hat, float), c, price0=p0)
    reg = true_profit(p_opt) - true_profit(p_hat)
    out = {"regret_mean": float(reg.mean()), "regret_median": float(np.median(reg)),
           "regret_p90": float(np.percentile(reg, 90)),
           "profit_at_oracle": float(true_profit(p_opt).mean()),
           "profit_at_model": float(true_profit(p_hat).mean()),
           "pct_of_oracle_profit": float(true_profit(p_hat).mean()
                                         / max(true_profit(p_opt).mean(), 1e-9))}
    if reduce == "all":
        out["regret"] = reg
    return out


def experiment_cost(d: pd.DataFrame, *, reference: str = "control") -> dict:
    """What the price test itself costs, per quote.

    Widening the arms buys statistical power and is not free: the cheap arm is deliberately
    underpriced and the dear arm deliberately overpriced, and the traffic sent there earns
    less than it would at the reference price.  A test design is a trade between the
    information gained and this, and quoting the first without the second is not a
    business case.

    ``reference`` is what you would otherwise have charged everyone — ``"control"``, or
    ``"best_flat"`` for the best of the three arms as measured.
    """
    arm = d["arm"].to_numpy()
    pi = np.asarray(d.attrs["arm_probs"], float)
    mult = np.asarray(d.attrs["arm_multipliers"], float)
    price = np.exp(d["log_technical_premium"].to_numpy(float))
    profit = d["converted"].to_numpy() * (price * mult[arm] - d["claims_cost"].to_numpy())
    by_arm = np.array([profit[arm == k].mean() for k in range(len(mult))])
    ref = by_arm[len(mult) // 2] if reference == "control" else by_arm.max()
    realised = float((pi * by_arm).sum())
    return {"profit_by_arm": by_arm, "reference_profit": float(ref),
            "realised_profit": realised,
            "cost_per_quote": float(ref - realised),
            "cost_pct": float((ref - realised) / max(abs(ref), 1e-9))}


# =======================================================================================
# Phase A: the treatment feature, the logit-space derivative, competitive position
# =======================================================================================
#: Features plus the two engineered terms of ``PLAN_INSURANCE_V2`` I1 and I4.
#:
#: ``arm_log_multiplier`` is **the treatment, isolated**.  Feeding a tree ``log_price``
#: instead is the mistake that cost the flexible models the benchmark: ``log_price =
#: log_technical_premium + arm_log_multiplier`` and the first term is *also* a feature, so
#: a split on ``log_price`` mixes "expensive customer" (a factor of ten across the book)
#: with "dear arm" (+/-10%).  A linear model separates them exactly; a tree cannot.
#:
#: ``price_vs_market`` is ``log_price - log_competitor_index`` — competitive position.  The
#: DGP drives conversion through exactly this difference, and in real motor pricing it is
#: the strongest conversion feature there is.  Both components were already available and
#: neither model was given the difference, which a tree needs many splits to approximate.
ENGINEERED = ["arm_log_multiplier", "price_vs_market"]


def add_engineered(d: pd.DataFrame, log_price=None) -> pd.DataFrame:
    """Attach the engineered treatment features, optionally at a counterfactual price."""
    out = d.copy()
    lp = out["log_price"].to_numpy(float) if log_price is None else np.asarray(log_price)
    out["arm_log_multiplier"] = lp - out["log_technical_premium"].to_numpy(float)
    out["price_vs_market"] = lp - out["log_competitor_index"].to_numpy(float)
    return out


def _design_v2(d, log_price, cols):
    lp = np.asarray(log_price, float)
    tech = d["log_technical_premium"].to_numpy(float)
    comp = d["log_competitor_index"].to_numpy(float)
    parts = []
    for c in cols:
        if c == "arm_log_multiplier":
            parts.append(lp - tech)
        elif c == "price_vs_market":
            parts.append(lp - comp)
        elif c == "log_price":
            parts.append(lp)
        else:
            parts.append(d[c].to_numpy(float))
    return np.column_stack(parts)


class ConversionGBMv2(ConversionModel):
    """The boosted conversion model, with I1–I4 applied.

    Four changes from :class:`ConversionGBM`, each addressing a diagnosed defect rather
    than a hyperparameter:

    **I1** the treatment enters as ``arm_log_multiplier``, isolated from customer identity.
    **I4** ``price_vs_market`` gives the model competitive position as one feature.
    **I2** ``monotone`` puts a non-increasing constraint on both price columns, so
    ``eps >= 0`` holds by construction.  Published cost of monotonicity in GBM credit
    models is 0–2.9% of accuracy; here it is measured rather than assumed.
    **I3** the derivative is taken in **logit space**:
    ``eps = -(d eta / d log p) * (1 - s)``.  The model is additive in ``eta``, so this is
    where a finite difference is stable, and it makes the bounded-demand structure of
    ``INSURANCE.md`` §1.2 explicit instead of hoping the probability scale reproduces it.
    """

    name = "conversion_gbm_v2"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_child_samples: int = 100,
                 reg_lambda: float = 1.0, subsample: float = 1.0,
                 colsample_bytree: float = 1.0, delta: float = 0.10,
                 monotone: bool = True, use_price_level: bool = False, seed: int = 0):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         reg_lambda=reg_lambda, subsample=subsample,
                         colsample_bytree=colsample_bytree, delta=delta,
                         monotone=monotone, use_price_level=use_price_level, seed=seed)

    def _cols(self):
        cols = list(QUOTE_FEATURES) + list(ENGINEERED)
        if self.params["use_price_level"]:
            cols = cols + ["log_price"]
        return cols

    def fit(self, d):
        import lightgbm as lgb
        self.cols_ = self._cols()
        p = {k: v for k, v in self.params.items()
             if k in ("n_estimators", "learning_rate", "num_leaves",
                      "min_child_samples", "reg_lambda", "colsample_bytree")}
        if self.params["subsample"] < 1.0:
            p["subsample"] = self.params["subsample"]
            p["subsample_freq"] = 1
        if self.params["monotone"]:
            # -1 = non-increasing.  Conversion cannot rise with the price charged, nor
            # with the price relative to the market.
            p["monotone_constraints"] = [
                -1 if c in ("arm_log_multiplier", "price_vs_market", "log_price") else 0
                for c in self.cols_]
        self.m_ = lgb.LGBMClassifier(objective="binary", verbosity=-1, n_jobs=N_THREADS,
                                     random_state=self.params["seed"],
                                     force_col_wise=True, **p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m_.fit(_design_v2(d, d["log_price"].to_numpy(float), self.cols_),
                        d["converted"].to_numpy(int))
        return self

    def _eta(self, d, log_price):
        return self.m_.predict(_design_v2(d, log_price, self.cols_), raw_score=True)

    def conversion(self, d, log_price=None):
        lp = d["log_price"].to_numpy(float) if log_price is None else np.asarray(log_price)
        return 1.0 / (1.0 + np.exp(-self._eta(d, lp)))

    def elasticity(self, d, at="control"):
        lp = self._log_price(d, at)
        h = self.params["delta"]
        # I3: differentiate the LOGIT, then convert.  eps = -(d eta / d log p)(1 - s).
        deta = (self._eta(d, lp + h) - self._eta(d, lp - h)) / (2 * h)
        s = 1.0 / (1.0 + np.exp(-self._eta(d, lp)))
        return -deta * (1.0 - s)


class ConversionGLMv2(ConversionModel):
    """The **fair** parametric baseline: a logit with price interactions and splines (I11).

    "A GBM beats a GLM" is only interesting if the GLM was allowed to express what the GBM
    discovers.  The actuarial literature's recommended specification is a logit in log price
    with interactions against the main rating factors and splines on the continuous ones —
    not a plain main-effects model.  This is that.

    ``interact_with`` names the features whose interaction with the treatment is fitted, so
    the elasticity varies with them:

        eta = a'x + spline(x)  -  (beta_0 + sum_j beta_j z_j) * log p

    and therefore ``eps = (beta_0 + sum_j beta_j z_j)(1 - s)`` — still closed form, still
    guaranteed non-negative when the coefficients are, but now heterogeneous in a way the
    main-effects GLM could not represent.
    """

    name = "conversion_glm_v2"

    def __init__(self, C: float = 1.0, n_knots: int = 5,
                 interact_with=("shopping_intensity", "channel_aggregator",
                                "is_renewal", "log_competitor_index", "driver_age")):
        super().__init__(C=C, n_knots=n_knots, interact_with=tuple(interact_with))

    def _blocks(self, d, lp):
        X = d[QUOTE_FEATURES].to_numpy(float)
        Z = np.column_stack([d[c].to_numpy(float) for c in self.params["interact_with"]])
        Zs = (Z - self.zmu_) / self.zsd_
        spl = self.spline_.transform(d[self.spline_cols_].to_numpy(float))
        # treatment block: log p, and log p interacted with each standardised z
        t = np.column_stack([lp] + [lp * Zs[:, j] for j in range(Zs.shape[1])])
        return np.column_stack([X, spl, t]), t.shape[1]

    def fit(self, d):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import SplineTransformer

        self.spline_cols_ = ["driver_age", "log_technical_premium",
                             "log_competitor_index", "shopping_intensity"]
        self.spline_ = SplineTransformer(n_knots=self.params["n_knots"], degree=3,
                                         include_bias=False)
        self.spline_.fit(d[self.spline_cols_].to_numpy(float))
        Z = np.column_stack([d[c].to_numpy(float)
                             for c in self.params["interact_with"]])
        self.zmu_, self.zsd_ = Z.mean(0), np.where(Z.std(0) > 1e-12, Z.std(0), 1.0)

        lp = d["log_price"].to_numpy(float)
        A, n_t = self._blocks(d, lp)
        self.n_t_ = n_t
        self.mu_, self.sd_ = A.mean(0), np.where(A.std(0) > 1e-12, A.std(0), 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m_ = LogisticRegression(C=self.params["C"], max_iter=3000).fit(
                (A - self.mu_) / self.sd_, d["converted"].to_numpy(int))
        self.coef_ = self.m_.coef_[0] / self.sd_
        return self

    def _beta(self, d):
        """The customer-specific price coefficient, ``beta_0 + sum_j beta_j z_j``."""
        Z = np.column_stack([d[c].to_numpy(float)
                             for c in self.params["interact_with"]])
        Zs = (Z - self.zmu_) / self.zsd_
        c = self.coef_[-self.n_t_:]
        return -(c[0] + Zs @ c[1:])

    def conversion(self, d, log_price=None):
        lp = d["log_price"].to_numpy(float) if log_price is None else np.asarray(log_price)
        A, _ = self._blocks(d, lp)
        return self.m_.predict_proba((A - self.mu_) / self.sd_)[:, 1]

    def elasticity(self, d, at="control"):
        lp = self._log_price(d, at)
        return self._beta(d) * (1.0 - self.conversion(d, lp))


class AIPWArmEffects(ConversionModel):
    """Doubly-robust per-arm conversion, then the elasticity from the fitted curve (I6).

    The propensity is **known exactly** — this is a randomised test — so the augmented
    estimator

        mu_k(x)  =  m_k(x)  +  1[A = k] / pi_k  *  ( Y - m_k(x) )

    is unbiased for ``s_k(x)`` whatever the outcome model ``m_k`` does, and has strictly
    lower variance than plain inverse-propensity weighting because the ``m_k`` term absorbs
    the predictable part of ``Y``.  With 0.1/0.8/0.1 that variance reduction is the whole
    game: the naive IPW estimate of a side arm rests on a tenth of the traffic.

    Implemented in two stages: cross-fitted outcome models ``m_k`` (so a quote's own
    residual never enters its own fitted value), then a **second-stage regression of the
    AIPW pseudo-outcome on the covariates**, which turns the unbiased-but-noisy pointwise
    scores into a smooth per-arm curve.

    Doubly robust and, here, robust for free: the propensity leg cannot be wrong.
    """

    name = "aipw_arm_effects"

    def __init__(self, n_folds: int = 4, n_estimators: int = 300,
                 learning_rate: float = 0.05, num_leaves: int = 31,
                 min_child_samples: int = 100, final_num_leaves: int = 15,
                 final_n_estimators: int = 200, monotone: bool = True, seed: int = 0):
        super().__init__(n_folds=n_folds, n_estimators=n_estimators,
                         learning_rate=learning_rate, num_leaves=num_leaves,
                         min_child_samples=min_child_samples,
                         final_num_leaves=final_num_leaves,
                         final_n_estimators=final_n_estimators,
                         monotone=monotone, seed=seed)

    def fit(self, d):
        import lightgbm as lgb
        self.mult_ = np.asarray(d.attrs["arm_multipliers"], float)
        pi = np.asarray(d.attrs["arm_probs"], float)
        self.pi_ = pi / pi.sum()
        K = len(self.mult_)
        X = d[QUOTE_FEATURES].to_numpy(float)
        y = d["converted"].to_numpy(float)
        arm = d["arm"].to_numpy()
        n = len(d)

        rng = np.random.default_rng(self.params["seed"])
        fold = rng.integers(0, self.params["n_folds"], n)
        m_hat = np.zeros((n, K))
        base = {k: v for k, v in self.params.items()
                if k in ("n_estimators", "learning_rate", "num_leaves",
                         "min_child_samples")}
        for f in range(self.params["n_folds"]):
            tr, te = fold != f, fold == f
            for k in range(K):
                sel = tr & (arm == k)
                if sel.sum() < 50:
                    m_hat[te, k] = y[tr].mean()
                    continue
                mk = lgb.LGBMClassifier(objective="binary", verbosity=-1,
                                        n_jobs=N_THREADS, force_col_wise=True,
                                        random_state=self.params["seed"], **base)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mk.fit(X[sel], y[sel].astype(int))
                m_hat[te, k] = mk.predict_proba(X[te])[:, 1]

        # the AIPW pseudo-outcome, one column per arm
        psi = m_hat.copy()
        for k in range(K):
            hit = arm == k
            psi[hit, k] += (y[hit] - m_hat[hit, k]) / self.pi_[k]

        # second stage: smooth each arm's pseudo-outcome back onto the covariates
        self.final_ = []
        fp = dict(n_estimators=self.params["final_n_estimators"],
                  learning_rate=self.params["learning_rate"],
                  num_leaves=self.params["final_num_leaves"],
                  min_child_samples=self.params["min_child_samples"])
        for k in range(K):
            g = lgb.LGBMRegressor(objective="regression", verbosity=-1, n_jobs=N_THREADS,
                                  force_col_wise=True, random_state=self.params["seed"],
                                  **fp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.fit(X, psi[:, k])
            self.final_.append(g)
        self.ipw_only_ = psi           # kept so the variance gain can be measured
        return self

    def _curve(self, d):
        X = d[QUOTE_FEATURES].to_numpy(float)
        s = np.column_stack([g.predict(X) for g in self.final_])
        s = np.clip(s, 1e-4, 1 - 1e-4)
        if self.params["monotone"]:
            s = np.exp(isotonic_decreasing(np.log(s)))
        return s

    def conversion(self, d, log_price=None):
        return self._curve(d)[np.arange(len(d)), d["arm"].to_numpy()]

    def elasticity(self, d, at="control"):
        s = self._curve(d)
        x = np.log(self.mult_) - np.log(self.mult_).mean()
        return -(np.log(s) * x[None, :]).sum(1) / np.sum(x ** 2)


class AnchoredTLearner(ConversionModel):
    """A T-learner that does not throw away the control arm (I7).

    The plain T-learner fits three independent models, so each side arm is estimated from a
    tenth of the traffic.  This one fits the **control** arm on its 80% — where the data
    is — and then models only the *increment* ``eta_k - eta_control`` for the side arms,
    with the control fit supplied as an ``init_score`` offset.  The side-arm models
    therefore start from a good answer and only have to learn a small correction, which is
    the same construction as :class:`GLMThenGBM` applied across arms rather than across
    model classes.
    """

    name = "anchored_tlearner"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_child_samples: int = 100,
                 side_n_estimators: int = 150, side_num_leaves: int = 8,
                 side_learning_rate: float = 0.03, monotone: bool = True, seed: int = 0):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         side_n_estimators=side_n_estimators,
                         side_num_leaves=side_num_leaves,
                         side_learning_rate=side_learning_rate,
                         monotone=monotone, seed=seed)

    def fit(self, d):
        import lightgbm as lgb
        self.mult_ = np.asarray(d.attrs["arm_multipliers"], float)
        K = len(self.mult_)
        self.control_ = K // 2
        X = d[QUOTE_FEATURES].to_numpy(float)
        y = d["converted"].to_numpy(int)
        arm = d["arm"].to_numpy()

        base = lgb.LGBMClassifier(
            objective="binary", verbosity=-1, n_jobs=N_THREADS, force_col_wise=True,
            random_state=self.params["seed"],
            **{k: self.params[k] for k in ("n_estimators", "learning_rate",
                                           "num_leaves", "min_child_samples")})
        sel = arm == self.control_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base.fit(X[sel], y[sel])
        self.base_ = base

        self.side_ = {}
        for k in range(K):
            if k == self.control_:
                continue
            s = arm == k
            init = base.predict(X[s], raw_score=True)
            g = lgb.LGBMClassifier(
                objective="binary", verbosity=-1, n_jobs=N_THREADS, force_col_wise=True,
                random_state=self.params["seed"],
                n_estimators=self.params["side_n_estimators"],
                num_leaves=self.params["side_num_leaves"],
                learning_rate=self.params["side_learning_rate"],
                min_child_samples=self.params["min_child_samples"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.fit(X[s], y[s], init_score=init)
            self.side_[k] = g
        return self

    def _curve(self, d):
        X = d[QUOTE_FEATURES].to_numpy(float)
        eta0 = self.base_.predict(X, raw_score=True)
        etas = []
        for k in range(len(self.mult_)):
            e = eta0 if k == self.control_ else eta0 + self.side_[k].predict(
                X, raw_score=True)
            etas.append(e)
        s = 1.0 / (1.0 + np.exp(-np.column_stack(etas)))
        if self.params["monotone"]:
            s = np.exp(isotonic_decreasing(np.log(np.clip(s, 1e-6, 1 - 1e-6))))
        return s

    def conversion(self, d, log_price=None):
        return self._curve(d)[np.arange(len(d)), d["arm"].to_numpy()]

    def elasticity(self, d, at="control"):
        s = self._curve(d)
        x = np.log(self.mult_) - np.log(self.mult_).mean()
        return -(np.log(s) * x[None, :]).sum(1) / np.sum(x ** 2)


MOTOR_MODELS_V2 = {m.name: m for m in (ConversionGLMv2, ConversionGBMv2,
                                       AIPWArmEffects, AnchoredTLearner)}
