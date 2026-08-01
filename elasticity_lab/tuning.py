"""
elasticity_lab.tuning
=====================

Hyperparameter optimisation for elasticity models — and the question that makes it
different from ordinary HPO.

The problem with tuning a causal model
--------------------------------------
Ordinary HPO minimises held-out predictive loss.  For an elasticity model that is the
wrong objective, and not by a little.  The outcome ``log q`` is dominated by things that
have nothing to do with price: the product's baseline volume, the season, last week's
sales.  A search that maximises predictive accuracy will happily spend all its capacity
on those, flatten the price dimension to reduce variance, and return a model with a
superb RMSE and a badly attenuated elasticity.  Nothing in the predictive objective ever
asks the model to get ``d log q / d log p`` right.

So this module implements **three** objectives and treats "which one should you tune on?"
as the experiment rather than the assumption:

=================  =================================================================
``"rmse"``         Held-out RMSE of ``log q``.  What everyone actually does.
``"rloss"``        Nie & Wager's orthogonal **R-loss**, computable from observed data
                   alone.  The causal model-selection criterion.
``"oracle"``       RMSE against the true elasticity.  Only exists on the synthetic
                   panel; it is the *grader*, never a legitimate objective on real data.
=================  =================================================================

The R-loss, concretely
----------------------
Write the structural model as ``log q = theta(x)·log p + g(x) + noise`` with
``theta = −elasticity``.  Residualise both sides on the controls using nuisance models
fitted on the **training** part of the fold::

    ry = log q − Ê[log q | x]          rt = log p − Ê[log p | x]

Then for any candidate model, whatever its internals,

.. math::  R(\\text{model}) \\;=\\; \\frac{1}{n}\\sum_i \\bigl(ry_i - \\theta(x_i)\\, rt_i\\bigr)^2

This is the key to the whole module: it is **model-agnostic**.  It scores a two-way
fixed-effects regression and a gradient-boosted S-learner on the same axis, using only
data you have.  Its population minimiser is the true ``theta(·)``, so lower really is
better — unlike predictive RMSE, whose minimiser has nothing to do with the elasticity.

**The nuisances are fitted once per fold and shared by every candidate.**  If each
candidate got its own residuals, a candidate could win by being paired with a lucky
nuisance fit, and the comparison would be meaningless.

Reading an R-loss
-----------------
Raw R-loss values are on the scale of ``var(ry)`` and are not interpretable alone, so
:func:`cv_score` also reports two normalisations:

``rloss_skill``       ``1 − R(model) / mean(ry²)`` — improvement over "price does nothing".
``rloss_vs_constant`` ``1 − R(model) / R(best constant θ)`` — improvement over the best
                      single elasticity for everyone.  **Positive here is the only
                      honest evidence that estimated heterogeneity is real**, because
                      the best constant is fitted on the same validation fold.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .features import CONTROL_FEATURES, OUTCOME, TREATMENT, add_features
from .models import MODEL_REGISTRY, N_THREADS, build_model
from .splits import RollingOriginSplit

__all__ = ["CausalScorer", "cv_score", "SEARCH_SPACES", "suggest_params", "tune",
           "objective_comparison", "nested_cv", "tuning_history"]


# =======================================================================================
# fold-wise nuisance residuals, shared across candidates
# =======================================================================================
@dataclass
class CausalScorer:
    """Pre-computes ``(ry, rt)`` on every validation fold, once.

    The nuisance learner is deliberately *fixed* and moderately regularised: it is
    infrastructure for scoring, not a thing being tuned.  Tuning it inside the objective
    would let a candidate be rewarded for a nuisance fit that happens to flatter it.
    """

    df: pd.DataFrame
    splitter: Any = field(default_factory=RollingOriginSplit)
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 40
    seed: int = 0

    def __post_init__(self):
        self.folds_: list[tuple[np.ndarray, np.ndarray]] = list(self.splitter.split(self.df))
        self.cache_: dict[int, dict[str, np.ndarray]] = {}

    def _model(self):
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="regression", verbosity=-1,
                                 n_jobs=N_THREADS,
                                 random_state=self.seed, force_col_wise=True,
                                 n_estimators=self.n_estimators,
                                 learning_rate=self.learning_rate,
                                 num_leaves=self.num_leaves,
                                 min_child_samples=self.min_child_samples)

    def residuals(self, k: int) -> dict[str, np.ndarray]:
        """``ry``/``rt`` on fold ``k``'s validation block, computed once and cached."""
        if k in self.cache_:
            return self.cache_[k]
        tr, va = self.folds_[k]
        Xtr = self.df.iloc[tr][CONTROL_FEATURES].to_numpy(float)
        Xva = self.df.iloc[va][CONTROL_FEATURES].to_numpy(float)
        ytr = self.df.iloc[tr][OUTCOME].to_numpy(float)
        ttr = self.df.iloc[tr][TREATMENT].to_numpy(float)
        yva = self.df.iloc[va][OUTCOME].to_numpy(float)
        tva = self.df.iloc[va][TREATMENT].to_numpy(float)
        my = self._model().fit(Xtr, ytr)
        mt = self._model().fit(Xtr, ttr)
        ry = yva - my.predict(Xva)
        rt = tva - mt.predict(Xva)
        # the best single elasticity on this fold — the reference every candidate must beat
        theta_c = float(np.dot(rt, ry) / max(np.dot(rt, rt), 1e-12))
        out = {"ry": ry, "rt": rt,
               "rloss_zero": float(np.mean(ry ** 2)),
               "rloss_const": float(np.mean((ry - theta_c * rt) ** 2)),
               "theta_const": theta_c}
        self.cache_[k] = out
        return out

    @staticmethod
    def r_loss(res: dict, elasticity: np.ndarray) -> float:
        """``mean((ry + eps·rt)²)`` — note ``theta = −elasticity``."""
        return float(np.mean((res["ry"] + np.asarray(elasticity, float) * res["rt"]) ** 2))


# =======================================================================================
# one cross-validated evaluation of one hyperparameter setting
# =======================================================================================
def cv_score(model_name: str, params: dict, df: pd.DataFrame, scorer: CausalScorer, *,
             truth_col: str | None = None, trial=None,
             objective: str = "rloss") -> dict[str, float]:
    """Fit ``model_name(**params)`` on each fold and return every metric at once.

    Returning all metrics from one pass is what makes the objective comparison cheap:
    a search that *optimises* RMSE still records the R-loss and (on synthetic data) the
    oracle error of every trial it visits, so the whole search history can be re-examined
    afterwards under a different objective without refitting anything.

    ``trial`` (an Optuna trial) enables pruning: the running fold mean is reported after
    each fold and a hopeless configuration is abandoned early.
    """
    import optuna

    per_fold = []
    for k, (tr, va) in enumerate(scorer.folds_):
        dtr, dva = df.iloc[tr], df.iloc[va]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = build_model(model_name, **params).fit(dtr)
                eps = np.asarray(m.elasticity(dva), float)
                yhat = np.asarray(m.predict_log_qty(dva), float)
        except Exception:
            return {"rmse": np.inf, "rloss": np.inf, "oracle": np.inf,
                    "rloss_skill": -np.inf, "rloss_vs_constant": -np.inf,
                    "avg_eps": np.nan, "eps_sd": np.nan, "failed": 1.0}
        if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(yhat)):
            return {"rmse": np.inf, "rloss": np.inf, "oracle": np.inf,
                    "rloss_skill": -np.inf, "rloss_vs_constant": -np.inf,
                    "avg_eps": np.nan, "eps_sd": np.nan, "failed": 1.0}

        res = scorer.residuals(k)
        rl = CausalScorer.r_loss(res, eps)
        row = {
            "rmse": float(np.sqrt(np.mean((yhat - dva[OUTCOME].to_numpy(float)) ** 2))),
            "rloss": rl,
            "rloss_skill": 1.0 - rl / max(res["rloss_zero"], 1e-12),
            "rloss_vs_constant": 1.0 - rl / max(res["rloss_const"], 1e-12),
            "avg_eps": float(np.mean(eps)),
            "eps_sd": float(np.std(eps)),
        }
        if truth_col is not None:
            t = dva[truth_col].to_numpy(float)
            row["oracle"] = float(np.sqrt(np.mean((eps - t) ** 2)))
            row["oracle_bias"] = float(np.mean(eps) - np.mean(t))
        per_fold.append(row)

        if trial is not None:
            running = float(np.mean([r[objective] for r in per_fold]))
            trial.report(running if objective != "rloss_vs_constant" else -running, k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    out = {k: float(np.mean([r[k] for r in per_fold])) for k in per_fold[0]}
    out["failed"] = 0.0
    out["n_folds"] = float(len(per_fold))
    return out


# =======================================================================================
# search spaces
# =======================================================================================
def _sp_pooled(t):
    return {"use_controls": t.suggest_categorical("use_controls", [True, False]),
            "ridge": t.suggest_float("ridge", 1e-6, 10.0, log=True)}


def _sp_fe(t):
    return {"add_controls": t.suggest_categorical("add_controls", [True, False]),
            "n_iter": t.suggest_int("n_iter", 2, 20),
            "collinear_tol": t.suggest_float("collinear_tol", 1e-6, 0.2, log=True)}


def _sp_poisson(t):
    return {"use_controls": t.suggest_categorical("use_controls", [True, False]),
            "alpha": t.suggest_float("alpha", 1e-8, 1.0, log=True),
            "max_iter": t.suggest_int("max_iter", 100, 500, step=100)}


def _sp_lgbm(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 255, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 400, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
            "delta": t.suggest_float("delta", 0.01, 0.25, log=True)}


def _sp_dml(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 900, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 255, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 400, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
            "n_folds": t.suggest_int("n_folds", 2, 5)}


def _sp_rlearner(t):
    p = _sp_dml(t)
    p.update(final_n_estimators=t.suggest_int("final_n_estimators", 50, 600, step=50),
             final_learning_rate=t.suggest_float("final_learning_rate", 0.01, 0.3, log=True),
             final_num_leaves=t.suggest_int("final_num_leaves", 4, 127, log=True),
             final_min_child_samples=t.suggest_int("final_min_child_samples", 10, 500,
                                                   log=True),
             min_weight_quantile=t.suggest_float("min_weight_quantile", 0.0, 0.5))
    return p


def _sp_shrunk(t):
    # tau2=None means "estimate the shrinkage strength by method of moments"; the search
    # is allowed to override it with a fixed value, which is a genuinely different model.
    fixed = t.suggest_categorical("tau2_mode", ["auto", "fixed"]) == "fixed"
    tau2 = t.suggest_float("tau2", 1e-4, 5.0, log=True) if fixed else None
    return {"min_obs": t.suggest_int("min_obs", 6, 40),
            "min_price_sd": t.suggest_float("min_price_sd", 1e-4, 0.2, log=True),
            "tau2": tau2,
            "demean_week": t.suggest_categorical("demean_week", [True, False])}


SEARCH_SPACES: dict[str, Callable] = {
    "pooled_loglog": _sp_pooled,
    "twoway_fe": _sp_fe,
    "poisson_glm": _sp_poisson,
    "lgbm_demand": _sp_lgbm,
    "dml_partialling": _sp_dml,
    "r_learner": _sp_rlearner,
    "shrunk_per_product": _sp_shrunk,
}


def suggest_params(model_name: str, trial) -> dict:
    return SEARCH_SPACES[model_name](trial)


# =======================================================================================
# the search
# =======================================================================================
#: Metrics where larger is better; everything else is minimised.
_MAXIMISE = {"rloss_skill", "rloss_vs_constant"}


def tune(model_name: str, df: pd.DataFrame, *, objective: str = "rloss",
         n_trials: int = 40, scorer: CausalScorer | None = None,
         splitter=None, truth_col: str | None = None, seed: int = 0,
         pruning: bool = True, timeout: float | None = None,
         show_progress: bool = False) -> dict:
    """Run an Optuna TPE search and return the study plus a tidy trial table.

    Every trial records **all** metrics, not just the one being optimised, so the same
    history can be re-read under a different objective afterwards — that is what
    :func:`objective_comparison` exploits.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if scorer is None:
        scorer = CausalScorer(df, splitter or RollingOriginSplit(), seed=seed)
    records: list[dict] = []
    # The params are also kept as their original dicts, not just flattened into `records`.
    # A DataFrame row has one dtype, so reading them back out of `history` upcasts every
    # int to float — and LightGBM rejects ``n_estimators=450.0`` outright.
    param_store: dict[int, dict] = {}

    def obj(trial):
        params = suggest_params(model_name, trial)
        param_store[trial.number] = dict(params)
        s = cv_score(model_name, params, df, scorer, truth_col=truth_col,
                     trial=trial if pruning else None, objective=objective)
        records.append({"trial": trial.number, **params, **s})
        v = s[objective]
        return -v if objective in _MAXIMISE else v

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
              if pruning else optuna.pruners.NopPruner())
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(obj, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=show_progress, catch=(Exception,))

    hist = pd.DataFrame(records)
    ok = hist[hist.get("failed", 0) == 0] if len(hist) else hist
    if len(ok) == 0:
        raise RuntimeError(f"every trial failed for {model_name}")
    # A pruned trial raises before it reaches `records`, so it is absent from `hist`
    # entirely — counting `len(hist) - len(ok)` would report *failures* as prunes.  Ask the
    # study, which is the only object that saw them.
    states = [t.state for t in study.trials]
    n_pruned = sum(s == optuna.trial.TrialState.PRUNED for s in states)
    n_failed = int((hist.get("failed", pd.Series(dtype=float)) == 1).sum())
    best_row = (ok.loc[ok[objective].idxmax()] if objective in _MAXIMISE
                else ok.loc[ok[objective].idxmin()])
    best_params = dict(param_store[int(best_row["trial"])])
    return {"model": model_name, "objective": objective, "study": study,
            "history": hist, "best_params": best_params,
            "best_metrics": {k: float(best_row[k]) for k in
                             ("rmse", "rloss", "rloss_skill", "rloss_vs_constant",
                              "avg_eps", "eps_sd") if k in best_row}
            | ({"oracle": float(best_row["oracle"]),
                "oracle_bias": float(best_row["oracle_bias"])}
               if "oracle" in best_row else {}),
            "n_complete": int(len(ok)), "n_pruned": n_pruned, "n_failed": n_failed}


class _DummyTrial:
    """Walks a search space once, without an Optuna run.

    Used by :func:`default_params` to report what a space contains.  ``tune`` no longer
    needs it: it keeps each trial's parameter dict verbatim rather than reconstructing one
    from the history table, because a DataFrame row has a single dtype and reading an int
    back out of one returns a float.
    """

    def suggest_float(self, name, lo, hi, **kw):
        return lo

    def suggest_int(self, name, lo, hi, **kw):
        return lo

    def suggest_categorical(self, name, choices):
        return choices[0]


def default_params(model_name: str) -> dict:
    """The low corner of a model's search space — handy for smoke tests and docs."""
    return SEARCH_SPACES[model_name](_DummyTrial())


def tuning_history(result: dict) -> pd.DataFrame:
    """Trial table with a running best, for the convergence plot."""
    h = result["history"].copy()
    o = result["objective"]
    h = h[h.get("failed", 0) == 0].sort_values("trial")
    h["running_best"] = h[o].cummax() if o in _MAXIMISE else h[o].cummin()
    return h


# =======================================================================================
# the central experiment: which objective should you tune on?
# =======================================================================================
def objective_comparison(model_name: str, df_syn: pd.DataFrame, *,
                         truth_col: str = "true_elasticity",
                         objectives: Iterable[str] = ("rmse", "rloss"),
                         n_trials: int = 40, splitter=None, seed: int = 0) -> pd.DataFrame:
    """Tune the same model under each objective; grade all winners against the truth.

    This is the experiment the module exists for.  Only the synthetic panel can run it,
    because only there is ``truth_col`` available — but what it establishes (which
    *observable* objective picks hyperparameters that recover elasticity) is exactly what
    transfers to the real panel, where the oracle column does not exist.
    """
    scorer = CausalScorer(df_syn, splitter or RollingOriginSplit(), seed=seed)
    rows, histories = [], []
    for o in objectives:
        r = tune(model_name, df_syn, objective=o, n_trials=n_trials, scorer=scorer,
                 truth_col=truth_col, seed=seed)
        histories.append(r["history"].assign(tuned_on=o))
        rows.append({"model": model_name, "tuned_on": o, **r["best_metrics"],
                     "n_complete": r["n_complete"], "n_pruned": r["n_pruned"],
                     "best_params": r["best_params"]})

    # The oracle-tuned winner: the best elasticity recovery reachable anywhere in the
    # region the searches actually explored.  It is not achievable in practice — picking
    # it requires the truth — but it bounds what a better observable objective could buy.
    allh = pd.concat(histories, ignore_index=True)
    ok = allh[(allh.get("failed", 0) == 0) & allh["oracle"].notna()]
    if len(ok):
        b = ok.loc[ok["oracle"].idxmin()]
        rows.append({"model": model_name, "tuned_on": "oracle (unachievable bound)",
                     **{k: float(b[k]) for k in
                        ("rmse", "rloss", "rloss_skill", "rloss_vs_constant",
                         "avg_eps", "eps_sd", "oracle", "oracle_bias") if k in b},
                     "n_complete": int(len(ok)), "n_pruned": 0, "best_params": {}})
    out = pd.DataFrame(rows)
    out.attrs["history"] = allh
    return out


# =======================================================================================
# nested CV — honest reporting after a search
# =======================================================================================
def nested_cv(model_name: str, df: pd.DataFrame, *, objective: str = "rloss",
              n_trials: int = 25, n_outer: int = 3, truth_col: str | None = None,
              seed: int = 0) -> pd.DataFrame:
    """Tune inside each outer fold; score on data the search never touched.

    A single tuned score is optimistically biased — the winner is the maximum over many
    noisy estimates, so it carries the maximum's upward bias.  Nested CV pays roughly
    ``n_outer ×`` the compute to remove it, and the gap between the two is itself worth
    reporting: it says how much of the apparent gain was selection noise.
    """
    outer = RollingOriginSplit(n_splits=n_outer + 1, horizon=8, min_train_weeks=26)
    folds = list(outer.split(df))
    rows = []
    for i, (tr, va) in enumerate(folds[-n_outer:]):
        dtr, dva = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
        inner = RollingOriginSplit(n_splits=3, horizon=6,
                                   min_train_weeks=max(12, dtr.week.nunique() - 24))
        r = tune(model_name, dtr, objective=objective, n_trials=n_trials,
                 splitter=inner, truth_col=truth_col, seed=seed + i)
        # Score the winner on the untouched outer block.  attrs are cleared first: pandas
        # compares them element-wise on concat, and a non-scalar entry raises.
        dtr.attrs, dva.attrs = {}, {}
        combined = pd.concat([dtr, dva], ignore_index=True)
        outer_scorer = CausalScorer(combined, _FixedSplit(len(dtr), len(dva)), seed=seed)
        s = cv_score(model_name, r["best_params"], combined, outer_scorer,
                     truth_col=truth_col)
        rows.append({"outer_fold": i, "model": model_name, "tuned_on": objective,
                     "inner_best": r["best_metrics"].get(objective, np.nan),
                     **{f"outer_{k}": v for k, v in s.items()},
                     "best_params": r["best_params"]})
    return pd.DataFrame(rows)


@dataclass
class _FixedSplit:
    """A single pre-determined train/validation cut, for scoring inside nested CV."""

    n_train: int
    n_valid: int

    def split(self, df):
        yield np.arange(self.n_train), np.arange(self.n_train, self.n_train + self.n_valid)

    def get_n_splits(self, df):
        return 1
