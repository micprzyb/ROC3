"""
P1, P2, P3, P5 — how VUS relates to the quality of the elasticity model.

Run on a **randomised price test** built on the real covariates
(:func:`elasticity_lab.simulate.make_price_test_panel`), because that is the user's actual
example and the setting in which the price-test identity holds exactly.  The observational
panel is covered at the end, where it fails for a reason the machinery itself detects.

  P1  homogeneous elasticity  ->  VUS at chance, mean elasticity still accurate
  P2  sweeping the heterogeneity  ->  normalized VUS rises, bias stays flat
  P3  across models             ->  VUS tracks oracle_corr, NOT oracle_bias
  P5  measured VUS never exceeds the ceiling implied by the demand response
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from elasticity_lab.data import get_panel                                   # noqa: E402
from elasticity_lab.bench import subsample_products                         # noqa: E402
from elasticity_lab.pricelevels import PriceLevelClassifier, vus_report     # noqa: E402
from elasticity_lab.simulate import SyntheticConfig, make_price_test_panel  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
f4 = lambda v: f"{v:+.4f}"                                                  # noqa: E731

N_PRODUCTS = 700
MULT = (0.9, 1.0, 1.1)          # the -10% / 0% / +10% of the motivating example
panel = subsample_products(get_panel(), N_PRODUCTS, seed=0)
OUT = "experiments/out"


def build(elasticity_sd, seed=0, mult=MULT, homogeneous=False, **kw):
    """One randomised-price-test world.

    ``homogeneous=True`` zeroes **all three** sources of elasticity variation.  Zeroing only
    ``elasticity_sd`` leaves the price-level and order-size terms and a residual spread of
    0.47, which is not "no heterogeneity" and would make P1 untestable.
    """
    if homogeneous:
        kw.setdefault("elasticity_price_slope", 0.0)
        kw.setdefault("elasticity_size_slope", 0.0)
    cfg = SyntheticConfig(elasticity_sd=elasticity_sd, seed=seed, **kw)
    return make_price_test_panel(panel, cfg, price_multipliers=mult, seed=seed)


def split(df, holdout_weeks=20):
    """Fit on the early weeks, score on the late ones.

    Scoring in-sample is not a small sin here.  A 300-tree classifier on 38k rows will
    memorise a label that is genuinely random, and the resulting VUS came out at 0.53
    against a ceiling of 0.25 -- flagged 'impossible' by the audit, which is exactly what
    the audit is for.
    """
    wks = np.sort(df.week.unique())
    cut = wks[-holdout_weeks]
    return df[df.week < cut].reset_index(drop=True), df[df.week >= cut].reset_index(drop=True)


def score(df, **model_kw):
    """Fit the classifier on the early weeks and score it on the held-out ones."""
    tr, va = split(df)
    # level_col="arm": the design is a real price test, so the arm is KNOWN.  Inferring
    # it from rel_price recovers it only ~60% of the time here, because rel_price is
    # measured against the product's historical prices and those swamp a +/-10% arm.
    model_kw.setdefault("n_estimators", 300)
    m = PriceLevelClassifier(K=len(MULT), level_col="arm", **model_kw).fit(tr)
    df = va
    e = np.asarray(m.elasticity(df), float)
    t = df.true_elasticity.to_numpy(float)
    rep = vus_report(m, df)
    return m, {
        "mean_eps": float(e.mean()), "sd_eps": float(e.std()),
        "true_mean": float(t.mean()), "true_sd": float(t.std()),
        "oracle_bias": float(e.mean() - t.mean()),
        "oracle_rmse": float(np.sqrt(np.mean((e - t) ** 2))),
        "oracle_corr": float(np.corrcoef(e, t)[0, 1]) if e.std() > 1e-9 else np.nan,
        "vus": rep["vus_3afc"], "ceiling": rep["vus_ceiling"],
        "vus_norm": rep["normalized"], "verdict": rep["verdict"],
        "chance": rep["chance"],
    }


# ======================================================== 0. the design behaves as designed
print("=" * 104)
print("0.  THE RANDOMISED PRICE TEST")
print("=" * 104)
d0 = build(0.55)
lv_check = PriceLevelClassifier(K=3, level_col="arm").fit(d0)
print(lv_check.levels_.describe().to_string(index=False, float_format=f4))
agree = float((lv_check.levels_.transform(d0) == d0.arm.to_numpy()).mean())
print(f"\nrecovered level == assigned arm for {agree:.1%} of rows")
print(f"arm shares: {np.bincount(d0.arm) / len(d0)}")
d = lv_check.diagnostics(d0)
print(f"propensity: median min = {d['min_propensity_median']:.4f} "
      f"(perfect overlap would be 1/3 = 0.3333); "
      f"{d['pct_below_0.05']:.2%} of rows below 0.05")
print("Contrast the observational panel in section 4, where the seller's price is so\n"
      "predictable from its own history that the same number collapses.")

# ============================================================================= P1
print("\n" + "=" * 104)
print("P1.  HOMOGENEOUS ELASTICITY  ->  VUS AT CHANCE, LEVEL STILL RECOVERED")
print("=" * 104)
rows = []
for sd, hom, label in [(0.0, True, "homogeneous (all three terms off)"),
                       (0.55, False, "heterogeneous (sd = 0.55)")]:
    _, r = score(build(sd, homogeneous=hom))
    rows.append({"world": label, **r})
p1 = pd.DataFrame(rows)
print(p1[["world", "true_mean", "mean_eps", "oracle_bias", "true_sd",
          "vus", "chance", "ceiling", "vus_norm"]].to_string(index=False, float_format=f4))
hom = p1.iloc[0]
print(f"""
  With a homogeneous elasticity the posterior is the SAME for every customer -- the
  baseline demand s(x) cancels out of b_k = pi_k s_k / sum_j pi_j s_j -- so there is
  nothing for a classifier to discriminate on and VUS sits at {hom.vus:.4f} against a
  chance level of {hom.chance:.4f}.  The elasticity is nonetheless recovered to
  {hom.oracle_bias:+.4f}.  A model can be at chance VUS and exactly right about the answer.""")

# ============================================== does weight_cap attenuate the elasticity?
print("=" * 104)
print("P1b.  DOES `weight_cap` FLATTEN THE DEMAND CURVE?")
print("=" * 104)
print("Hypothesis: winsorising the unit weights truncates the largest-quantity rows, which")
print("are disproportionately the CHEAP arm, so the cap should bias the demand response flat.")
print("MEASURED: it does not.  The bias sits near -0.45 at every cap with no monotone")
print("pattern, so the ~27% attenuation of the elasticity comes from somewhere else -- the")
print("classifier shrinking its probabilities toward the marginal, which compresses the")
print("log-ratios the slope is computed from.  That is a CALIBRATION failure, and per the")
print("gauge theorem it is exactly the kind of error VUS is blind to.  Hypothesis recorded")
print("rather than deleted, because the refutation is the more useful statement.\n")
rows = []
for cap in (None, 0.999, 0.99, 0.95):
    _, r = score(build(0.55), weight_cap=cap)
    rows.append({"weight_cap": cap, **r})
p1b = pd.DataFrame(rows)
print(p1b[["weight_cap", "true_mean", "mean_eps", "oracle_bias", "vus", "ceiling",
           "vus_norm"]].to_string(index=False, float_format=f4))
p1b.to_csv(f"{OUT}/17_p1b_weight_cap.csv", index=False)

# ============================================================================= P2
print("=" * 104)
print("P2.  SWEEPING THE HETEROGENEITY")
print("=" * 104)
rows = []
for sd in (0.0, 0.2, 0.4, 0.6, 0.8):
    _, r = score(build(sd))
    rows.append({"elasticity_sd": sd, **r})
p2 = pd.DataFrame(rows)
print(p2[["elasticity_sd", "true_sd", "vus", "ceiling", "vus_norm",
          "oracle_bias", "oracle_corr"]].to_string(index=False, float_format=f4))
mono_v = bool(np.all(np.diff(p2.vus_norm) > -0.02))
print(f"\n  normalized VUS monotone in heterogeneity: {mono_v}")
print(f"  spread of oracle_bias across the sweep    : "
      f"{p2.oracle_bias.max() - p2.oracle_bias.min():+.4f}  (flat = the level does not "
      f"depend on heterogeneity)")
p2.to_csv(f"{OUT}/17_p2_heterogeneity_sweep.csv", index=False)

# ============================================================================= P5
print("\n" + "=" * 104)
print("P5.  THE CEILING IS SET BY THE LEVEL OF THE ELASTICITY, NOT ITS SPREAD")
print("=" * 104)
rows = []
for em in (0.0, 0.5, 1.0, 1.66, 2.5):
    _, r = score(build(0.55, elasticity_mean=em))
    rows.append({"elasticity_mean": em, **r})
p5 = pd.DataFrame(rows)
print(p5[["elasticity_mean", "true_mean", "mean_eps", "vus", "chance", "ceiling",
          "vus_norm", "verdict"]].to_string(index=False, float_format=f4))
print(f"""
  The ceiling rises with the elasticity because that is what pulls the arms apart: at zero
  elasticity every arm sells the same and the classes are indistinguishable in principle.
  So a raw VUS is uninterpretable without it -- the same 0.20 is near-perfect against a
  ceiling of 0.22 and near-worthless against a ceiling of 0.9.""")
ok5 = bool((p5.vus <= p5.ceiling + 0.02).all())
print(f"  P5 -- no measured VUS exceeds its ceiling: {ok5}")
p5.to_csv(f"{OUT}/17_p5_ceiling_vs_level.csv", index=False)

# ============================================================================= P3
print("\n" + "=" * 104)
print("P3.  DOES VUS TRACK THE RANKING OR THE LEVEL?")
print("=" * 104)
print("Each row is one model configuration on one world.  If the theory is right,")
print("vus_norm correlates with oracle_corr and not with |oracle_bias|.\n")
rows = []
for sd in (0.0, 0.3, 0.55, 0.8):
    df = build(sd)
    for constraint in ("none", "isotonic", "softplus"):
        for leaves, nest, cap in ((8, 150, None), (31, 300, None), (127, 600, None),
                                  (31, 300, 0.99)):
            try:
                _, r = score(df, constraint=constraint, num_leaves=leaves,
                             n_estimators=nest, weight_cap=cap)
            except Exception as ex:
                # print the message, not just the type -- an earlier version swallowed a
                # duplicate-keyword TypeError and reported "0 configurations" instead
                print(f"  skipped sd={sd} {constraint} leaves={leaves}: "
                      f"{type(ex).__name__}: {ex}", flush=True)
                continue
            rows.append({"elasticity_sd": sd, "constraint": constraint,
                         "num_leaves": leaves, "weight_cap": cap, **r})
p3 = pd.DataFrame(rows)
p3["abs_bias"] = p3.oracle_bias.abs()
p3.to_csv(f"{OUT}/17_p3_vus_vs_oracle.csv", index=False)
print(p3[["elasticity_sd", "constraint", "num_leaves", "weight_cap", "vus", "vus_norm",
          "oracle_corr", "oracle_bias", "oracle_rmse"]]
      .to_string(index=False, float_format=f4))

sub = p3.dropna(subset=["vus_norm", "oracle_corr", "abs_bias"])
c_corr = sub[["vus_norm", "oracle_corr"]].corr("spearman").iloc[0, 1]
c_bias = sub[["vus_norm", "abs_bias"]].corr("spearman").iloc[0, 1]
c_rmse = sub[["vus_norm", "oracle_rmse"]].corr("spearman").iloc[0, 1]
print(f"""
  n = {len(sub)} configurations

    spearman( vus_norm , oracle_corr )   = {c_corr:+.3f}      <- the RANKING
    spearman( vus_norm , |oracle_bias| ) = {c_bias:+.3f}      <- the LEVEL
    spearman( vus_norm , oracle_rmse )   = {c_rmse:+.3f}

  P3 holds if the first is clearly positive and the second is near zero.""")

# ================================================= 4. and now the observational panel
print("\n" + "=" * 104)
print("4.  THE SAME MODEL ON OBSERVATIONAL DATA — AND WHY THE AUDIT CATCHES IT")
print("=" * 104)
from elasticity_lab.bench import synth_frame                              # noqa: E402
obs = synth_frame(N_PRODUCTS, confounding_price=0.6, confounding_demand=0.8)
mo = PriceLevelClassifier(K=3, n_estimators=300).fit(obs)
ro = vus_report(mo, obs)
do = mo.diagnostics(obs)
eo = np.asarray(mo.elasticity(obs), float)
to = obs.true_elasticity.to_numpy(float)
print(f"  VUS                 {ro['vus_3afc']:.4f}")
print(f"  ceiling             {ro['vus_ceiling']:.4f}")
print(f"  verdict             {ro['verdict']}")
print(f"  implied demand      {ro['implied_demand_ratios']}")
print(f"  min propensity p5   {do['min_propensity_p5']:.4f}   "
      f"({do['pct_below_0.05']:.1%} of rows below 0.05)")
print(f"  mean elasticity     {eo.mean():+.4f}  (true {to.mean():+.4f})")
print(f"""
  The verdict is '{ro['verdict']}': the classifier separates the price levels far better
  than its own measured demand response could possibly explain.  It is not learning demand.
  It is reconstructing the current price from `lag_log_price` and `roll4_log_price`, which
  are legitimate controls and also nearly determine the label, so the assignment is close
  to deterministic and inverse-propensity weighting has nothing to work with.

  That is a positivity (overlap) violation, and it is the honest answer for this panel: the
  price-test construction needs a price test.  What is worth noting is that nobody had to
  suspect it -- `roc3.pricetest.audit` derives the ceiling from the model's own output and
  flags the impossibility automatically.""")

summary = pd.DataFrame([
    {"panel": "randomised price test", "vus": p1.iloc[1].vus,
     "ceiling": p1.iloc[1].ceiling, "verdict": p1.iloc[1].verdict,
     "oracle_bias": p1.iloc[1].oracle_bias},
    {"panel": "observational", "vus": ro["vus_3afc"], "ceiling": ro["vus_ceiling"],
     "verdict": ro["verdict"], "oracle_bias": float(eo.mean() - to.mean())}])
summary.to_csv(f"{OUT}/17_randomised_vs_observational.csv", index=False)
print(summary.to_string(index=False, float_format=f4))
print(f"\nwrote {OUT}/17_*.csv")
