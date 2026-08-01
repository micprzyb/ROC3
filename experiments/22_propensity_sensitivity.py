"""
How much of the insurance work depends on the allocation being exactly 0.1 / 0.8 / 0.1?

The number came from the user's description of a live test.  It was hard-coded as a default
and never independently verified, and no public dataset was found with randomised price
arms and a documented allocation -- the allocation and the elasticity are exactly what an
insurer would not publish.  So the honest thing is to measure the dependence.

Three questions.

  1  Does the *design* allocation change the qualitative conclusions?
  2  What if pi is MISSPECIFIED -- analysed as 0.1/0.8/0.1 when the truth is something else?
  3  What if pi is HETEROGENEOUS -- the test excludes or under-samples some segments, so
     pi_k(x) is not constant?  This is the realistic failure: staged rollouts, channel-
     specific test rates, business overrides on renewals.

Question 2 has a closed-form answer, and it is the gauge theorem again.  Analysing with
pi' instead of pi gives

    log s_hat_k  =  log b_k - log pi'_k  =  log s_k + (log pi_k - log pi'_k)

which is precisely the gauge map with c_k = pi_k / pi'_k.  So VUS is untouched, and each
arc elasticity shifts by -(d log c) / (d r).  The prediction that follows is worth stating
before measuring it:

    a SYMMETRIC error in the side-arm rate (0.05/0.90/0.05 read as 0.1/0.8/0.1) leaves the
    error vector symmetric, so its projection on the linear trend in r is ZERO and the
    overall slope is UNBIASED -- while the individual arc elasticities are wrecked.

    an ASYMMETRIC error (0.08/0.80/0.12 read as 0.1/0.8/0.1) does bias the slope.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from elasticity_lab.insurance import (ArmPosteriorModel, ConversionGBMv2,   # noqa: E402
                                      ConversionGLM, MotorQuoteConfig,
                                      make_quote_panel, policy_profit, profit_regret,
                                      lerner_price)

pd.set_option("display.width", 220)
f4 = lambda v: f"{v:+.4f}"                                                  # noqa: E731
OUT = "experiments/out"
N = 200_000
MULT = np.array([0.9, 1.0, 1.1])
LOGM = np.log(MULT)


def build(alloc, link="logit", seed=0):
    cfg = MotorQuoteConfig(n_quotes=N, seed=seed, link=link,
                           arm_multipliers=tuple(MULT), arm_probs=tuple(alloc))
    d = make_quote_panel(cfg)
    k = int(0.7 * len(d))
    tr, te = d.iloc[:k].copy(), d.iloc[k:].copy()
    for f in (tr, te):
        f.attrs.update(d.attrs)
    return d, tr, te


def slope_from_logs(err):
    """Projection of a per-arm log error onto the elasticity slope, exactly as step D does."""
    x = LOGM - LOGM.mean()
    return -float(np.dot(err, x) / np.dot(x, x))


# ==================================================== 1. the closed-form prediction
print("=" * 108)
print("1.  MISSPECIFYING pi IS A GAUGE TRANSFORM -- THE BIAS IS PREDICTABLE IN CLOSED FORM")
print("=" * 108)
assumed = np.array([0.1, 0.8, 0.1])
cases = {"correct": [0.1, 0.8, 0.1],
         "symmetric, sides halved": [0.05, 0.90, 0.05],
         "symmetric, sides doubled": [0.20, 0.60, 0.20],
         "asymmetric, dear arm bigger": [0.08, 0.80, 0.12],
         "asymmetric, cheap arm bigger": [0.12, 0.80, 0.08]}
rows = []
for lbl, true_pi in cases.items():
    tp = np.array(true_pi) / np.sum(true_pi)
    err = np.log(tp) - np.log(assumed)          # what gets added to log s_hat
    rows.append({"true pi": str(list(np.round(tp, 3))), "case": lbl,
                 "log error by arm": str(list(np.round(err, 3))),
                 "predicted slope bias": slope_from_logs(err),
                 "predicted arc1 bias": -(err[1] - err[0]) / (LOGM[1] - LOGM[0]),
                 "predicted arc2 bias": -(err[2] - err[1]) / (LOGM[2] - LOGM[1])})
pred = pd.DataFrame(rows)
print(pred.to_string(index=False, float_format=f4))
print("""
  The symmetric rows have a predicted SLOPE bias of exactly zero while their ARC biases are
  +/-8 or more.  A symmetric error in the side-arm rate cancels in the linear projection.
  The asymmetric rows do not cancel.  Now measure it.""")

# ==================================================== 2. measured, on simulated data
print("=" * 108)
print("2.  MEASURED: generate with the true pi, analyse as if it were 0.1/0.8/0.1")
print("=" * 108)
rows = []
for lbl, true_pi in cases.items():
    d, tr, te = build(true_pi)
    truth = te["true_elasticity_control"].to_numpy(float)
    # honest: the model is told the real allocation
    m_ok = ArmPosteriorModel(balance="none", constraint="none").fit(tr)
    e_ok = np.asarray(m_ok.elasticity(te), float)
    # misspecified: override the model's pi with the assumed one
    m_bad = ArmPosteriorModel(balance="none", constraint="none").fit(tr)
    m_bad.pi_ = assumed.copy()
    e_bad = np.asarray(m_bad.elasticity(te), float)
    rows.append({"case": lbl,
                 "true_mean_eps": truth.mean(),
                 "eps_correct_pi": e_ok.mean(), "eps_assumed_pi": e_bad.mean(),
                 "measured_slope_bias": float(e_bad.mean() - e_ok.mean()),
                 "predicted_slope_bias": pred.loc[pred.case == lbl,
                                                  "predicted slope bias"].iloc[0]})
    print(f"  {lbl} done", flush=True)
m = pd.DataFrame(rows)
m["prediction_error"] = m.measured_slope_bias - m.predicted_slope_bias
print("\n" + m.to_string(index=False, float_format=f4))
m.to_csv(f"{OUT}/22_pi_misspecification.csv", index=False)
ok = bool(np.abs(m.prediction_error).max() < 0.02)
print(f"\n  gauge theorem predicts the bias to within {np.abs(m.prediction_error).max():.4f}"
      f"   -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")

# ==================================================== 3. which models even care?
print("\n" + "=" * 108)
print("3.  WHICH MODELS DEPEND ON pi AT ALL?")
print("=" * 108)
d, tr, te = build([0.1, 0.8, 0.1])
truth = te["true_elasticity_control"].to_numpy(float)
rows = []
for name, mk in (("conversion_glm", lambda: ConversionGLM()),
                 ("conversion_gbm_v2", lambda: ConversionGBMv2()),
                 ("arm_posterior", lambda: ArmPosteriorModel(balance="none",
                                                             constraint="none"))):
    mm = mk().fit(tr)
    base = float(np.asarray(mm.elasticity(te), float).mean())
    if hasattr(mm, "pi_"):
        mm.pi_ = np.array([0.05, 0.90, 0.05])
        shifted = float(np.asarray(mm.elasticity(te), float).mean())
    else:
        shifted = base                       # the model never sees pi
    rows.append({"model": name, "uses_pi": hasattr(mm, "pi_"),
                 "eps_true_pi": base, "eps_wrong_pi": shifted,
                 "shift": shifted - base})
print(pd.DataFrame(rows).to_string(index=False, float_format=f4))
print("""
  Only the arm-posterior family reads pi.  The GLM and the GBM take the price as a feature
  and never touch the allocation, so a misspecified pi cannot hurt them -- which, given how
  uncertain the 0.1/0.8/0.1 figure is, is a point in their favour that had not been counted.""")

# ==================================================== 4. heterogeneous allocation
print("\n" + "=" * 108)
print("4.  THE REALISTIC FAILURE: pi VARIES BY SEGMENT AND IS ASSUMED CONSTANT")
print("=" * 108)
print("Staged rollouts, channel-specific test rates and business overrides all produce")
print("this.  Here the test is run at 0.2/0.6/0.2 on aggregator traffic and 0.02/0.96/0.02")
print("elsewhere, then analysed as if it were the pooled average.\n")
d, tr, te = build([0.1, 0.8, 0.1])
rng = np.random.default_rng(0)
rows = []
for frame in (tr, te):
    agg = frame["channel_aggregator"].to_numpy() > 0.5
    pi_hi, pi_lo = np.array([0.2, 0.6, 0.2]), np.array([0.02, 0.96, 0.02])
    p = np.where(agg[:, None], pi_hi[None, :], pi_lo[None, :])
    u = rng.random(len(frame))
    cum = np.cumsum(p, axis=1)
    new_arm = (u[:, None] > cum).sum(axis=1)
    frame["arm"] = new_arm
    frame["arm_prob"] = p[np.arange(len(frame)), new_arm]
    frame["arm_log_multiplier"] = LOGM[new_arm]
    frame["log_price"] = frame["log_technical_premium"] + frame["arm_log_multiplier"]
    a = frame["_a"].to_numpy(); b = frame["true_beta"].to_numpy()
    s = 1 / (1 + np.exp(-(a - b * frame["log_price"].to_numpy())))
    frame["converted"] = (rng.random(len(frame)) < s).astype(int)

pooled = np.array([tr.arm_prob.mean() if False else 0, 0, 0])
pooled = np.bincount(tr.arm.to_numpy(), minlength=3) / len(tr)
print(f"  realised pooled allocation: {np.round(pooled, 4)}")
truth = te["true_elasticity_control"].to_numpy(float)
for lbl, pi_used in (("assume the pooled average", pooled),
                     ("assume the design 0.1/0.8/0.1", assumed)):
    mm = ArmPosteriorModel(balance="none", constraint="none").fit(tr)
    mm.pi_ = np.asarray(pi_used, float)
    e = np.asarray(mm.elasticity(te), float)
    rows.append({"handling": lbl, "mean_eps": e.mean(),
                 "bias": e.mean() - truth.mean(),
                 "rmse": float(np.sqrt(np.mean((e - truth) ** 2)))})
# and the correct handling: per-row propensity
mm = ArmPosteriorModel(balance="none", constraint="none").fit(tr)
b = mm.posterior(te, balanced=False)
ls = np.log(b) - np.log(te[["arm_prob"]].to_numpy())      # wrong on purpose? no --
# the honest version needs pi_k(x) for EVERY k, which the design supplies
agg_te = te["channel_aggregator"].to_numpy() > 0.5
pi_row = np.where(agg_te[:, None], np.array([0.2, 0.6, 0.2])[None, :],
                  np.array([0.02, 0.96, 0.02])[None, :])
ls = np.log(b) - np.log(pi_row)
x = LOGM - LOGM.mean()
e_row = -(ls * x[None, :]).sum(1) / np.sum(x ** 2)
rows.append({"handling": "use the true pi_k(x) per row", "mean_eps": e_row.mean(),
             "bias": e_row.mean() - truth.mean(),
             "rmse": float(np.sqrt(np.mean((e_row - truth) ** 2)))})
h = pd.DataFrame(rows)
print("\n" + h.to_string(index=False, float_format=f4))
h.to_csv(f"{OUT}/22_heterogeneous_pi.csv", index=False)
print("""
  A constant pi is not a harmless approximation when the test was not run uniformly.  The
  fix is cheap and exact -- record pi_k(x) per quote at assignment time and carry it -- but
  it has to be DESIGNED IN.  Reconstructing it afterwards from realised shares is precisely
  the pooled-average row.""")
print(f"\nwrote {OUT}/22_*.csv")
