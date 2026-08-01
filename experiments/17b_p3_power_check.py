"""
P3 failed.  Is the prediction wrong, or was the test underpowered?

At the motivating -10%/0%/+10% arms the measured VUS spanned 0.170 to 0.180 against a
chance level of 0.167 — a signal of roughly 0.01 — and over 48 configurations the rank
correlation with ``oracle_corr`` came out at **-0.358**, the opposite of the prediction.

Two readings, and they are distinguishable.  If the failure is a power problem, widening
the arms raises the ceiling, gives VUS room to move, and the predicted correlation should
appear.  If the prediction is simply wrong, it will not.  This script runs the same design
at three arm widths and reports the correlation for each.

Widening the arms is not cheating: it changes the experiment, not the analysis.  A real
price test with wider arms genuinely does carry more information about heterogeneity, and
the question here is whether VUS reflects that information when it exists.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from elasticity_lab.bench import subsample_products                         # noqa: E402
from elasticity_lab.data import get_panel                                   # noqa: E402
from elasticity_lab.pricelevels import PriceLevelClassifier, vus_report     # noqa: E402
from elasticity_lab.simulate import SyntheticConfig, make_price_test_panel  # noqa: E402

pd.set_option("display.width", 200)
f4 = lambda v: f"{v:+.4f}"                                                  # noqa: E731
panel = subsample_products(get_panel(), 700, seed=0)

WIDTHS = {"+/-10% (the motivating test)": (0.9, 1.0, 1.1),
          "+/-25%": (0.75, 1.0, 1.25),
          "+/-50%": (0.5, 1.0, 1.5)}


def run(mult, sd, **kw):
    cfg = SyntheticConfig(elasticity_sd=sd, seed=0)
    df = make_price_test_panel(panel, cfg, price_multipliers=mult, seed=0)
    wks = np.sort(df.week.unique())
    tr, va = df[df.week < wks[-20]], df[df.week >= wks[-20]]
    kw.setdefault("n_estimators", 300)
    m = PriceLevelClassifier(K=3, level_col="arm", **kw).fit(tr)
    e = np.asarray(m.elasticity(va), float)
    t = va.true_elasticity.to_numpy(float)
    rep = vus_report(m, va)
    return {"vus": rep["vus_3afc"], "ceiling": rep["vus_ceiling"],
            "vus_norm": rep["normalized"], "verdict": rep["verdict"],
            "oracle_corr": float(np.corrcoef(e, t)[0, 1]) if e.std() > 1e-9 else np.nan,
            "oracle_bias": float(e.mean() - t.mean()),
            "oracle_rmse": float(np.sqrt(np.mean((e - t) ** 2)))}


rows = []
for label, mult in WIDTHS.items():
    for sd in (0.0, 0.3, 0.55, 0.8):
        for leaves, nest in ((8, 150), (31, 300), (127, 600)):
            r = run(mult, sd, num_leaves=leaves, n_estimators=nest)
            rows.append({"arms": label, "elasticity_sd": sd, "num_leaves": leaves, **r})
            print(f"  {label:28s} sd={sd:.2f} leaves={leaves:3d}  "
                  f"vus={r['vus']:.4f} ceil={r['ceiling']:.4f} "
                  f"norm={r['vus_norm']:+.3f} corr={r['oracle_corr']:+.3f}", flush=True)
t = pd.DataFrame(rows)
t["abs_bias"] = t.oracle_bias.abs()
t.to_csv("experiments/out/17b_p3_power.csv", index=False)

print("\n" + "=" * 96)
print("P3 BY ARM WIDTH")
print("=" * 96)
out = []
for label in WIDTHS:
    g = t[t.arms == label].dropna(subset=["vus_norm", "oracle_corr"])
    out.append({
        "arms": label, "n": len(g),
        "vus_span": float(g.vus.max() - g.vus.min()),
        "mean_ceiling": float(g.ceiling.mean()),
        "rho(vus_norm, oracle_corr)": g[["vus_norm", "oracle_corr"]].corr("spearman").iloc[0, 1],
        "rho(vus_norm, |bias|)": g[["vus_norm", "abs_bias"]].corr("spearman").iloc[0, 1]})
o = pd.DataFrame(out)
print(o.to_string(index=False, float_format=f4))

print(f"""
READING THIS

  If `rho(vus_norm, oracle_corr)` turns positive as the arms widen, the original failure
  was a power problem: at +/-10% the VUS signal is about 0.01 above chance and a rank
  correlation over that is measuring noise.  The theory of PLAN section 4 is a theorem
  either way -- the gauge argument does not depend on any of this -- but the empirical
  corollary would then be "detectable only when the test is wide enough".

  If it stays negative at every width, the corollary is wrong and must be withdrawn.

  Either way, the practical advice is unchanged and is worth more than the correlation:
  a raw VUS on a +/-10% price test lives in [0.167, ~0.25], so quoting it without the
  ceiling is meaningless, and quoting it as evidence about the elasticity LEVEL is
  provably meaningless.""")


# =======================================================================================
# The design error, and the corrected test
# =======================================================================================
print("\n" + "=" * 96)
print("THE TEST WAS MIS-SPECIFIED.  `elasticity_sd` IS NOT LEARNABLE HETEROGENEITY.")
print("=" * 96)
print("""
`elasticity_sd` adds an independent random draw per product.  The controls contain only
coarse product summaries (`prod_log_price_mean`, `prod_n_weeks`, ...) and do not identify
the product, so that draw is **invisible to any classifier**.  Raising it raises
`oracle_corr`'s denominator without giving VUS anything to find -- which is why `vus_span`
stayed at 0.007 even at +/-50% arms where the ceiling had risen to 0.55.

VUS measures the heterogeneity a model can *recover*.  The corrected sweep therefore varies
`elasticity_price_slope`, which makes the elasticity a deterministic function of an observed
control, and is learnable in principle.
""", flush=True)

rows = []
for label, mult in [("+/-10% (the motivating test)", (0.9, 1.0, 1.1)),
                    ("+/-50%", (0.5, 1.0, 1.5))]:
    for slope in (0.0, -0.3, -0.6, -1.2):
        cfg = SyntheticConfig(elasticity_sd=0.0, elasticity_size_slope=0.0,
                              elasticity_price_slope=slope, seed=0)
        df = make_price_test_panel(panel, cfg, price_multipliers=mult, seed=0)
        wks = np.sort(df.week.unique())
        tr, va = df[df.week < wks[-20]], df[df.week >= wks[-20]]
        for leaves, nest in ((8, 150), (31, 300), (127, 600)):
            m = PriceLevelClassifier(K=3, level_col="arm", num_leaves=leaves,
                                     n_estimators=nest).fit(tr)
            e = np.asarray(m.elasticity(va), float)
            t = va.true_elasticity.to_numpy(float)
            rep = vus_report(m, va)
            rows.append({"arms": label, "price_slope": slope, "num_leaves": leaves,
                         "true_sd": float(t.std()), "vus": rep["vus_3afc"],
                         "ceiling": rep["vus_ceiling"], "vus_norm": rep["normalized"],
                         "oracle_corr": float(np.corrcoef(e, t)[0, 1])
                         if e.std() > 1e-9 and t.std() > 1e-9 else np.nan,
                         "oracle_bias": float(e.mean() - t.mean())})
            print(f"  {label:28s} slope={slope:+.1f} leaves={leaves:3d}  "
                  f"true_sd={t.std():.3f} vus={rep['vus_3afc']:.4f} "
                  f"norm={rep['normalized']:+.3f} corr={rows[-1]['oracle_corr']:+.3f}",
                  flush=True)

c = pd.DataFrame(rows)
c["abs_bias"] = c.oracle_bias.abs()
c.to_csv("experiments/out/17b_p3_learnable.csv", index=False)

print("\n" + "=" * 96)
print("P3, WITH LEARNABLE HETEROGENEITY")
print("=" * 96)
out = []
for label in c.arms.unique():
    g = c[c.arms == label].dropna(subset=["vus_norm", "oracle_corr"])
    out.append({"arms": label, "n": len(g),
                "vus_span": float(g.vus.max() - g.vus.min()),
                "rho(vus_norm, oracle_corr)":
                    g[["vus_norm", "oracle_corr"]].corr("spearman").iloc[0, 1],
                "rho(vus_norm, |bias|)":
                    g[["vus_norm", "abs_bias"]].corr("spearman").iloc[0, 1]})
print(pd.DataFrame(out).to_string(index=False, float_format=f4))
print("\nAlso: does the VUS itself rise with the amount of learnable heterogeneity?")
print(c.groupby(["arms", "price_slope"])[["true_sd", "vus", "vus_norm", "oracle_corr"]]
      .mean().to_string(float_format=f4))
