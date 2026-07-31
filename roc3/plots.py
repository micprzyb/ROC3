"""
roc3.plots
==========

The 3-class ROC dashboard.

A single 3-D render is a poor decision tool -- you cannot read numbers off a rotated
surface.  The dashboard therefore pairs the surface with its *dual* (the weight simplex,
where you actually pick the rule) and with contour views that you can read values from.

Panels
------
A  3-D ROC surface: the monotone envelope ``Z(u,v) = max{S3 : S1>=u, S2>=v}``, whose
   volume literally is the VUS, with the chance plane for reference.
B  Ternary map of the weight simplex, filled with the chosen objective.  **This is where
   you read the thresholds off.**
C  Ternary map with iso-sensitivity contours for all three classes overlaid.
D  Topographic contour map of the same ``Z(u,v)`` -- the readable twin of panel A.
E  Iso-slices: the achievable ``(S1, S2)`` frontier subject to ``S3 >= z``.
F  Reference panel: one-vs-rest ROC curves plus the metric table (the table view).
G  Operating-point card: confusion matrix heat map and the deployable rule.

Colour follows ``docs`` conventions: the three classes always take categorical slots
1/2/3 (blue / orange / aqua, validated for CVD at all pairs in both modes); magnitude
uses a single-hue blue sequential ramp; ordered iso-levels use a 4-step ordinal ramp.
"""

from __future__ import annotations

import numpy as np

from .core import CHANCE_VUS, lattice_surface, monotone_envelope, roc_surface
from .thresholds import CRITERION_LABELS

__all__ = ["THEMES", "dashboard", "ordinal_dashboard", "plot_surface_3d",
           "plot_weight_simplex", "plot_weight_isolines", "plot_iso_slices",
           "plot_envelope_map", "plot_cutpoint_plane", "plot_operating_card",
           "plot_reference_roc", "interactive_surface"]


# --------------------------------------------------------------------------------------
# theme (light and dark are separately stepped, not an automatic flip)
# --------------------------------------------------------------------------------------
THEMES = {
    "light": dict(
        surface="#fcfcfb", page="#f9f9f7",
        ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
        seq=("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
        ordinal=("#86b6ef", "#5598e7", "#2a78d6", "#184f95"),
        accent="#e34948",
    ),
    "dark": dict(
        surface="#1a1a19", page="#0d0d0d",
        ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series=("#3987e5", "#d95926", "#199e70"),
        seq=("#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
             "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"),
        ordinal=("#184f95", "#2a78d6", "#5598e7", "#86b6ef"),
        accent="#e66767",
    ),
}


def _cmap(theme):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("roc3_seq", list(theme["seq"]), N=256)


def _style_axes(ax, theme, grid=True):
    ax.set_facecolor(theme["surface"])
    for s in ax.spines.values():
        s.set_color(theme["axis"])
        s.set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=3, width=0.8)
    ax.xaxis.label.set_color(theme["ink2"])
    ax.yaxis.label.set_color(theme["ink2"])
    ax.title.set_color(theme["ink"])
    if grid:
        ax.grid(True, color=theme["grid"], linewidth=0.6, linestyle="-", alpha=0.9)
        ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------
# barycentric helpers
# --------------------------------------------------------------------------------------
_SQ3 = np.sqrt(3.0) / 2.0


def _bary_xy(W):
    """Weights on the 2-simplex -> plane coordinates of an equilateral triangle."""
    W = np.asarray(W, dtype=float)
    return W[:, 1] + 0.5 * W[:, 2], _SQ3 * W[:, 2]


def _draw_ternary_frame(ax, theme, labels, title):
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQ3], [0.0, 0.0]])
    ax.plot(tri[:, 0], tri[:, 1], color=theme["axis"], linewidth=1.0, zorder=5)
    for frac in (0.25, 0.5, 0.75):                      # recessive barycentric grid
        w2 = np.array([[frac, 0.0], [(1 + frac) / 2, (1 - frac) * _SQ3]])
        w3 = np.array([[frac / 2, frac * _SQ3], [1 - frac / 2, frac * _SQ3]])
        w1 = np.array([[1 - frac, 0.0], [(1 - frac) / 2, (1 - frac) * _SQ3]])
        for seg in (w1, w2, w3):
            ax.plot(seg[:, 0], seg[:, 1], color=theme["grid"], linewidth=0.6, zorder=1)
    ax.text(-0.02, -0.04, f"{labels[0]}\n$w$=1", ha="center", va="top",
            fontsize=8, color=theme["ink2"])
    ax.text(1.02, -0.04, f"{labels[1]}\n$w$=1", ha="center", va="top",
            fontsize=8, color=theme["ink2"])
    ax.text(0.5, _SQ3 + 0.04, f"{labels[2]}  $w$=1", ha="center", va="bottom",
            fontsize=8, color=theme["ink2"])
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.16, _SQ3 + 0.14)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(theme["surface"])
    ax.set_title(title, fontsize=10, color=theme["ink"], pad=8)


# --------------------------------------------------------------------------------------
# individual panels
# --------------------------------------------------------------------------------------
def plot_surface_3d(ax, surf, theme, *, resolution=61, mark=None, show_chance=True):
    """Panel A: the ROC surface as the graph of the monotone envelope."""
    u, v, Z = monotone_envelope(surf.S, resolution=resolution)
    U, V = np.meshgrid(u, v, indexing="ij")
    cls = surf.classes

    ax.set_facecolor(theme["surface"])
    ax.plot_surface(U, V, Z, cmap=_cmap(theme), linewidth=0, antialiased=True,
                    rstride=1, cstride=1, vmin=0, vmax=1, shade=False)
    if show_chance:
        # The chance plane S1+S2+S3=1 meets the unit cube in the triangle e1-e2-e3.
        # Drawing its edges (rather than a translucent slab) avoids z-fighting with
        # the surface while still anchoring the eye.
        tri = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color=theme["muted"],
                linewidth=1.2, linestyle="-")
        ax.text(0.34, 0.34, 0.34, "chance", color=theme["muted"], fontsize=7.5)
    ax.scatter([1], [1], [1], color=theme["accent"], s=42, depthshade=False)
    ax.text(1, 1, 1.06, "perfect", color=theme["accent"], fontsize=8)
    if mark is not None:
        s = mark.sensitivities
        ax.scatter([s[0]], [s[1]], [s[2]], color=theme["ink"], s=44, marker="X",
                   depthshade=False)

    ax.set_xlabel(f"$S_1$ = Se[{cls[0]}]", fontsize=8.5, color=theme["ink2"], labelpad=6)
    ax.set_ylabel(f"$S_2$ = Se[{cls[1]}]", fontsize=8.5, color=theme["ink2"], labelpad=6)
    ax.set_zlabel(f"$S_3$ = Se[{cls[2]}]", fontsize=8.5, color=theme["ink2"], labelpad=4)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a._axinfo["grid"]["color"] = theme["grid"]
        a._axinfo["grid"]["linewidth"] = 0.5
    ax.tick_params(colors=theme["muted"], labelsize=7, pad=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.view_init(elev=24, azim=-128)
    ax.set_title(f"A · ROC surface   VUS = {surf.vus:.3f}\n"
                 f"volume under it (perfect 1.000 · chance {CHANCE_VUS:.3f})",
                 fontsize=10, color=theme["ink"], pad=12)


def plot_weight_simplex(ax, lat, theme, *, criterion="youden", mark=None,
                        priors=None, costs=None):
    """Panel B: objective over the weight simplex -- the panel you pick the rule from."""
    from .thresholds import objective_values

    obj = objective_values(lat.S, lat.rates, lat.counts, criterion,
                           priors=priors, costs=costs)
    x, y = _bary_xy(lat.W)
    cf = ax.tricontourf(x, y, obj, levels=14, cmap=_cmap(theme), zorder=2)
    ax.tricontour(x, y, obj, levels=14, colors=theme["surface"],
                  linewidths=0.35, zorder=3)
    _draw_ternary_frame(ax, theme, [str(c) for c in lat.classes],
                        f"B · where to set the weights\nfill: "
                        f"{CRITERION_LABELS.get(criterion, criterion)}")

    dx, dy = _bary_xy(np.full((1, 3), 1 / 3))
    ax.scatter(dx, dy, s=30, color=theme["ink"], zorder=7)
    ax.annotate("plain argmax", (dx[0], dy[0]), textcoords="offset points",
                xytext=(0, -16), ha="center", fontsize=8, color=theme["ink2"],
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc=theme["surface"],
                          ec="none", alpha=0.85))
    if mark is not None:
        mx, my = _bary_xy(mark.weights[None, :])
        ax.scatter(mx, my, s=110, marker="X", color=theme["ink"],
                   edgecolor=theme["surface"], linewidth=1.4, zorder=8)
        ax.annotate("selected", (mx[0], my[0]), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=8, color=theme["ink"],
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc=theme["surface"],
                              ec="none", alpha=0.85))
    return cf


def plot_weight_isolines(ax, lat, theme, *, mark=None, levels=(0.60, 0.80, 0.90)):
    """Panel C: iso-sensitivity contours for all three classes on the weight simplex."""
    from matplotlib.lines import Line2D

    x, y = _bary_xy(lat.W)
    handles = []
    for k in range(3):
        col = theme["series"][k]
        lv = [z for z in levels if lat.S[:, k].min() < z < lat.S[:, k].max()]
        if not lv:
            continue
        cs = ax.tricontour(x, y, lat.S[:, k], levels=lv, colors=[col],
                           linewidths=1.3, zorder=4)
        ax.clabel(cs, inline=True, inline_spacing=2, fontsize=6.5, fmt="%.2f",
                  colors=col)
        handles.append(Line2D([], [], color=col, linewidth=1.8,
                              label=f"Se[{lat.classes[k]}]"))
    _draw_ternary_frame(ax, theme, [str(c) for c in lat.classes],
                        "C · iso-sensitivity contours\n"
                        "each class's Se as a function of the weights")
    dx, dy = _bary_xy(np.full((1, 3), 1 / 3))
    ax.scatter(dx, dy, s=30, color=theme["ink"], zorder=7)
    if mark is not None:
        mx, my = _bary_xy(mark.weights[None, :])
        ax.scatter(mx, my, s=110, marker="X", color=theme["ink"],
                   edgecolor=theme["surface"], linewidth=1.4, zorder=8)
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True,
                    facecolor=theme["surface"], edgecolor=theme["grid"],
                    borderpad=0.4, handlelength=1.4)
    for t in leg.get_texts():
        t.set_color(theme["ink2"])


def plot_envelope_map(ax, surf, theme, *, resolution=201, mark=None):
    """Panel D: topographic contour map of ``Z(u,v)`` -- the readable twin of panel A."""
    u, v, Z = monotone_envelope(surf.S, resolution=resolution)
    cls = surf.classes
    cf = ax.contourf(u, v, Z.T, levels=np.linspace(0, 1, 11), cmap=_cmap(theme),
                     zorder=1)
    cs = ax.contour(u, v, Z.T, levels=[0.5, 0.7, 0.8, 0.9, 0.95],
                    colors=theme["surface"], linewidths=0.9, zorder=2)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f", colors=theme["surface"])
    ax.plot([0, 1], [1, 0], color=theme["muted"], linewidth=1.0, linestyle="-",
            zorder=3)
    ax.annotate("chance", (0.62, 0.42), fontsize=7, color=theme["muted"], zorder=3)
    if mark is not None:
        s = mark.sensitivities
        ax.scatter([s[0]], [s[1]], s=90, marker="X", color=theme["ink"],
                   edgecolor=theme["surface"], linewidth=1.2, zorder=5)
    _style_axes(ax, theme, grid=False)
    ax.set_xlabel(f"required $S_1$ = Se[{cls[0]}]", fontsize=9)
    ax.set_ylabel(f"required $S_2$ = Se[{cls[1]}]", fontsize=9)
    ax.set_title(f"D · best achievable Se[{cls[2]}] given the other two\n"
                 f"mean height = VUS = {surf.vus:.3f}",
                 fontsize=10, color=theme["ink"], pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    return cf


def plot_iso_slices(ax, surf, theme, *, hold=2, levels=(0.70, 0.80, 0.90, 0.95),
                    resolution=201):
    """Panel E: the ``(S_i, S_j)`` frontier subject to ``S_hold >= z``, for several z."""
    cls = surf.classes
    others = [k for k in range(3) if k != hold]
    S = surf.S
    from matplotlib.lines import Line2D

    handles = []
    for n, (z, col) in enumerate(zip(levels, theme["ordinal"])):
        keep = S[:, hold] >= z
        if not keep.any():
            continue
        pts = S[keep][:, others]
        # upper-right staircase of the retained points
        order = np.lexsort((-pts[:, 1], -pts[:, 0]))
        p = pts[order]
        best, front = -np.inf, []
        for xx, yy in p:
            if yy > best:
                front.append((xx, yy))
                best = yy
        front = np.array(front)
        xs = np.repeat(front[:, 0], 2)[:-1]
        ys = np.repeat(front[:, 1], 2)[1:]
        # close the staircase: drop to the axis on the right, run to the axis on the left
        xs = np.r_[front[0, 0], xs, 0.0]
        ys = np.r_[0.0, ys, ys[-1] if len(ys) else 0.0]
        ax.plot(xs, ys, color=col, linewidth=1.8, zorder=3 + n,
                solid_capstyle="round")
        handles.append(Line2D([], [], color=col, linewidth=2.0,
                              label=f"Se[{cls[hold]}] $\\geq$ {z:.2f}"))
    ax.plot([0, 1], [1, 0], color=theme["muted"], linewidth=1.0, zorder=2)
    _style_axes(ax, theme)
    ax.set_xlabel(f"$S$ = Se[{cls[others[0]]}]", fontsize=9)
    ax.set_ylabel(f"$S$ = Se[{cls[others[1]]}]", fontsize=9)
    ax.set_title(f"E · price of guaranteeing class '{cls[hold]}'\n"
                 f"frontier of the other two at each floor",
                 fontsize=10, color=theme["ink"], pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    if handles:
        leg = ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True,
                        facecolor=theme["surface"], edgecolor=theme["grid"])
        for t in leg.get_texts():
            t.set_color(theme["ink2"])


def plot_reference_roc(ax, surf, theme):
    """Panel F: the familiar one-vs-rest ROC curves, for calibration of intuition."""
    from matplotlib.lines import Line2D

    from .metrics import binary_auc, roc_curve_points

    logp, y = surf.log_scores, surf.y_idx
    handles = []
    for k in range(3):
        fpr, tpr = roc_curve_points(logp[y == k, k], logp[y != k, k])
        auc = binary_auc(logp[y == k, k], logp[y != k, k])
        ax.plot(fpr, tpr, color=theme["series"][k], linewidth=1.8, zorder=3 + k)
        handles.append(Line2D([], [], color=theme["series"][k], linewidth=2.0,
                              label=f"{surf.classes[k]} vs rest   AUC {auc:.3f}"))
    ax.plot([0, 1], [0, 1], color=theme["muted"], linewidth=1.0, zorder=2)
    _style_axes(ax, theme)
    ax.set_xlabel("false positive rate", fontsize=9)
    ax.set_ylabel("true positive rate", fontsize=9)
    ax.set_title("F · one-vs-rest ROC (reference only)\n"
                 "three thresholds here do NOT define a classifier",
                 fontsize=10, color=theme["ink"], pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    leg = ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True,
                    facecolor=theme["surface"], edgecolor=theme["grid"])
    for t in leg.get_texts():
        t.set_color(theme["ink2"])


def plot_operating_card(ax, op, theme, surf, extra_lines=()):
    """Panel G: confusion heat map + the deployable rule, as a readable table."""
    from .metrics import confusion_report

    cls = [str(c) for c in op.classes]
    cm = op.confusion
    rates = op.rates
    rep = confusion_report(cm, cls)

    ax.imshow(rates, cmap=_cmap(theme), vmin=0, vmax=1, zorder=1)
    for i in range(3):
        for j in range(3):
            v = rates[i, j]
            fg = theme["surface"] if v > 0.55 else theme["ink"]
            ax.text(j, i, f"{cm[i, j]:d}\n{v:.2f}", ha="center", va="center",
                    fontsize=9, color=fg, zorder=3)
    ax.set_xticks(range(3), [f"→{c}" for c in cls], fontsize=7,
                  rotation=28, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(3), [f"true {c}" for c in cls], fontsize=7.5)
    ax.tick_params(colors=theme["ink2"], length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(theme["surface"])
    title = (f"G · selected rule  ·  {op.criterion}"
             + (f"  ·  floors {op.constraints}" if op.constraints else ""))
    ax.set_title(title, fontsize=10, color=theme["ink"], pad=8)
    if hasattr(op, "thresholds"):                       # ordinal two-cut-point rule
        rule = [f"cut-points on the marker",
                f"    t1 = {op.thresholds[0]:.4g}",
                f"    t2 = {op.thresholds[1]:.4g}"]
    else:
        w = op.weights / op.weights.max()
        rule = [r"$\hat{y}(x)=\arg\max_k\; w_k\, p_k(x)$",
                "relative weights w",
                "    " + ",  ".join(f"{c}: {v:.2f}" for c, v in zip(cls, w))]
    lines = [
        *rule,
        "",
        "sensitivity  " + ",  ".join(f"{c}: {s:.3f}"
                                     for c, s in zip(cls, op.sensitivities)),
        "precision    " + ",  ".join(f"{c}: {s:.3f}"
                                     for c, s in zip(cls, rep["precision"])),
        "",
        f"accuracy      {rep['accuracy']:.3f}",
        f"bal. accuracy {rep['balanced_accuracy']:.3f}",
        f"macro-F1      {rep['macro_f1']:.3f}",
        *extra_lines,
    ]
    ax.text(0.0, -0.34, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", color=theme["ink2"], linespacing=1.55,
            family="monospace")


# --------------------------------------------------------------------------------------
# the dashboard
# --------------------------------------------------------------------------------------
def dashboard(
    y=None,
    scores=None,
    *,
    surface=None,
    classes=None,
    score_type="proba",
    criterion="youden",
    constraints=None,
    costs=None,
    priors=None,
    hold=2,
    resolution=200,
    lattice=96,
    theme="light",
    title=None,
    path=None,
    dpi=150,
):
    """Render the full 7-panel dashboard.

    Either pass ``y``/``scores`` or a precomputed ``surface``.  Returns
    ``(fig, {"surface":…, "lattice":…, "operating_point":…, "summary":…})``.
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    th = THEMES[theme] if isinstance(theme, str) else theme
    surf = surface if surface is not None else roc_surface(
        y, scores, classes=classes, score_type=score_type, resolution=resolution
    )
    lat = lattice_surface(surf, n=lattice)
    op = surf.select(criterion, constraints=constraints, costs=costs, priors=priors)

    from .metrics import format_summary, summary as _summary

    summ = _summary(surf.y_idx, surf.log_scores, classes=np.arange(3),
                    score_type="log", surface=surf)
    summ["classes"] = list(surf.classes)      # report the user's own labels

    fig = plt.figure(figsize=(18.0, 16.5), facecolor=th["page"])
    gs = GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.30,
                  left=0.045, right=0.97, top=0.90, bottom=0.03,
                  height_ratios=[1.0, 1.0, 0.95])

    axA = fig.add_subplot(gs[0, 0], projection="3d")
    plot_surface_3d(axA, surf, th, mark=op)
    axA.set_facecolor(th["page"])

    axD = fig.add_subplot(gs[0, 1])
    cf = plot_envelope_map(axD, surf, th, mark=op)
    cb = fig.colorbar(cf, ax=axD, fraction=0.046, pad=0.03)
    cb.set_label(f"Se[{surf.classes[2]}]", fontsize=8, color=th["ink2"])
    cb.ax.tick_params(colors=th["muted"], labelsize=7)
    cb.outline.set_edgecolor(th["axis"])

    axE = fig.add_subplot(gs[0, 2])
    plot_iso_slices(axE, surf, th, hold=hold)

    axB = fig.add_subplot(gs[1, 0])
    cf2 = plot_weight_simplex(axB, lat, th, criterion=criterion, mark=op,
                              priors=priors, costs=costs)
    cb2 = fig.colorbar(cf2, ax=axB, fraction=0.032, pad=0.0, shrink=0.75)
    cb2.ax.tick_params(colors=th["muted"], labelsize=7)
    cb2.outline.set_edgecolor(th["axis"])

    axC = fig.add_subplot(gs[1, 1])
    plot_weight_isolines(axC, lat, th, mark=op)

    axF = fig.add_subplot(gs[1, 2])
    plot_reference_roc(axF, surf, th)

    axG = fig.add_subplot(gs[2, 0])
    axG.set_position([0.055, 0.155, 0.125, 0.125])
    plot_operating_card(axG, op, th, surf)

    axT = fig.add_subplot(gs[2, 1:])
    axT.axis("off")
    axT.set_facecolor(th["surface"])
    axT.text(0.0, 1.06, format_summary(summ), transform=axT.transAxes,
             fontsize=8.6, family="monospace", va="top", ha="left", color=th["ink2"],
             linespacing=1.35)

    head = title or "Three-class ROC surface"
    fig.suptitle(head, fontsize=17, color=th["ink"], x=0.05, ha="left", y=0.975)
    fig.text(0.05, 0.945,
             f"VUS {surf.vus:.3f}  (chance {CHANCE_VUS:.3f}, perfect 1.000)   ·   "
             f"chance-corrected {surf.vus_adjusted:+.3f}   ·   "
             f"3AFC {summ['VUS_3AFC']:.3f}   ·   "
             f"every surface point is the rule  argmax_k  w_k p_k(x)",
             fontsize=10.5, color=th["ink2"], ha="left")

    if path is not None:
        fig.savefig(path, dpi=dpi, facecolor=th["page"], bbox_inches="tight")
    return fig, {"surface": surf, "lattice": lat, "operating_point": op, "summary": summ}


def plot_cutpoint_plane(ax, osurf, theme, *, criterion="youden", mark=None,
                        priors=None, costs=None, levels=(0.6, 0.8, 0.9)):
    """Ordinal twin of panel B: the objective over the ``(t1, t2)`` cut-point plane.

    The two axes are *literal cut-off values on the marker*, so the optimum can be read
    straight off and typed into production.
    """
    from matplotlib.lines import Line2D

    from .thresholds import objective_values

    obj = objective_values(osurf.S, osurf.rates, osurf.counts, criterion,
                           priors=priors, costs=costs)
    t1, t2 = osurf.T1, osurf.T2
    cf = ax.tricontourf(t1, t2, obj, levels=14, cmap=_cmap(theme), zorder=1)
    handles = []
    for k in range(3):
        col = theme["series"][k]
        lv = [z for z in levels if osurf.S[:, k].min() < z < osurf.S[:, k].max()]
        if not lv:
            continue
        cs = ax.tricontour(t1, t2, osurf.S[:, k], levels=lv, colors=[col],
                           linewidths=1.2, zorder=3)
        ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.2f", colors=col)
        handles.append(Line2D([], [], color=col, linewidth=1.8,
                              label=f"Se[{osurf.classes[k]}]"))
    if mark is not None:
        ax.scatter([mark.thresholds[0]], [mark.thresholds[1]], s=120, marker="X",
                   color=theme["ink"], edgecolor=theme["surface"], linewidth=1.4,
                   zorder=6)
        ax.annotate(f"$t_1$={mark.thresholds[0]:.3g}, $t_2$={mark.thresholds[1]:.3g}",
                    mark.thresholds, textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8.5, color=theme["ink"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=theme["surface"],
                              ec="none", alpha=0.85))
    _style_axes(ax, theme, grid=False)
    ax.set_xlabel("lower cut-point $t_1$", fontsize=9)
    ax.set_ylabel("upper cut-point $t_2$", fontsize=9)
    ax.set_title(f"B · where to put the two cut-offs\n"
                 f"fill: {CRITERION_LABELS.get(criterion, criterion)}",
                 fontsize=10, color=theme["ink"], pad=8)
    if handles:
        leg = ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True,
                        facecolor=theme["surface"], edgecolor=theme["grid"])
        for t in leg.get_texts():
            t.set_color(theme["ink2"])
    return cf


def ordinal_dashboard(
    y=None,
    marker=None,
    *,
    surface=None,
    classes=None,
    direction="increasing",
    criterion="youden",
    constraints=None,
    costs=None,
    priors=None,
    hold=2,
    resolution=300,
    theme="light",
    title=None,
    path=None,
    dpi=150,
):
    """Dashboard for the **ordered** three-class problem with a single marker.

    Panels: the ROC surface, the cut-point plane (where you read ``t1``/``t2`` off), the
    iso-slice frontiers, the topographic envelope map, and the operating-point card.
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from .metrics import confusion_report
    from .ordinal import ordinal_roc_surface, vus_ordinal

    th = THEMES[theme] if isinstance(theme, str) else theme
    osurf = surface if surface is not None else ordinal_roc_surface(
        y, marker, classes=classes, direction=direction, resolution=resolution
    )
    op = osurf.select(criterion, constraints=constraints, costs=costs, priors=priors)
    rank_vus = vus_ordinal(
        osurf.classes[osurf.y_idx],
        osurf.marker if direction == "increasing" else -osurf.marker,
        classes=osurf.classes, direction=direction,
    )

    fig = plt.figure(figsize=(17.5, 10.4), facecolor=th["page"])
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.28,
                  left=0.05, right=0.965, top=0.875, bottom=0.06)

    axA = fig.add_subplot(gs[0, 0], projection="3d")
    plot_surface_3d(axA, osurf, th)
    axA.scatter([op.sensitivities[0]], [op.sensitivities[1]], [op.sensitivities[2]],
                color=th["ink"], s=44, marker="X", depthshade=False)
    axA.set_facecolor(th["page"])

    axB = fig.add_subplot(gs[0, 1])
    cf = plot_cutpoint_plane(axB, osurf, th, criterion=criterion, mark=op,
                             priors=priors, costs=costs)
    cb = fig.colorbar(cf, ax=axB, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=th["muted"], labelsize=7)
    cb.outline.set_edgecolor(th["axis"])

    axC = fig.add_subplot(gs[0, 2])
    plot_iso_slices(axC, osurf, th, hold=hold)

    axD = fig.add_subplot(gs[1, 0])
    cf2 = plot_envelope_map(axD, osurf, th)
    cb2 = fig.colorbar(cf2, ax=axD, fraction=0.046, pad=0.03)
    cb2.ax.tick_params(colors=th["muted"], labelsize=7)
    cb2.outline.set_edgecolor(th["axis"])

    axE = fig.add_subplot(gs[1, 1])
    axE.set_position([0.40, 0.20, 0.11, 0.16])
    plot_operating_card(axE, op, th, osurf)

    axT = fig.add_subplot(gs[1, 2])
    axT.axis("off")
    cls = [str(c) for c in osurf.classes]
    rep = confusion_report(op.confusion, cls)
    t1, t2 = op.thresholds
    rule = (f"{cls[0]} if s < {t1:.4g}\n    {cls[1]} if {t1:.4g} <= s < {t2:.4g}"
            f"\n    {cls[2]} if s >= {t2:.4g}") if direction == "increasing" else (
        f"{cls[0]} if s > {t2:.4g}\n    {cls[1]} if {t1:.4g} < s <= {t2:.4g}"
        f"\n    {cls[2]} if s <= {t1:.4g}")
    txt = [
        "ORDERED 3-CLASS ROC (Mossman / Nakas)",
        "=" * 52,
        "classes (increasing): " + " < ".join(cls),
        "",
        f"VUS  geometric, two cut-points : {osurf.vus:.4f}",
        f"VUS  rank  P(s1 < s2 < s3)     : {rank_vus:.4f}",
        f"VUS  chance-corrected          : {osurf.vus_adjusted:+.4f}",
        f"     (perfect 1.000, chance {CHANCE_VUS:.3f})",
        "",
        f"selected rule ({criterion}):",
        "    " + rule,
        "",
        f"sensitivity : " + ",  ".join(f"{c} {s:.3f}"
                                       for c, s in zip(cls, op.sensitivities)),
        f"accuracy {rep['accuracy']:.3f}   bal.acc {rep['balanced_accuracy']:.3f}"
        f"   macro-F1 {rep['macro_f1']:.3f}",
    ]
    axT.text(0.0, 1.0, "\n".join(txt), transform=axT.transAxes, fontsize=9,
             family="monospace", va="top", ha="left", color=th["ink2"],
             linespacing=1.5)

    head = title or "Ordered three-class ROC surface (single marker, two cut-points)"
    fig.suptitle(head, fontsize=16, color=th["ink"], x=0.05, ha="left", y=0.965)
    fig.text(0.05, 0.925,
             f"VUS {osurf.vus:.3f}  ·  rank VUS  P(s₁<s₂<s₃) = {rank_vus:.3f}  ·  "
             f"chance {CHANCE_VUS:.3f}  ·  every surface point is a pair of cut-offs "
             f"(t₁, t₂)",
             fontsize=10.5, color=th["ink2"], ha="left")

    if path is not None:
        fig.savefig(path, dpi=dpi, facecolor=th["page"], bbox_inches="tight")
    return fig, {"surface": osurf, "operating_point": op, "rank_vus": rank_vus}


# --------------------------------------------------------------------------------------
# interactive
# --------------------------------------------------------------------------------------
def interactive_surface(surf, *, path="roc3_surface.html", theme="light",
                        resolution=101, lattice=72, title="3-class ROC surface"):
    """Self-contained plotly HTML: hover any point to read ``w`` and the operating point.

    The static dashboard is the deliverable; this is the exploration tool.  Hovering the
    parametric cloud shows the weight vector and the full sensitivity triple, so a
    reviewer can find a rule by pointing at the trade-off they want.
    """
    import plotly.graph_objects as go

    th = THEMES[theme] if isinstance(theme, str) else theme
    cls = [str(c) for c in surf.classes]
    u, v, Z = monotone_envelope(surf.S, resolution=resolution)
    lat = lattice_surface(surf, n=lattice)

    scale = [[i / (len(th["seq"]) - 1), c] for i, c in enumerate(th["seq"])]
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=u, y=v, z=Z.T, colorscale=scale, cmin=0, cmax=1, opacity=0.92,
        name="achievable envelope", showscale=True,
        colorbar=dict(title=f"Se[{cls[2]}]", len=0.6),
        hovertemplate=(f"required Se[{cls[0]}] %{{x:.3f}}<br>"
                       f"required Se[{cls[1]}] %{{y:.3f}}<br>"
                       f"best Se[{cls[2]}] %{{z:.3f}}<extra></extra>"),
    ))
    cu = np.linspace(0, 1, 2)
    CU, CV = np.meshgrid(cu, cu, indexing="ij")
    fig.add_trace(go.Surface(
        x=cu, y=cu, z=np.clip(1 - CU - CV, 0, 1).T, opacity=0.14, showscale=False,
        colorscale=[[0, th["muted"]], [1, th["muted"]]], name="chance plane",
        hoverinfo="skip",
    ))
    txt = [f"w = ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})" for w in lat.W]
    fig.add_trace(go.Scatter3d(
        x=lat.S[:, 0], y=lat.S[:, 1], z=lat.S[:, 2], mode="markers",
        marker=dict(size=2.0, color=th["series"][1], opacity=0.55),
        name="attainable rules", text=txt,
        hovertemplate=("%{text}<br>"
                       f"Se[{cls[0]}] %{{x:.3f}} · Se[{cls[1]}] %{{y:.3f}} · "
                       f"Se[{cls[2]}] %{{z:.3f}}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[1], y=[1], z=[1], mode="markers+text", text=["perfect"],
        textposition="top center", marker=dict(size=6, color=th["accent"]),
        name="perfect", hoverinfo="skip",
    ))
    fig.update_layout(
        title=dict(text=f"{title} — VUS {surf.vus:.3f} "
                        f"(chance {CHANCE_VUS:.3f}, perfect 1.000)"),
        scene=dict(
            xaxis=dict(title=f"S₁ Se[{cls[0]}]", range=[0, 1], backgroundcolor=th["surface"]),
            yaxis=dict(title=f"S₂ Se[{cls[1]}]", range=[0, 1], backgroundcolor=th["surface"]),
            zaxis=dict(title=f"S₃ Se[{cls[2]}]", range=[0, 1], backgroundcolor=th["surface"]),
            aspectmode="cube",
        ),
        paper_bgcolor=th["page"], font=dict(color=th["ink"]),
        margin=dict(l=0, r=0, t=60, b=0), height=760,
    )
    if path:
        fig.write_html(path, include_plotlyjs="inline", full_html=True)
    return fig
