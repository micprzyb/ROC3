"""
roc3 -- a 3-dimensional analogue of the ROC curve and AUC for 3-class classifiers.

Quick start
-----------
>>> from roc3 import roc_surface, summary, format_summary, dashboard
>>> surf = roc_surface(y, proba)                    # the 2-D surface in 3-D
>>> surf.vus                                        # perfect 1.0, chance 1/6
>>> op = surf.select("maximin", constraints={2: 0.90})
>>> print(op.describe())                            # the deployable rule
>>> dashboard(y, proba, path="roc3.png")            # the 6-panel figure

What the objects mean
---------------------
Every point of the surface is a rule ``y_hat(x) = argmax_k w_k * p_k(x)`` for some weight
vector ``w`` on the 2-simplex -- the exact 3-class generalisation of "move the threshold".
Its coordinates are the three per-class sensitivities (true class fractions).  The volume
of the region under the surface is the VUS: the probability that a randomly demanded
triple of sensitivity requirements is achievable, and equivalently the probability of
correctly sorting a trio of cases drawn one from each class.

See ``docs/PLAN.md`` for the derivation and ``docs/IDEAS_LOG.md`` for what was tried.
"""

from .constrained import (
    check_order_constraint,
    constrained_report,
    format_constrained_report,
    gauge_normalize,
    normalized_vus,
    to_ordered_gauge,
    vus_ceiling,
)
from .core import (
    CHANCE_VUS,
    OperatingSurface,
    hypervolume3d,
    monotone_envelope,
    prepare_scores,
    roc_surface,
)
from .metrics import (
    binary_auc,
    confusion_report,
    format_summary,
    hand_till_m,
    ovr_auc,
    summary,
)
from .ordinal import OrdinalSurface, ordinal_roc_surface, vus_ordinal
from .thresholds import CRITERIA, OperatingPoint, achievable, select_operating_point
from .vus import ForcedChoiceResult, forced_choice_profile, hum, vus_forced_choice

__version__ = "0.1.0"

__all__ = [
    "CHANCE_VUS",
    "CRITERIA",
    "ForcedChoiceResult",
    "OperatingPoint",
    "OperatingSurface",
    "OrdinalSurface",
    "achievable",
    "binary_auc",
    "check_order_constraint",
    "confusion_report",
    "constrained_report",
    "forced_choice_profile",
    "format_constrained_report",
    "format_summary",
    "gauge_normalize",
    "normalized_vus",
    "to_ordered_gauge",
    "vus_ceiling",
    "hand_till_m",
    "hum",
    "hypervolume3d",
    "monotone_envelope",
    "ordinal_roc_surface",
    "ovr_auc",
    "prepare_scores",
    "roc_surface",
    "select_operating_point",
    "summary",
    "vus_forced_choice",
    "vus_ordinal",
    "__version__",
]


def dashboard(*args, **kwargs):
    """Lazy re-export of :func:`roc3.plots.dashboard` (imports matplotlib on demand)."""
    from .plots import dashboard as _d

    return _d(*args, **kwargs)


def interactive_surface(*args, **kwargs):
    """Lazy re-export of :func:`roc3.plots.interactive_surface` (needs plotly)."""
    from .plots import interactive_surface as _i

    return _i(*args, **kwargs)
