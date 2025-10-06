"""
Summary sections registry + compatibility helpers.

This module exposes:
- build_all(ctx) -> List[str]  : assembles all enabled sections in the right order
- Re-exports of section modules for direct use
"""

from typing import List, Callable, Optional, Any

# Always import common types
from .common import SummaryContext  # noqa: F401

# Core sections (import guarded so repo remains tolerant to partial checkouts)
def _try_import(modname: str):
    try:
        return __import__(f"{__name__}.{modname}", fromlist=["*"])
    except Exception:
        return None

market_context = _try_import("market_context")
# Social context sections
social_context_reddit = _try_import("social_context_reddit")
social_context_twitter = _try_import("social_context_twitter")

# Correlation (existing)
cross_origin_correlation = _try_import("cross_origin_correlation")
# NEW: Lead–Lag Analysis (v0.7.5)
cross_origin_analysis = _try_import("cross_origin_analysis")
# NEW: Influence Graph (v0.7.6)
influence_graph = _try_import("influence_graph")

calibration_reliability_trend = _try_import("calibration_reliability_trend")
drift_response = _try_import("drift_response")
retrain_automation = _try_import("retrain_automation")
trigger_explainability = _try_import("trigger_explainability")

# (Optional) Other sections that may exist in your repo. We import guarded so older
# pipelines keep running even if some sections are missing in this branch.
OPTIONAL_SECTIONS: list[Any] = []
for _modname in (
    "accuracy_by_version",
    "signal_quality",
    "signal_quality_per_origin",
    "signal_quality_per_version",
    "trigger_coverage_per_origin",
    "trigger_precision_by_origin",
    "suppression_rate_per_origin",
    "threshold_quality_per_origin",
    "threshold_recommendations",
    "threshold_backtest",
    "threshold_auto_apply",
    "header_overview",
    "source_yield_plan",
):
    _mod = _try_import(_modname)
    if _mod is not None:
        OPTIONAL_SECTIONS.append(_mod)


def _maybe_append(module: Any, md: List[str], ctx: SummaryContext, title: str) -> None:
    """
    Call module.append(md, ctx) if available.
    On failure, record an inline, human-friendly error in the markdown
    (don’t crash the entire summary).
    """
    if module is None:
        md.append(f"\n> ⚠️ Skippin
