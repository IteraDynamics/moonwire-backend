"""
Summary sections registry + compatibility helpers.

This module exposes:
- build_all(ctx) -> List[str]  : assembles all enabled sections in the right order
- Re-exports of section modules for direct use
"""

from typing import List, Callable, Optional, Any

# Always import common types
from .common import SummaryContext  # noqa: F401


def _try_import(modname: str):
    try:
        return __import__(f"{__name__}.{modname}", fromlist=["*"])
    except Exception:
        return None


# Core sections (import guarded so repo remains tolerant to partial checkouts)
market_context = _try_import("market_context")
# Social blocks (reddit/twitter) — guard-imported
social_context_reddit = _try_import("social_context_reddit")
social_context_twitter = _try_import("social_context_twitter")
# Correlation (new)
cross_origin_correlation = _try_import("cross_origin_correlation")
# Calibration & governance
calibration_reliability_trend = _try_import("calibration_reliability_trend")
drift_response = _try_import("drift_response")
retrain_automation = _try_import("retrain_automation")
# Explainability (optional)
trigger_explainability = _try_import("trigger_explainability")

# (Optional) Other sections that may exist in your repo.
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
    """Call module.append(md, ctx) if available; on failure, log inline error (don’t crash)."""
    if module is None:
        md.append(f"\n> ⚠️ Skipping **{title}** (module not available in this branch).\n")
        return
    fn: Optional[Callable[[List[str], SummaryContext], None]] = getattr(module, "append", None)
    if not callable(fn):
        md.append(f"\n> ⚠️ Skipping **{title}** (no `append(md, ctx)` function found).\n")
        return
    try:
        fn(md, ctx)
    except Exception as e:
        md.append(f"\n> ❌ **{title}** failed: `{type(e).__name__}: {e}`\n")


def build_all(ctx: SummaryContext) -> List[str]:
    """Build all sections in the recommended order."""
    md: List[str] = []

    # 1) Market Context first (ensures artifacts for later)
    _maybe_append(market_context, md, ctx, "Market Context")

    # 2) Social context blocks
    _maybe_append(social_context_reddit, md, ctx, "Social Context — Reddit")
    _maybe_append(social_context_twitter, md, ctx, "Social Context — Twitter")

    # 3) Cross-Origin Correlations (new, v0.7.4)
    _maybe_append(cross_origin_correlation, md, ctx, "Cross-Origin Correlations")

    # 4) Calibration trend with market + social overlays
    _maybe_append(calibration_reliability_trend, md, ctx, "Calibration Trend vs Market + Social")

    # 5) Governance
    _maybe_append(drift_response, md, ctx, "Automated Drift Response")
    _maybe_append(retrain_automation, md, ctx, "Retrain Automation")

    # 6) Explainability (lite)
    _maybe_append(trigger_explainability, md, ctx, "Trigger Explainability")

    # 7) Any optional sections present in this repo
    for _mod in OPTIONAL_SECTIONS:
        _title = getattr(_mod, "__name__", "Section").split(".")[-1].replace("_", " ").title()
        _maybe_append(_mod, md, ctx, _title)

    return md


__all__ = [
    "SummaryContext",
    "build_all",
    "market_context",
    "social_context_reddit",
    "social_context_twitter",
    "cross_origin_correlation",
    "calibration_reliability_trend",
    "drift_response",
    "retrain_automation",
    "trigger_explainability",
]
