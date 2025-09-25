# scripts/summary_sections/__init__.py
"""
Summary sections registry + compatibility helpers.

This module exposes:
- build_all(ctx) -> List[str]  : assembles all enabled sections in the right order
- Re-exports of section modules (market_context, social_context_reddit, calibration_reliability_trend) for direct use
"""

from typing import List, Callable, Optional, Any

# Always import common types
from .common import SummaryContext

# Section modules (import guarded so repo remains tolerant to partial checkouts)
try:
    from . import market_context  # must provide append(md, ctx)
except Exception as _e_mc:
    market_context = None  # type: ignore

try:
    from . import social_context_reddit  # NEW: Reddit Lite Ingest (append(md, ctx))
except Exception as _e_red:
    social_context_reddit = None  # type: ignore

try:
    from . import calibration_reliability_trend  # must provide append(md, ctx)
except Exception as _e_cal:
    calibration_reliability_trend = None  # type: ignore

# (Optional) Other sections that may exist in your repo. We import guarded so older
# pipelines keep running even if some sections are missing in this branch.
OPTIONAL_SECTIONS = []
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
):
    try:
        _mod = __import__(f"{__name__}.{_modname}", fromlist=["*"])
        OPTIONAL_SECTIONS.append(_mod)
    except Exception:
        # silently skip if not present in this branch
        pass


def _maybe_append(module: Any, md: List[str], ctx: SummaryContext, title: str) -> None:
    """
    Call module.append(md, ctx) if available.
    On failure, record an inline, human-friendly error in the markdown
    (don’t crash the entire summary).
    """
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
    """
    Build all sections in the recommended order.
    Returns a list of markdown lines (paragraphs) that can be joined with newlines.
    """
    md: List[str] = []

    # 1) Market Context first (this also ensures its artifacts exist for later sections)
    _maybe_append(market_context, md, ctx, "Market Context")

    # 2) Social Context — Reddit (new in v0.6.8), after market and before calibration
    _maybe_append(social_context_reddit, md, ctx, "Social Context — Reddit")

    # 3) Calibration trend with market regime overlays (v0.6.6+)
    _maybe_append(calibration_reliability_trend, md, ctx, "Calibration Trend vs Market Regimes")

    # 4) Any optional sections present in this repo
    for _mod in OPTIONAL_SECTIONS:
        # Use module name for a friendlier label
        _title = getattr(_mod, "__name__", "Section").split(".")[-1].replace("_", " ").title()
        _maybe_append(_mod, md, ctx, _title)

    return md


__all__ = [
    "SummaryContext",
    "build_all",
    "market_context",
    "social_context_reddit",
    "calibration_reliability_trend",
]