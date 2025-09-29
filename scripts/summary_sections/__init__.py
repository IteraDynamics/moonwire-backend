"""
Summary sections registry + compatibility helpers.

This module exposes:
- build_all(ctx) -> List[str]  : assembles all enabled sections in the right order
- Re-exports of section modules (market_context, calibration_reliability_trend, drift_response)
"""

from typing import List, Callable, Optional, Any

# Always import common types
from .common import SummaryContext

# Section modules (guarded imports keep older branches working)
try:
    from . import market_context  # must provide append(md, ctx)
except Exception:
    market_context = None  # type: ignore

try:
    from . import calibration_reliability_trend  # must provide append(md, ctx)
except Exception:
    calibration_reliability_trend = None  # type: ignore

try:
    from . import social_context_reddit
except Exception:
    social_context_reddit = None  # type: ignore

try:
    from . import drift_response  # NEW: automated governance
except Exception:
    drift_response = None  # type: ignore

# Optional sections that may exist in your repo
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
    "header_overview",
    "source_yield_plan",
):
    try:
        _mod = __import__(f"{__name__}.{_modname}", fromlist=["*"])
        OPTIONAL_SECTIONS.append(_mod)
    except Exception:
        pass


def _maybe_append(module: Any, md: List[str], ctx: SummaryContext, title: str) -> None:
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
    Build sections in recommended order.
    """
    md: List[str] = []

    # 1) Market context
    _maybe_append(market_context, md, ctx, "Market Context")

    # 2) Social context (Reddit)
    _maybe_append(social_context_reddit, md, ctx, "Social Context — Reddit")

    # 3) Calibration trend w/ regimes + social overlays
    _maybe_append(calibration_reliability_trend, md, ctx, "Calibration Trend vs Market + Social")

    # 4) Automated Drift Response (uses calibration+overlays)
    _maybe_append(drift_response, md, ctx, "Automated Drift Response")

    # 5) Any remaining optional sections
    for _mod in OPTIONAL_SECTIONS:
        _title = getattr(_mod, "__name__", "Section").split(".")[-1].replace("_", " ").title()
        _maybe_append(_mod, md, ctx, _title)

    return md


__all__ = [
    "SummaryContext",
    "build_all",
    "market_context",
    "social_context_reddit",
    "calibration_reliability_trend",
    "drift_response",
]
