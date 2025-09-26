"""
Summary sections registry + compatibility helpers.

This module exposes:
- build_all(ctx) -> List[str]  : assembles all enabled sections in the right order
- Re-exports of section modules for direct use
"""

from typing import List, Callable, Optional, Any

from .common import SummaryContext

# Import sections (guarded)
def _try_import(name: str) -> Any:
    try:
        return __import__(f"{__name__}.{name}", fromlist=["*"])
    except Exception:
        return None

market_context = _try_import("market_context")
social_context_reddit = _try_import("social_context_reddit")
calibration_reliability_trend = _try_import("calibration_reliability_trend")

OPTIONAL_SECTIONS: List[Any] = []
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
    m = _try_import(_modname)
    if m is not None:
        OPTIONAL_SECTIONS.append(m)


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
    md: List[str] = []
    # 1) Market Context (live)
    _maybe_append(market_context, md, ctx, "Market Context")
    # 2) Social (Reddit) — runs after market so it can reference timing if needed
    _maybe_append(social_context_reddit, md, ctx, "Social Context — Reddit")
    # 3) Calibration vs Market Regimes
    _maybe_append(calibration_reliability_trend, md, ctx, "Calibration Trend vs Market Regimes")
    # 4) Optional sections
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
]