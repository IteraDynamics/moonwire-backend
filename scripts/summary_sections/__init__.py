"""
Summary sections registry + compatibility helpers.

This module exposes:
- build_all(ctx) -> List[str]  : assembles all enabled sections in the right order
- Re-exports of section modules (market_context, calibration_reliability_trend, drift_response) for direct use
"""

from typing import List, Callable, Optional, Any

# Always import common types
from .common import SummaryContext

# --- Core sections (import guarded so repo remains tolerant to partial checkouts) ---

try:
    from . import market_context  # must provide append(md, ctx)
except Exception:
    market_context = None  # type: ignore

try:
    from . import calibration_reliability_trend  # must provide append(md, ctx)
except Exception:
    calibration_reliability_trend = None  # type: ignore

# NEW: Automated Drift Response (v0.6.10+)
try:
    from . import drift_response  # must provide append(md, ctx)
except Exception:
    drift_response = None  # type: ignore

# (Optional) Other sections that may exist in your repo. We import guarded so older
# pipelines keep running even if some sections are missing in this branch.
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

    # 1) Market Context first (ensures artifacts exist for later sections)
    _maybe_append(market_context, md, ctx, "Market Context")

    # 2) Calibration trend with market & social overlays
    _maybe_append(calibration_reliability_trend, md, ctx, "Calibration Trend vs Market + Social")

    # 3) Automated Drift Response (acts on calibration artifacts)
    _maybe_append(drift_response, md, ctx, "Automated Drift Response")

    # 4) Any optional sections present in this repo
    for _mod in OPTIONAL_SECTIONS:
        _title = getattr(_mod, "__name__", "Section").split(".")[-1].replace("_", " ").title()
        _maybe_append(_mod, md, ctx, _title)

    return md


__all__ = [
    "SummaryContext",
    "build_all",
    "market_context",
    "calibration_reliability_trend",
    "drift_response",
]
