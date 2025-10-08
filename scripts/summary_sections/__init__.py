# scripts/summary_sections/__init__.py
from __future__ import annotations

from importlib import import_module
from typing import Callable, List, Tuple

from .common import SummaryContext

# Helper to import a section defensively
def _load(section_name: str) -> Callable[[List[str], SummaryContext], None] | None:
    """
    Try importing 'scripts.summary_sections.<section_name>.append'.
    Returns the callable or None if import fails.
    """
    try:
        mod = import_module(f"scripts.summary_sections.{section_name}")
        return getattr(mod, "append", None)
    except Exception:
        return None


# Ordered sections. Keep 'header_overview' first; it emits the single H1.
_SECTION_NAMES: Tuple[str, ...] = (
    "header_overview",
    "market_context",
    "cross_origin_correlation",
    "leadlag_analysis",
    "drift_response",
    "model_lineage",
    "model_performance_trend",
    "model_governance_actions",
    "explainability",
    "signal_quality",
    "thresholds",
    "yield_plan",
)


def build_all(ctx: SummaryContext) -> List[str]:
    """
    Iterate sections and build the full CI markdown.
    This function DOES NOT inject any standalone title/header; `header_overview`
    is the single source of truth for the header block.
    """
    md: List[str] = []
    for name in _SECTION_NAMES:
        appender = _load(name)
        if appender is None:
            # emit a compact failure line and continue
            md.append(f"❌ scripts.summary_sections.{name} failed: No module named 'scripts.summary_sections.{name}'")
            continue
        try:
            appender(md, ctx)  # every section must append to md
        except Exception as e:
            md.append(f"❌ scripts.summary_sections.{name} failed: {e}")
    return md


# Re-export appenders for tests that import submodules directly
# (No-op if a given submodule is absent.)
try:
    from . import model_lineage  # noqa: F401
except Exception:
    pass
try:
    from . import model_performance_trend  # noqa: F401
except Exception:
    pass
try:
    from . import model_governance_actions  # noqa: F401
except Exception:
    pass