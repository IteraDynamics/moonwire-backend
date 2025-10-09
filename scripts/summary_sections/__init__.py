# scripts/summary_sections/__init__.py
from __future__ import annotations

"""
Package entrypoint for CI summary sections.

Fixes:
  • Removes eager top-level imports that caused circular imports in tests.
  • Uses lazy module loading via importlib when sections are accessed or built.
  • Restores yesterday's section order and error formatting.
  • Adds the two new blocks (model_performance_trend, model_governance_actions).

This module intentionally does NOT emit any header text. The header block is
owned by the 'header_overview' section itself.
"""

from typing import List, Dict, Any, Callable
import importlib

# --- Section order (restored) ---
SECTION_ORDER: List[str] = [
    "header_overview",
    "market_context",
    "social_reddit_context",
    "social_twitter_context",
    "cross_origin_correlation",
    "leadlag_analysis",
    "drift_response",
    "model_lineage",
    "model_performance_trend",   # new
    "model_governance_actions",  # new
    "explainability",
    "signal_quality_summary",
    "thresholds",
    "source_yield_plan",
]

# Map logical section name -> fully qualified module path
_SECTION_MODULES: Dict[str, str] = {
    "header_overview": "scripts.summary_sections.header_overview",
    "market_context": "scripts.summary_sections.market_context",
    "social_reddit_context": "scripts.summary_sections.social_reddit_context",
    "social_twitter_context": "scripts.summary_sections.social_twitter_context",
    "cross_origin_correlation": "scripts.summary_sections.cross_origin_correlation",
    "leadlag_analysis": "scripts.summary_sections.leadlag_analysis",
    "drift_response": "scripts.summary_sections.drift_response",
    "model_lineage": "scripts.summary_sections.model_lineage",
    "model_performance_trend": "scripts.summary_sections.model_performance_trend",
    # lives under scripts.governance but exposed here as a section:
    "model_governance_actions": "scripts.governance.model_governance_actions",
    "explainability": "scripts.summary_sections.explainability",
    "signal_quality_summary": "scripts.summary_sections.signal_quality_summary",
    "thresholds": "scripts.summary_sections.thresholds",
    "source_yield_plan": "scripts.summary_sections.source_yield_plan",
}

__all__ = list(_SECTION_MODULES.keys()) + ["build_all"]


def _load_section(name: str):
    """Lazy import a section module by logical name."""
    mod_path = _SECTION_MODULES.get(name)
    if not mod_path:
        raise ModuleNotFoundError(f"No module named 'scripts.summary_sections.{name}'")
    return importlib.import_module(mod_path)


def __getattr__(name: str):
    """
    Allow `from scripts.summary_sections import header_overview` to work
    without eager imports. This is called only on demand.
    """
    if name in _SECTION_MODULES:
        return _load_section(name)
    raise AttributeError(name)


def _safe_append(mod: Any) -> Callable[[List[str], Any], None]:
    """
    Wrap mod.append(md, ctx) and on error emit a single ❌ line
    (matching existing CI formatting).
    """
    def _call(md: List[str], ctx: Any) -> None:
        try:
            mod.append(md, ctx)  # type: ignore[attr-defined]
        except Exception as e:
            sec_name = getattr(mod, "__name__", str(mod)).split(".")[-1]
            md.append(f"❌ scripts.summary_sections.{sec_name} failed: {e}")
    return _call


def build_all(ctx: Any) -> List[str]:
    """
    Build the entire markdown by running each section in SECTION_ORDER.
    No headers are injected here — header_overview is responsible for the top block.
    """
    md: List[str] = []

    for name in SECTION_ORDER:
        try:
            mod = _load_section(name)
        except Exception as e:
            # If truly missing, keep yesterday's failure style
            md.append(f"❌ scripts.summary_sections.{name} failed: {e}")
            continue

        _safe_append(mod)(md, ctx)

    return md
