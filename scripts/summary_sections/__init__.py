# scripts/summary_sections/__init__.py
from __future__ import annotations

"""
Summary sections entrypoint.

• No eager imports (prevents circulars).
• Detects which sections actually exist; only builds those.
• Keeps stable preferred order (same as yesterday’s merge),
  but silently skips missing modules instead of spamming errors.
• Exposes modules lazily so `from scripts.summary_sections import header_overview`
  still works when present.

New sections supported:
  - model_performance_trend
  - model_governance_actions
"""

from typing import List, Dict, Any
import importlib
import importlib.util

# Preferred order (we’ll filter this list down to modules that exist)
_PREFERRED_ORDER: List[str] = [
    "header_overview",
    "market_context",
    "social_reddit_context",
    "social_twitter_context",
    "cross_origin_correlation",
    "leadlag_analysis",
    "drift_response",
    "model_lineage",
    "model_performance_trend",
    "model_governance_actions",  # lives under scripts.governance
    "explainability",
    "signal_quality_summary",
    "thresholds",
    "source_yield_plan",
]

# Logical name -> module path
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
    "model_governance_actions": "scripts.governance.model_governance_actions",
    "explainability": "scripts.summary_sections.explainability",
    "signal_quality_summary": "scripts.summary_sections.signal_quality_summary",
    "thresholds": "scripts.summary_sections.thresholds",
    "source_yield_plan": "scripts.summary_sections.source_yield_plan",
}

__all__ = list(_SECTION_MODULES.keys()) + ["build_all", "available_sections"]


def _exists(mod_path: str) -> bool:
    return importlib.util.find_spec(mod_path) is not None


def available_sections() -> List[str]:
    """Return the sections that actually exist in this checkout, in preferred order."""
    out: List[str] = []
    for name in _PREFERRED_ORDER:
        mpath = _SECTION_MODULES.get(name)
        if mpath and _exists(mpath):
            out.append(name)
    return out


def _load(name: str):
    mpath = _SECTION_MODULES[name]
    return importlib.import_module(mpath)


def __getattr__(name: str):
    if name in _SECTION_MODULES and _exists(_SECTION_MODULES[name]):
        return _load(name)
    # keep normal AttributeError semantics
    raise AttributeError(name)


def build_all(ctx: Any) -> List[str]:
    """
    Build the markdown by running append(md, ctx) for each available section.
    We don’t print headers here—`header_overview` owns the top header.
    """
    md: List[str] = []
    for name in available_sections():
        try:
            mod = _load(name)
            mod.append(md, ctx)  # type: ignore[attr-defined]
        except Exception as e:
            # Single-line failure (matches existing CI style) but only
            # for sections that actually exist; non-existent ones are skipped.
            sec_mod = _SECTION_MODULES.get(name, f"scripts.summary_sections.{name}")
            md.append(f"❌ {sec_mod} failed: {e}")
    return md
