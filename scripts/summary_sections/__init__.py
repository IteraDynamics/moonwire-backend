# scripts/summary_sections/__init__.py
from __future__ import annotations

"""
Package entrypoint for CI summary sections.

This file intentionally:
  1) Re-exports ALL existing sections exactly as main had them
     so imports like `from scripts.summary_sections import header_overview`
     continue to work.
  2) Adds ONLY two new blocks (model_performance_trend, model_governance_actions)
     without changing any other section behavior.
  3) Provides build_all(ctx) that calls each section's append(md, ctx) in order,
     catching exceptions and emitting a single ❌ line per failure
     (matching yesterday's formatting).
  4) Avoids ANY header text. The header line is produced by header_overview only.
"""

from typing import List, Any, Dict, Callable

# --- Re-export all existing sections (must match your main branch names) ---
from . import header_overview
from . import market_context
from . import social_reddit_context
from . import social_twitter_context
from . import cross_origin_correlation
from . import leadlag_analysis
from . import drift_response
from . import model_lineage
from . import explainability
from . import signal_quality_summary
from . import thresholds
from . import source_yield_plan

# New in v0.7.8 (already added yesterday)
from . import model_performance_trend  # local module in summary_sections

# New in v0.7.9 (lives under scripts.governance but re-exported here)
# so downstream code can treat it like a "summary section".
from scripts.governance import model_governance_actions as model_governance_actions  # type: ignore


# --- Section order (restored to yesterday’s proven order) ---
# NOTE: Do not add any header text here. `header_overview` owns the top block.
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
    "model_governance_actions",  # new (re-exported)
    "explainability",
    "signal_quality_summary",
    "thresholds",
    "source_yield_plan",
]


def _resolve_sections() -> Dict[str, Any]:
    """
    Build a name->module map from objects we imported above.
    Keeping this explicit avoids importlib/namespace surprises.
    """
    return {
        "header_overview": header_overview,
        "market_context": market_context,
        "social_reddit_context": social_reddit_context,
        "social_twitter_context": social_twitter_context,
        "cross_origin_correlation": cross_origin_correlation,
        "leadlag_analysis": leadlag_analysis,
        "drift_response": drift_response,
        "model_lineage": model_lineage,
        "model_performance_trend": model_performance_trend,
        "model_governance_actions": model_governance_actions,
        "explainability": explainability,
        "signal_quality_summary": signal_quality_summary,
        "thresholds": thresholds,
        "source_yield_plan": source_yield_plan,
    }


def _safe_append(mod: Any) -> Callable[[List[str], Any], None]:
    """
    Return a wrapper that calls mod.append(md, ctx) and forwards exceptions
    as a single ❌ line, matching existing CI formatting.
    """
    def _call(md: List[str], ctx: Any) -> None:
        try:
            # All sections share the same signature append(md, ctx)
            mod.append(md, ctx)  # type: ignore[attr-defined]
        except Exception as e:
            md.append(f"❌ scripts.summary_sections.{mod.__name__.split('.')[-1]} failed: {e}")
    return _call


def build_all(ctx: Any) -> List[str]:
    """
    Build the entire markdown by running each section in SECTION_ORDER.
    No headers are injected here — header_overview is responsible for the top block.
    """
    md: List[str] = []
    registry = _resolve_sections()

    for name in SECTION_ORDER:
        mod = registry.get(name)
        if mod is None:
            # Module truly missing: keep yesterday's failure style
            md.append(f"❌ scripts.summary_sections.{name} failed: No module named 'scripts.summary_sections.{name}'")
            continue

        _safe_append(mod)(md, ctx)

    return md