# scripts/summary_sections/__init__.py
from __future__ import annotations

import importlib
from typing import List
from pathlib import Path

from .common import SummaryContext

# --- Public re-exports for tests ------------------------------------------------
# tests import these:  from scripts.summary_sections import header_overview, source_yield_plan
# We import lazily and tolerate absence (CI summary will still render with a ❌ line).
try:
    from . import header_overview as header_overview  # type: ignore
except Exception:  # pragma: no cover
    header_overview = None  # type: ignore

# Some repos name this module "source_yield_plan", others "yield_plan".
# Try both and fall back to a null shim if neither exists.
try:
    from . import source_yield_plan as source_yield_plan  # type: ignore
except Exception:  # pragma: no cover
    try:
        from . import yield_plan as source_yield_plan  # type: ignore
    except Exception:  # pragma: no cover
        class _NullYieldPlan:
            @staticmethod
            def append(md: List[str], ctx: SummaryContext) -> None:
                md.append("⚠️ Yield plan failed: module not available")
        source_yield_plan = _NullYieldPlan()  # type: ignore

__all__ = ["build_all", "header_overview", "source_yield_plan"]

# --- One true order of sections (called exactly once) ---------------------------
SECTION_ORDER: List[str] = [
    # Compact pipeline proof / header
    "header_overview",

    # Market + social
    "market_context",
    "social_reddit_context",
    "social_twitter_context",

    # Correlations & timing
    "cross_origin_correlation",
    "leadlag_analysis",

    # Governance & model modules
    "drift_response",
    "model_lineage",
    "model_performance_trend",      # Task 2
    "model_governance_actions",     # Task 3

    # Explainability
    "explainability",

    # Quality & thresholds
    "signal_quality_summary",
    "thresholds",

    # Sourcing / yield
    "source_yield_plan",            # prefer canonical name
]

def _import_section(mod_name: str):
    return importlib.import_module(f"scripts.summary_sections.{mod_name}")

def _append_section(md: List[str], ctx: SummaryContext, mod_name: str) -> None:
    try:
        mod = _import_section(mod_name)
    except Exception as e:
        md.append(f"❌ scripts.summary_sections.{mod_name} failed: {e}")
        return

    try:
        # All sections expose append(md, ctx); some accept extra kwargs but must tolerate being omitted.
        mod.append(md, ctx)  # type: ignore[attr-defined]
    except TypeError as e:
        # If a section still requires legacy kwargs, try the most common legacy call signature.
        # This keeps older modules working without breaking newer ones.
        try:
            mod.append(md, ctx, reviewers=[], threshold=0.5, sig_id="demo")  # type: ignore
        except Exception:
            md.append(f"❌ scripts.summary_sections.{mod_name} failed: {e}")
    except Exception as e:  # pragma: no cover
        md.append(f"❌ scripts.summary_sections.{mod_name} failed: {e}")

def build_all(ctx: SummaryContext) -> List[str]:
    """
    Build the entire CI summary in a stable order, with each block appended exactly once.
    Robust to missing modules: failures get summarized inline and the rest continues.
    """
    md: List[str] = []
    for name in SECTION_ORDER:
        _append_section(md, ctx, name)
    return md