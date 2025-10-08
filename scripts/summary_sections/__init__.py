# scripts/summary_sections/__init__.py
from __future__ import annotations

from typing import List
from scripts.summary_sections.common import SummaryContext

# Import sections we call; each must expose append(md, ctx)
# These imports are intentionally tolerant: failures are caught in build_all.
from . import model_lineage            # v0.7.7
from . import model_performance_trend  # v0.7.8
# Governance actions (new v0.7.9) lives under scripts/governance
from scripts.governance import model_governance_actions

# Optional sections (wrapped via try/except in build_all)
_OPTIONAL_SECTIONS = [
    ("scripts.summary_sections.header_overview", "header_overview"),
    ("scripts.summary_sections.market_context", "market_context"),
    ("scripts.summary_sections.social_reddit_context", "social_reddit_context"),
    ("scripts.summary_sections.social_twitter_context", "social_twitter_context"),
    ("scripts.summary_sections.cross_origin_correlation", "cross_origin_correlation"),
    ("scripts.summary_sections.leadlag_analysis", "leadlag_analysis"),
    ("scripts.summary_sections.drift_response", "drift_response"),
    ("scripts.summary_sections.thresholds", "thresholds"),
]


def _safe_append(mod, md: List[str], ctx: SummaryContext, qualname: str) -> None:
    try:
        mod.append(md, ctx)
    except Exception as e:
        md.append(f"❌ {qualname} failed: {e}")


def build_all(ctx: SummaryContext) -> List[str]:
    md: List[str] = []

    # Optional/context sections first
    for qual, name in _OPTIONAL_SECTIONS:
        try:
            mod = __import__(qual, fromlist=[name])
            _safe_append(mod, md, ctx, qual)
        except Exception as e:
            md.append(f"❌ {qual} failed: {e}")

    # Lineage (v0.7.7)
    _safe_append(model_lineage, md, ctx, "scripts.summary_sections.model_lineage")

    # Performance trend (v0.7.8)
    _safe_append(model_performance_trend, md, ctx, "scripts.summary_sections.model_performance_trend")

    # Governance actions (v0.7.9) AFTER trend
    _safe_append(model_governance_actions, md, ctx, "scripts.governance.model_governance_actions")

    return md
