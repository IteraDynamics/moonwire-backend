"""
Summary section registry for MoonWire CI.
"""
from typing import List
from .common import SummaryContext

def _try(name: str):
    try:
        return __import__(f"{__name__}.{name}", fromlist=["*"])
    except Exception:
        return None

# Core sections (lazy imports; match main naming)
market_context = _try("market_context")
social_context_reddit = _try("social_context_reddit")
social_context_twitter = _try("social_context_twitter")
cross_origin_correlation = _try("cross_origin_correlation")
cross_origin_analysis = _try("cross_origin_analysis")
drift_response = _try("drift_response")
model_lineage = _try("model_lineage")
model_performance_trend = _try("model_performance_trend")
retrain_automation = _try("retrain_automation")
trigger_explainability = _try("trigger_explainability")

# Optional extras (exactly as in main)
OPTIONAL = []
for name in (
    "signal_quality_per_version", "threshold_quality_per_origin",
    "threshold_recommendations", "threshold_backtest", "threshold_auto_apply",
    "header_overview", "source_yield_plan"
):
    m = _try(name)
    if m:
        OPTIONAL.append(m)

# NEW: model governance (kept OUTSIDE summary_sections package;
# lazy import and fully optional; will not break if missing)
try:
    from scripts.governance import model_governance_actions as _governance
except Exception:
    _governance = None

def _maybe(mod, md, ctx, title):
    fn = getattr(mod, "append", None) if mod else None
    if not callable(fn):
        md.append(f"\n> ⚠️ Skipping **{title}**")
        return
    try:
        fn(md, ctx)
    except Exception as e:
        md.append(f"\n> ❌ **{title}** failed: {e}")

def build_all(ctx: SummaryContext) -> List[str]:
    md: List[str] = []
    _maybe(market_context, md, ctx, "Market Context")
    _maybe(social_context_reddit, md, ctx, "Social Context — Reddit")
    _maybe(social_context_twitter, md, ctx, "Social Context — Twitter")
    _maybe(cross_origin_correlation, md, ctx, "Cross-Origin Correlations")
    _maybe(cross_origin_analysis, md, ctx, "Lead–Lag Analysis")
    _maybe(drift_response, md, ctx, "Automated Drift Response")
    _maybe(model_lineage, md, ctx, "Model Lineage & Provenance")
    _maybe(model_performance_trend, md, ctx, "Model Performance Trends")

    # NEW: Governance section inserted exactly here, per task spec
    if _governance:
        _maybe(_governance, md, ctx, "Model Governance Actions")

    _maybe(retrain_automation, md, ctx, "Retrain Automation")
    _maybe(trigger_explainability, md, ctx, "Trigger Explainability")
    for m in OPTIONAL:
        _maybe(m, md, ctx, m.__name__)
    return md

__all__ = [
    "SummaryContext", "build_all", "model_performance_trend", "model_lineage"
]