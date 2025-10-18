# scripts/summary_sections/__init__.py
"""
Summary section registry for MoonWire CI.

This registry collects summary sections and invokes their `append(md, ctx)` (or
equivalent) in a stable order. It also wires in governance modules that live
outside this package so they render via the same registry pathway.
"""
from typing import List, Any
from .common import SummaryContext

def _try(name: str):
    try:
        return __import__(f"{__name__}.{name}", fromlist=["*"])
    except Exception:
        return None

# Sections that live under scripts.summary_sections.*
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

# Optional sections (best-effort)
OPTIONAL = []
for name in (
    "signal_quality_per_version",
    "threshold_quality_per_origin",
    "threshold_recommendations",
    "threshold_backtest",
    "threshold_auto_apply",
    "header_overview",
    "source_yield_plan",
    # v0.8.3 notifications section lives here (shim UI block)
    "governance_notifications_section",
    # v0.9.0 performance validation section (compact CI block)
    "performance_validation",
    "ml_validation_tuning",
):
    m = _try(name)
    if m:
        OPTIONAL.append(m)

# --- Governance: modules that live outside summary_sections -------------------
# We import them directly and wrap with a small adapter so they participate in the registry.

# v0.8.0 apply (already present before, kept here for consistency)
try:
    from scripts.governance.governance_apply import append as _gov_apply_append
except Exception:  # pragma: no cover
    _gov_apply_append = None

# v0.8.1 blue-green simulation (module: scripts/governance/bluegreen_promotion.py)
try:
    from scripts.governance.bluegreen_promotion import append as _bg_append
except Exception:  # pragma: no cover
    _bg_append = None

# v0.8.2 governance alerts (module: scripts/governance/governance_alerts.py)
try:
    from scripts.governance.governance_alerts import run_alerts as _alerts_run
except Exception:  # pragma: no cover
    _alerts_run = None


def _maybe(mod: Any, md: List[str], ctx: SummaryContext, title: str) -> None:
    """Call mod.append(md, ctx) if present; otherwise emit a skip line."""
    fn = getattr(mod, "append", None)
    if not callable(fn):
        md.append(f"\n> ⚠️ Skipping {title}")
        return
    try:
        fn(md, ctx)
    except Exception as e:  # be resilient in CI
        md.append(f"\n> ❌ {title} failed: {e}")


def _maybe_call(fn, md: List[str], ctx: SummaryContext, title: str) -> None:
    """Call a function(md, ctx) if available; else emit a skip line."""
    if fn is None:
        md.append(f"\n> ⚠️ Skipping {title}")
        return
    try:
        fn(md, ctx)
    except Exception as e:
        md.append(f"\n> ❌ {title} failed: {e}")


def _maybe_run(fn, ctx: SummaryContext, md: List[str], title: str) -> None:
    """Call a function(ctx) if available; else emit a skip line."""
    if fn is None:
        md.append(f"\n> ⚠️ Skipping {title}")
        return
    try:
        fn(ctx)
    except Exception as e:
        md.append(f"\n> ❌ {title} failed: {e}")


def build_all(ctx: SummaryContext) -> List[str]:
    md: List[str] = []

    # Core sections
    _maybe(market_context, md, ctx, "Market Context")
    _maybe(social_context_reddit, md, ctx, "Social Context — Reddit")
    _maybe(social_context_twitter, md, ctx, "Social Context — Twitter")
    _maybe(cross_origin_correlation, md, ctx, "Cross-Origin Correlations")
    _maybe(cross_origin_analysis, md, ctx, "Lead–Lag Analysis")
    _maybe(drift_response, md, ctx, "Automated Drift Response")
    _maybe(model_lineage, md, ctx, "Model Lineage & Provenance")
    _maybe(model_performance_trend, md, ctx, "Model Performance Trends")

    # Governance Apply (v0.8.0)
    _maybe_call(_gov_apply_append, md, ctx, "Governance Apply")

    # Blue-Green Promotion Simulation (v0.8.1)
    _maybe_call(_bg_append, md, ctx, "Blue-Green Promotion Simulation")

    # Governance Alerts (v0.8.2) — runs side-effecting function, then UI section (if present)
    _maybe_run(_alerts_run, ctx, md, "Governance Alerts")

    # Optional/derived sections (includes v0.8.3 notifications and v0.9.0 performance)
    for m in OPTIONAL:
        nice = m.__name__.split(".")[-1].replace("_", " ").title()  # <- basename for title
        _maybe(m, md, ctx, nice)

    # Retrain & Trigger Explainability at the end (kept after governance surfaces)
    _maybe(retrain_automation, md, ctx, "Retrain Automation")
    _maybe(trigger_explainability, md, ctx, "Trigger Explainability")

    return md


__all__ = [
    "SummaryContext",
    "build_all",
    "model_performance_trend",
    "model_lineage",
]
