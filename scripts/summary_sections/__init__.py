# scripts/summary_sections/__init__.py
from __future__ import annotations

import importlib
from datetime import datetime, timezone
from typing import Any, List, Tuple
from pathlib import Path

from .common import SummaryContext, ensure_dir

# ——————————————————————————————————————————
# Section order restored to Task 2 baseline,
# with corrected path for model_governance_actions.
# ——————————————————————————————————————————
SECTION_MODULES: List[Tuple[str, str]] = [
    ("scripts.summary_sections.header_overview", "Header Overview"),
    ("scripts.summary_sections.market_context", "Market Context"),
    ("scripts.summary_sections.social_reddit_context", "Social Context — Reddit"),
    ("scripts.summary_sections.social_twitter_context", "Social Context — Twitter"),
    ("scripts.summary_sections.cross_origin_correlation", "Cross-Origin Correlations"),
    ("scripts.summary_sections.leadlag_analysis", "Lead–Lag Analysis"),
    ("scripts.summary_sections.drift_response", "Automated Drift Response"),
    ("scripts.summary_sections.model_lineage", "Model Lineage & Provenance"),
    ("scripts.summary_sections.model_performance_trend", "Model Performance Trends"),
    ("scripts.governance.model_governance_actions", "Model Governance Actions"),  # ← fixed path
    ("scripts.summary_sections.trigger_explainability", "Trigger Explainability"),
    ("scripts.summary_sections.signal_quality", "Signal Quality"),
    ("scripts.summary_sections.thresholds", "Thresholds & Backtests"),
    ("scripts.summary_sections.source_yield_plan", "Source Yield Plan"),
]

def _env_bool(key: str, default: bool = False) -> bool:
    import os
    return str(os.getenv(key, str(default))).lower() in ("1", "true", "yes", "on")

def _append_error(md: List[str], module_path: str, err: BaseException) -> None:
    md.append(f"❌ {module_path} failed: {err}")

def _run_section(md: List[str], ctx: SummaryContext, module_path: str) -> None:
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        _append_error(md, module_path, e)
        return

    # header_overview has a different signature in your repo; pass safe demo defaults
    if module_path.endswith(".header_overview"):
        try:
            reviewers: List[dict] = []
            threshold: float = 0.50
            sig_id: str = "demo"
            # If demo seeding helper is available (from mw_demo_summary), use it to populate reviewers
            try:
                from scripts.mw_demo_summary import generate_demo_data_if_needed
                reviewers, _events = generate_demo_data_if_needed(reviewers)
            except Exception:
                pass
            mod.append(md, ctx, reviewers=reviewers, threshold=threshold, sig_id=sig_id)  # type: ignore
        except Exception as e:
            _append_error(md, module_path, e)
        return

    if not hasattr(mod, "append"):
        _append_error(md, module_path, RuntimeError("module has no append(md, ctx)"))
        return

    try:
        mod.append(md, ctx)  # type: ignore[attr-defined]
    except Exception as e:
        _append_error(md, module_path, e)

def build_all(ctx: SummaryContext) -> List[str]:
    ensure_dir(Path(ctx.artifacts_dir))
    lines: List[str] = []
    for module_path, _title in SECTION_MODULES:
        _run_section(lines, ctx, module_path)

    # single footer (mw_demo_summary handles the single header)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines.append("Job summary generated at run-time")
    return lines
