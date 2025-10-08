# scripts/summary_sections/__init__.py
from __future__ import annotations

import importlib
import traceback
from datetime import datetime, timezone
from typing import Any, List, Tuple

from pathlib import Path
from .common import SummaryContext, ensure_dir

# Keep the original order from your working summary (Task 2 baseline).
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
    ("scripts.summary_sections.model_governance_actions", "Model Governance Actions"),
    ("scripts.summary_sections.trigger_explainability", "Trigger Explainability"),
    ("scripts.summary_sections.signal_quality", "Signal Quality"),
    ("scripts.summary_sections.thresholds", "Thresholds & Backtests"),
    ("scripts.summary_sections.source_yield_plan", "Source Yield Plan"),
]

def _append_error(md: List[str], module_path: str, err: BaseException) -> None:
    # Emit the same-style error marker users expect in CI
    msg = f"❌ {module_path} failed: {err}"
    md.append(msg)

def _run_section(md: List[str], ctx: SummaryContext, module_path: str) -> None:
    try:
        mod = importlib.import_module(module_path)
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

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines.append(f"Job summary generated at run-time")
    return lines
