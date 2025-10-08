# scripts/summary_sections/__init__.py
from __future__ import annotations

import importlib
import os
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple
from pathlib import Path

from .common import SummaryContext, ensure_dir

# ---------------------------------------------------------------------------
# Section order: Task 2 baseline + Task 3 governance actions.
# For some sections we specify fallback module names to avoid noisy CI when
# branches differ.
# ---------------------------------------------------------------------------
SECTION_MODULES: List[Tuple[List[str], str]] = [
    (["scripts.summary_sections.header_overview"], "Header Overview"),
    (["scripts.summary_sections.market_context"], "Market Context"),
    (
        ["scripts.summary_sections.social_reddit_context",
         "scripts.summary_sections.social_reddit"],
        "Social Context — Reddit",
    ),
    (
        ["scripts.summary_sections.social_twitter_context",
         "scripts.summary_sections.social_twitter"],
        "Social Context — Twitter",
    ),
    (["scripts.summary_sections.cross_origin_correlation"], "Cross-Origin Correlations"),
    (
        ["scripts.summary_sections.leadlag_analysis",
         "scripts.summary_sections.lead_lag"],
        "Lead–Lag Analysis",
    ),
    (["scripts.summary_sections.drift_response"], "Automated Drift Response"),
    (["scripts.summary_sections.model_lineage"], "Model Lineage & Provenance"),
    (["scripts.summary_sections.model_performance_trend"], "Model Performance Trends"),
    (
        ["scripts.governance.model_governance_actions",
         "scripts.summary_sections.model_governance_actions"],
        "Model Governance Actions",
    ),
    (["scripts.summary_sections.trigger_explainability"], "Trigger Explainability"),
    (["scripts.summary_sections.signal_quality"], "Signal Quality"),
    (
        ["scripts.summary_sections.thresholds",
         "scripts.summary_sections.threshold_recommendations"],
        "Thresholds & Backtests",
    ),
    (["scripts.summary_sections.source_yield_plan"], "Source Yield Plan"),
]

# Modules with “soft presence” across branches — skip quietly if missing
OPTIONAL_PREFIXES = (
    "scripts.summary_sections.social_",
    "scripts.summary_sections.leadlag",
    "scripts.summary_sections.lead_lag",
    "scripts.summary_sections.threshold",
    "scripts.summary_sections.source_yield_plan",
)

HEADER_LINE = "MoonWire CI Demo Summary"
FOOTER_LINE = "Job summary generated at run-time"


def _env_bool(key: str, default: bool = False) -> bool:
    return str(os.getenv(key, str(default))).lower() in ("1", "true", "yes", "on")


def _append_error(md: List[str], module_path: str, err: BaseException) -> None:
    if module_path.startswith(OPTIONAL_PREFIXES):
        return  # optional, stay quiet
    md.append(f"❌ {module_path} failed: {err}")


def _import_first_available(candidates: List[str]) -> Optional[Any]:
    for m in candidates:
        try:
            return importlib.import_module(m)
        except Exception:
            continue
    return None


def _run_section(md: List[str], ctx: SummaryContext, module_candidates: List[str]) -> None:
    mod = _import_first_available(module_candidates)
    if mod is None:
        return

    # header_overview needs kwargs in this repo
    if any(m.endswith(".header_overview") for m in module_candidates):
        try:
            reviewers: List[dict] = []
            threshold: float = 0.50
            sig_id: str = "demo"
            # seed demo reviewers if available
            try:
                from scripts.mw_demo_summary import generate_demo_data_if_needed
                reviewers, _events = generate_demo_data_if_needed(reviewers)
            except Exception:
                pass
            mod.append(md, ctx, reviewers=reviewers, threshold=threshold, sig_id=sig_id)  # type: ignore
        except Exception as e:
            _append_error(md, module_candidates[0], e)
        return

    if not hasattr(mod, "append"):
        _append_error(md, module_candidates[0], RuntimeError("module has no append(md, ctx)"))
        return

    try:
        mod.append(md, ctx)  # type: ignore[attr-defined]
    except Exception as e:
        _append_error(md, module_candidates[0], e)


def _normalize_markdown(lines: List[str]) -> List[str]:
    """
    Post-process the assembled markdown to avoid duplicated headers/footers
    that can arise when individual sections print their own titles.
    Rules:
      - Keep the first HEADER_LINE; drop subsequent exact matches.
      - Keep only the last FOOTER_LINE (we add exactly one; remove extras if any).
      - Collapse accidental consecutive blank lines.
    """
    out: List[str] = []
    seen_header = False
    footer_indices: List[int] = []

    # First pass: record & filter
    for i, ln in enumerate(lines):
        if ln.strip() == HEADER_LINE:
            if seen_header:
                continue
            seen_header = True
        if ln.strip() == FOOTER_LINE:
            footer_indices.append(len(out))  # where it would be placed
        out.append(ln)

    # If multiple footers made it in, keep only the final one
    if len(footer_indices) > 1:
        # remove all but last occurrence
        keep_at = footer_indices[-1]
        idx_map = set(footer_indices[:-1])
        out = [ln for j, ln in enumerate(out) if not (ln.strip() == FOOTER_LINE and j in idx_map)]

    # Collapse double blanks
    compact: List[str] = []
    prev_blank = False
    for ln in out:
        is_blank = (ln.strip() == "")
        if is_blank and prev_blank:
            continue
        compact.append(ln)
        prev_blank = is_blank

    return compact


def build_all(ctx: SummaryContext) -> List[str]:
    ensure_dir(Path(ctx.artifacts_dir))
    lines: List[str] = []
    for candidates, _title in SECTION_MODULES:
        _run_section(lines, ctx, candidates)

    # Single footer (we'll normalize later if any section added its own)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines.append(FOOTER_LINE)

    return _normalize_markdown(lines)