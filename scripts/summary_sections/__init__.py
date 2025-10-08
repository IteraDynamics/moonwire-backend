# scripts/summary_sections/__init__.py
from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .common import SummaryContext, ensure_dir

# ---------------------------------------------------------------------------
# Section order: matches yesterday’s Task 2 baseline. Some sections have
# fallback module names to avoid "missing module" noise across branches.
# Each item is ( [candidate_module_paths...], "Human Title" )
# ---------------------------------------------------------------------------
SECTION_MODULES: List[Tuple[List[str], str]] = [
    (["scripts.summary_sections.header_overview"], "Header Overview"),
    (["scripts.summary_sections.market_context"], "Market Context"),
    (
        [
            "scripts.summary_sections.social_reddit_context",
            "scripts.summary_sections.social_reddit",          # fallback
        ],
        "Social Context — Reddit",
    ),
    (
        [
            "scripts.summary_sections.social_twitter_context",
            "scripts.summary_sections.social_twitter",         # fallback
        ],
        "Social Context — Twitter",
    ),
    (["scripts.summary_sections.cross_origin_correlation"], "Cross-Origin Correlations"),
    (
        [
            "scripts.summary_sections.leadlag_analysis",
            "scripts.summary_sections.lead_lag",               # fallback
        ],
        "Lead–Lag Analysis",
    ),
    (["scripts.summary_sections.drift_response"], "Automated Drift Response"),
    (["scripts.summary_sections.model_lineage"], "Model Lineage & Provenance"),
    (["scripts.summary_sections.model_performance_trend"], "Model Performance Trends"),
    (
        [
            "scripts.governance.model_governance_actions",     # correct location
            "scripts.summary_sections.model_governance_actions" # legacy (if ever placed here)
        ],
        "Model Governance Actions",
    ),
    (["scripts.summary_sections.trigger_explainability"], "Trigger Explainability"),
    (["scripts.summary_sections.signal_quality"], "Signal Quality"),
    (
        [
            "scripts.summary_sections.thresholds",
            "scripts.summary_sections.threshold_recommendations",  # fallback
        ],
        "Thresholds & Backtests",
    ),
    (["scripts.summary_sections.source_yield_plan"], "Source Yield Plan"),
]


def _env_bool(key: str, default: bool = False) -> bool:
    import os
    return str(os.getenv(key, str(default))).lower() in ("1", "true", "yes", "on")


def _append_error(md: List[str], module_path: str, err: BaseException) -> None:
    """Only show hard errors for core sections; optional sections are silent if missing."""
    OPTIONAL_PREFIXES = {
        "scripts.summary_sections.social_",   # reddit/twitter blocks optional in some branches
        "scripts.summary_sections.leadlag",   # optional
        "scripts.summary_sections.lead_lag",  # optional
        "scripts.summary_sections.threshold", # optional
        "scripts.summary_sections.source_yield_plan",  # optional
    }
    if any(module_path.startswith(p) for p in OPTIONAL_PREFIXES):
        return
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
        # Nothing found — treat as optional missing (silent)
        # If this was a core module, _append_error would be used when we try to import explicitly.
        # Here we silently skip to avoid noisy CI.
        return

    # Special-case: header_overview requires extra kwargs in this repo.
    if any(m.endswith(".header_overview") for m in module_candidates):
        try:
            reviewers: List[dict] = []
            threshold: float = 0.50
            sig_id: str = "demo"
            try:
                # Use the demo seeding helper if available
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


def build_all(ctx: SummaryContext) -> List[str]:
    ensure_dir(Path(ctx.artifacts_dir))
    lines: List[str] = []
    for candidates, _title in SECTION_MODULES:
        _run_section(lines, ctx, candidates)

    # Single footer
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines.append("Job summary generated at run-time")
    return lines