# scripts/summary_sections/__init__.py
from __future__ import annotations

import importlib
from typing import Any, Callable, List, Optional, Tuple

from .common import SummaryContext

def _try_import(name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(f"scripts.summary_sections.{name}"), None
    except ModuleNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"error: {e}"

def _safe_append(mod: Any, md: List[str], ctx: SummaryContext, **kwargs: Any) -> None:
    if not mod:
        return
    fn: Optional[Callable[..., None]] = getattr(mod, "append", None)
    if not callable(fn):
        return
    try:
        fn(md, ctx, **kwargs)
    except TypeError:
        fn(md, ctx)
    except Exception:
        # swallow to keep CI summary resilient
        pass

def build_all(ctx: SummaryContext) -> List[str]:
    md: List[str] = []

    order = [
        "header_overview",               # single header-ish block (NO H1)
        "market_context",
        "social_reddit_context",
        "social_twitter_context",
        "cross_origin_correlation",
        "leadlag_analysis",
        "drift_response",
        "model_lineage",
        "model_performance_trend",       # Task 2
        "model_governance_actions",      # Task 3
        "explainability",
        "signal_quality",
        "thresholds",
        "yield_plan",
    ]

    modules: dict[str, Any] = {}
    for name in order:
        mod, _reason = _try_import(name)
        if mod:
            modules[name] = mod

    # Header first (accepts compatibility kwargs)
    _safe_append(
        modules.get("header_overview"), md, ctx,
        reviewers=[], threshold=0.5, sig_id="demo", triggered_log=[]
    )

    for name in order:
        if name == "header_overview":
            continue
        _safe_append(modules.get(name), md, ctx)

    return md