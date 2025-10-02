# scripts/summary_sections/social_context_twitter.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from .common import SummaryContext


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Render the "Social Context — Twitter" block.
    If the artifact is missing, attempt to build it via the ingest orchestrator.
    """
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    logs_dir = Path(getattr(ctx, "logs_dir", "logs"))

    json_path = models_dir / "social_twitter_context.json"

    # If missing, try to run ingest (best-effort; safe in demo)
    if not json_path.exists():
        try:
            from scripts.social.twitter_lite_ingest import run_ingest, IngestPaths
            run_ingest(paths=IngestPaths(logs_dir=logs_dir, models_dir=models_dir, artifacts_dir=artifacts_dir))
        except Exception as _e:
            pass

    j = _load_json(json_path)
    if not j:
        md.append("\n> ⚠️ Social Context — Twitter: no data (ingest step did not run).")
        return

    mode_label = "api"
    if bool(j.get("demo")):
        mode_label = "demo"

    total = _safe_int(j.get("counts", {}).get("total_tweets"))
    uniq = _safe_int(j.get("counts", {}).get("unique_authors"))
    bursts = j.get("bursts") or []
    top_terms = j.get("top_terms") or []
    terms_txt = ", ".join([t.get("term") for t in top_terms[:3] if isinstance(t, dict) and t.get("term")]) or "n/a"

    md.append(f"\n### 🐦 Social Context — Twitter ({j.get('window_hours', 72)}h) ({mode_label})")
    md.append(f"Tweets: {total} | Unique authors: {uniq} | Bursts: {len(bursts)} | Top terms: {terms_txt}")
    md.append("\n_Footer: Data via Twitter API v2 (search/recent). Rate limits respected; demo fallback when no token._\n")