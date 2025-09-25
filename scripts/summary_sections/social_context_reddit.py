# scripts/summary_sections/social_context_reddit.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..social.reddit_lite_ingest import run_ingest as run_reddit_ingest, IngestPaths
from ..social.reddit_client_lite import LiteConfig

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and str(v).strip() != "" else default

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@dataclass
class SummaryContext:
    # Provided in tests in this repo; matching shape used elsewhere
    logs_dir: Path
    models_dir: Path
    is_demo: bool
    artifacts_dir: Optional[Path] = None

def _paths_from_ctx(ctx: SummaryContext) -> IngestPaths:
    logs = Path(ctx.logs_dir)
    models = Path(ctx.models_dir)
    arts = Path(getattr(ctx, "artifacts_dir", Path(os.getcwd()) / "artifacts"))
    arts.mkdir(parents=True, exist_ok=True)
    return IngestPaths(logs_dir=logs, models_dir=models, artifacts_dir=arts)

def _format_line(sub: str, c: Dict[str, Any], bursts_for_sub: int, top_terms: List[Dict[str, Any]]) -> str:
    posts = c.get("posts", 0)
    uniq = c.get("unique_authors", 0)
    terms = ", ".join(t["term"] for t in top_terms[:3]) if top_terms else "n/a"
    burst_txt = f"{bursts_for_sub} burst{'s' if bursts_for_sub != 1 else ''}"
    return f"{sub:<14} → {posts} posts | {burst_txt} | top: {terms}"

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build/refresh social_reddit_context.json (ingest-lite) and append markdown
    summary lines to the CI summary block.
    """
    cfg = LiteConfig.from_env()
    cfg.demo = bool(ctx.is_demo) or (str(_env("DEMO_MODE", "false")).lower() == "true")
    # auto-downgrade to RSS if API creds not set
    if cfg.mode == "api" and not (cfg.client_id and cfg.client_secret):
        cfg.mode = "rss"

    paths = _paths_from_ctx(ctx)
    # Always run ingest (it’s idempotent and cheap in RSS)
    artifact = run_reddit_ingest(paths=paths, cfg=cfg)

    mode_label = artifact.get("mode", "rss")
    window_h = int(artifact.get("window_hours", 72))
    md.append(f"### 🗞️ Social Context — Reddit ({window_h}h) ({mode_label})")

    counts = artifact.get("counts", {})
    bursts = artifact.get("bursts", [])
    bursts_by_sub: Dict[str, int] = {}
    for b in bursts:
        bursts_by_sub[b.get("subreddit","")] = bursts_by_sub.get(b.get("subreddit",""), 0) + 1

    series = artifact.get("series", {})
    subs = artifact.get("subs", [])
    if not subs:
        subs = list(counts.keys())

    for sub in subs:
        c = counts.get(sub, {"posts": 0, "unique_authors": 0})
        s = series.get(sub, {})
        top_terms: List[Dict[str, Any]] = s.get("top_terms", [])
        md.append(_format_line(sub, c, bursts_by_sub.get(sub, 0), top_terms))

    md.append("_Data via Reddit RSS/API. Rate limits respected in API mode._")