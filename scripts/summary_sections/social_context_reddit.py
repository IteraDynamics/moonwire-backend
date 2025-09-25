# -*- coding: utf-8 -*-
"""
Social Context — Reddit (72h)

Reads models/social_reddit_context.json (written by scripts.social.reddit_lite_ingest)
and appends a concise markdown block to the CI summary.
"""

from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Dict, Any

from .common import SummaryContext

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _fmt_top_terms(all_terms: List[Dict[str, Any]], sub: str) -> str:
    terms = [t for t in all_terms if t.get("subreddit")==sub]
    tops = [t["term"] for t in sorted(terms, key=lambda x: (-int(x.get("tf",0)), x.get("term","")))[:3]]
    return ", ".join(tops) if tops else "n/a"

def _burst_count(bursts: List[Dict[str, Any]], sub: str) -> int:
    return sum(1 for b in bursts if b.get("subreddit")==sub)

def append(md: List[str], ctx: SummaryContext) -> None:
    models = Path(ctx.models_dir)
    data = _read_json(models / "social_reddit_context.json")
    if not data:
        md.append("\n> ⚠️ Social Context — Reddit: no data (ingest step did not run).\n")
        return

    mode = data.get("mode","rss")
    window_h = int(data.get("window_hours", 72))
    subs = data.get("subs", [])
    counts = data.get("counts", {})
    bursts = data.get("bursts", [])
    terms = data.get("top_terms", [])

    md.append(f"### 🗞️ Social Context — Reddit ({window_h}h) ({mode})")
    lines = []
    for sub in subs:
        c = counts.get(sub, {})
        n_posts = int(c.get("posts", 0))
        n_bursts = _burst_count(bursts, sub)
        top3 = _fmt_top_terms(terms, sub)
        lines.append(f"{sub:<14} → {n_posts} posts | {n_bursts} bursts | top: {top3}")
    if lines:
        md.extend(lines)
    else:
        md.append("_no recent Reddit activity found_")