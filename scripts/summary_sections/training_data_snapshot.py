# scripts/summary_sections/training_data_snapshot.py
from __future__ import annotations
import os, json
from scripts.summary_sections.common import SummaryContext

def _load_jsonl_quiet(path):
    try:
        if not path.exists(): return []
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        return []

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 📦 Training Data Snapshot")
    td_path = ctx.models_dir / "training_data.jsonl"
    rows = _load_jsonl_quiet(td_path)

    demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("1","true","yes")
    if (not rows) and demo_mode:
        rows = [
            {"timestamp": "2025-09-08T14:30:00Z", "origin": "reddit",   "features": {"burst_z":1.6}, "label": True},
            {"timestamp": "2025-09-08T13:50:00Z", "origin": "rss_news", "features": {"burst_z":0.3}, "label": False},
            {"timestamp": "2025-09-08T13:30:00Z", "origin": "twitter",  "features": {"burst_z":1.1}, "label": True},
        ]

    if not rows:
        md.append("_No training rows yet (waiting for joined trigger+label pairs)._")
        return

    total = len(rows)
    by_origin = {}
    pos = neg = 0
    for r in rows:
        o = (r.get("origin") or "unknown").lower()
        by_origin[o] = by_origin.get(o, 0) + 1
        if bool(r.get("label", False)):
            pos += 1
        else:
            neg += 1
    md.append(f"- Total rows: **{total}**")
    for o in sorted(by_origin.keys()):
        md.append(f"- {o} = {by_origin[o]}")
    md.append(f"- Positives: **{pos}** | Negatives: **{neg}**")