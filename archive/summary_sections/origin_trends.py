# scripts/summary_sections/origin_trends.py
from __future__ import annotations
from typing import List

from src.analytics.origin_trends import compute_origin_trends
from .common import SummaryContext, generate_demo_origin_trends_if_needed

def append(md: List[str], ctx: SummaryContext) -> None:
    md.append("\n### 📊 Origin Trends (7d)")
    try:
        t = compute_origin_trends(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7,
            interval="day",
        )
        t = generate_demo_origin_trends_if_needed(t, days=7, interval="day")
        ctx.caches["origin_trends"] = t

        origins = t.get("origins", [])
        if not origins:
            md.append("_No trend data available._")
            return

        for item in origins:
            md.append(f"- **{item.get('origin','unknown')}**")
            for b in item.get("buckets", []):
                day = str(b.get("timestamp_bucket",""))[:10]
                md.append(f"  - {day}: flags={b.get('flags_count',0)}, triggers={b.get('triggers_count',0)}")
    except Exception as e:
        md.append(f"_⚠️ Origin trends failed: {e}_")