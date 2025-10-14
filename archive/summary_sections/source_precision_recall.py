# scripts/summary_sections/source_precision_recall.py
from __future__ import annotations
import random
from scripts.summary_sections.common import SummaryContext, is_demo_mode

def _generate_demo_source_metrics_if_needed(metrics: dict) -> dict:
    if not is_demo_mode():
        return metrics
    rows = (metrics or {}).get("origins") or []
    known = [r for r in rows if r.get("origin") != "unknown"]
    if known:
        return metrics
    demo_rows = []
    for origin in ["twitter", "reddit", "rss_news"]:
        precision = round(random.uniform(0.25, 0.9), 2)
        recall    = round(random.uniform(0.10, 0.6),  2)
        demo_rows.append({"origin": origin, "precision": precision, "recall": recall})
    return {"window_days": 7, "origins": demo_rows, "notes": ["_demo mode: metrics seeded_"]}

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 📉 Source Precision & Recall (7d)")
    try:
        from src.analytics.source_metrics import compute_source_metrics
    except Exception as e:
        md.append(f"_unavailable: {type(e).__name__}_")
        return

    try:
        metrics = compute_source_metrics(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7,
            min_count=1
        )
        metrics = _generate_demo_source_metrics_if_needed(metrics)
    except Exception as e:
        md.append(f"_⚠️ Source metrics failed: {e}_")
        return

    rows = metrics.get("origins", [])
    if not rows:
        md.append("_No eligible origins to display._")
        return

    for row in rows:
        md.append(f"- `{row['origin']}`: precision={row['precision']} | recall={row['recall']}")