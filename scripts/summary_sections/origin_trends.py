# scripts/summary_sections/origin_trends.py
from src.analytics.origin_trends import compute_origin_trends
from datetime import datetime, timezone, timedelta
import random

def _demo_seed(days=7):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    def daily(n):
        out = []
        for i in range(n):
            ts = (now - timedelta(days=(n - 1 - i))).replace(hour=0)
            out.append({"timestamp_bucket": ts.isoformat(), "flags_count": random.randint(0, 8), "triggers_count": random.randint(0, 4)})
        return out
    return {"window_days": days, "interval": "day", "origins": [
        {"origin": "reddit", "buckets": daily(days)},
        {"origin": "rss_news", "buckets": daily(days)},
        {"origin": "twitter", "buckets": daily(days)},
    ]}

def append(md, ctx, **kwargs):
    md.append("\n### 📊 Origin Trends (7d)")
    try:
        tr = compute_origin_trends(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7,
            interval="day",
        )
        if not tr or not tr.get("origins"):
            if ctx.is_demo:
                tr = _demo_seed(days=7)
        if not tr or not tr.get("origins"):
            md.append("_No trend data available._")
            ctx.caches["origin_trends"] = {}
            return

        for item in tr["origins"]:
            md.append(f"- **{item['origin']}**")
            for b in item.get("buckets", []):
                day = str(b.get("timestamp_bucket",""))[:10]
                md.append(f"  - {day}: flags={b.get('flags_count',0)}, triggers={b.get('triggers_count',0)}")
        ctx.caches["origin_trends"] = tr
    except Exception as e:
        md.append(f"_⚠️ Origin trends failed: {e}_")
        ctx.caches["origin_trends"] = {}