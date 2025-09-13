# scripts/summary_sections/cross_origin_correlations.py

from src.analytics.origin_correlations import compute_origin_correlations
from datetime import datetime, timezone

def _demo_seed(days=7, interval="day"):
    origins = ["twitter", "reddit", "rss_news"]
    pairs = [
        {"a": "twitter", "b": "rss_news", "correlation": 0.703},
        {"a": "twitter", "b": "reddit",   "correlation": 0.427},
        {"a": "reddit",  "b": "rss_news", "correlation": 0.338},
    ]
    return {"window_days": days, "interval": interval, "origins": origins, "pairs": pairs, "notes": ["demo seed"]}

def append(md, ctx, **kwargs):
    md.append("\n### 🔗 Cross-Origin Correlations (7d)")
    try:
        res = compute_origin_correlations(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7, interval="day",
        )
        if not res or not res.get("pairs"):
            if ctx.is_demo:
                res = _demo_seed()
        pairs = (res or {}).get("pairs", [])
        if not pairs:
            md.append("_No correlation data available._")
            ctx.caches["correlations"] = {}
            return
        for p in pairs[:3]:
            md.append(f"- `{p['a']}` ↔ `{p['b']}` → **{p['correlation']}**")
        ctx.caches["correlations"] = res
    except Exception as e:
        md.append(f"_⚠️ Correlation analysis failed: {e}_")
        ctx.caches["correlations"] = {}