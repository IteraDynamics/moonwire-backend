# scripts/summary_sections/lead_lag.py

from src.analytics.lead_lag import compute_lead_lag

def _demo_seed(days=7, interval="hour"):
    pairs = [
        {"a": "twitter",  "b": "rss_news", "best_lag": 21, "correlation": 0.645, "leader": "twitter"},
        {"a": "rss_news", "b": "reddit",   "best_lag": 11, "correlation": 0.626, "leader": "rss_news"},
        {"a": "reddit",   "b": "twitter",  "best_lag": 15, "correlation": 0.551, "leader": "reddit"},
    ]
    return {"window_days": days, "interval": interval, "pairs": pairs, "notes": ["demo seed"]}

def append(md, ctx, **kwargs):
    md.append("\n### ⏱️ Lead–Lag (7d, hour)")
    try:
        res = compute_lead_lag(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7, interval="hour", max_lag=24, use="flags",
        )
        if not res or not res.get("pairs"):
            if ctx.is_demo:
                res = _demo_seed()
        pairs = (res or {}).get("pairs", [])[:3]
        if not pairs:
            md.append("_No lead–lag pairs available._")
            ctx.caches["lead_lag"] = {}
            return
        for p in pairs:
            sign = "+" if p["best_lag"] >= 0 else ""
            unit = "h" if res.get("interval", "hour") == "hour" else "d"
            md.append(f"- {p['a']} → {p['b']}: {sign}{p['best_lag']}{unit} (r={p['correlation']})")
        ctx.caches["lead_lag"] = res
    except Exception as e:
        md.append(f"_⚠️ Lead–lag analysis failed: {e}_")
        ctx.caches["lead_lag"] = {}