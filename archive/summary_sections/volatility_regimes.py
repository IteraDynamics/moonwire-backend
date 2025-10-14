# scripts/summary_sections/volatility_regimes.py

from src.analytics.volatility_regimes import compute_volatility_regimes
from src.analytics.threshold_policy import threshold_for_regime

def _demo_seed(days=30, interval="hour"):
    return {
        "window_days": days, "interval": interval,
        "origins": [
            {"origin": "twitter",  "regime": "turbulent"},
            {"origin": "reddit",   "regime": "normal"},
            {"origin": "rss_news", "regime": "calm"},
        ],
        "notes": ["demo seed"],
    }

def append(md, ctx, **kwargs):
    md.append("\n### 🌫️ Volatility Regimes (hour)")
    try:
        raw = compute_volatility_regimes(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=30, interval="hour", lookback=72, q_calm=0.33, q_turb=0.80,
        )
        origins = []
        for o in (raw or {}).get("origins", []):
            if o.get("origin") != "unknown":
                origins.append(o)
        if not origins and ctx.is_demo:
            raw = _demo_seed()
            origins = raw.get("origins", [])
        if not origins:
            md.append("_No volatility data._")
            ctx.caches["volatility_regimes"] = {}
            return
        for row in origins[:3]:
            regime = row.get("regime", "normal")
            thr = threshold_for_regime(regime)
            md.append(f"- {row['origin']}: {regime} → threshold {thr}")
        # cache a simple origin->regime map for later sections
        ctx.caches["volatility_regimes"] = {r["origin"]: r.get("regime","normal") for r in origins}
    except Exception as e:
        md.append(f"_⚠️ Volatility regimes failed: {e}_")
        ctx.caches["volatility_regimes"] = {}