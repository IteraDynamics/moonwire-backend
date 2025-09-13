# scripts/summary_sections/nowcast_attention.py

from src.analytics.nowcast_attention import compute_nowcast_attention

def _demo_seed():
    return {
        "origins": [
            {"origin":"twitter","score":92.4,"rank":1,"components":{"z":3.1,"precision":0.7,"leadership":0.5,"regime":"turbulent"}},
            {"origin":"reddit","score":74.8,"rank":2,"components":{"z":1.8,"precision":0.55,"leadership":0.4,"regime":"normal"}},
            {"origin":"rss_news","score":66.3,"rank":3,"components":{"z":1.2,"precision":0.5,"leadership":0.2,"regime":"calm"}},
        ]
    }

def append(md, ctx, **kwargs):
    md.append("\n### ⚡ Nowcast Attention (hour)")
    try:
        na = compute_nowcast_attention(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7, interval="hour", lookback=72, z_cap=5.0, top=3,
        )
        rows = [o for o in (na or {}).get("origins", []) if o.get("origin") != "unknown"][:3]
        if not rows and ctx.is_demo:
            rows = _demo_seed()["origins"]
        if not rows:
            md.append("_No attention highlights._")
            ctx.caches["nowcast_attention"] = {}
            return
        for r in rows:
            c = r.get("components", {})
            md.append(f"- {r['origin']}: {r['score']}  (z={c.get('z')}, p={c.get('precision')}, lead={c.get('leadership')}, {c.get('regime','n/a')})")
        ctx.caches["nowcast_attention"] = rows
    except Exception as e:
        md.append(f"_⚠️ Nowcast attention failed: {e}_")
        ctx.caches["nowcast_attention"] = {}