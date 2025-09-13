# scripts/summary_sections/source_yield_plan.py
from src.analytics.source_yield import compute_source_yield

def _demo_fallback_from_breakdown(origins_rows):
    # tiny demo fallback if analytics return nothing
    total = sum(int(r.get("count", 0) or 0) for r in (origins_rows or [])) or 1
    out = []
    for r in (origins_rows or []):
        flags = int(r.get("count", 0) or 0)
        score = (flags / total) if total else 0.0
        out.append({"origin": r["origin"], "flags": flags, "triggers": 0, "yield_score": score})
    s = sum(x["yield_score"] for x in out) or 1.0
    budget = [{"origin": x["origin"], "pct": round(100 * x["yield_score"] / s, 1)} for x in out]
    return {"window_days": 7, "totals": {}, "origins": out, "budget_plan": budget, "notes": ["demo-fallback"]}

def append(md, ctx, **kwargs):
    md.append("\n### 📈 Source Yield Plan (last 7 days)")
    try:
        min_ev = 1 if ctx.is_demo else 5
        y = compute_source_yield(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7,
            min_events=min_ev,
            alpha=0.7,
        )
        if not y or not y.get("budget_plan"):
            if ctx.is_demo and ctx.origins_rows:
                y = _demo_fallback_from_breakdown(ctx.origins_rows)
        if not y or not y.get("budget_plan"):
            md.append("_No yield plan available (not enough recent activity)._")
            ctx.yield_data = y or {}
            return

        md.append("**Rate-limit budget plan:**")
        for item in y["budget_plan"]:
            md.append(f"- `{item['origin']}` → **{item['pct']}%**")

        md.append("\n**Raw Origin Stats:**")
        for o in (y.get("origins") or []):
            md.append(f"- `{o['origin']}`: {o.get('flags',0)} flags, {o.get('triggers',0)} triggers → score={o.get('yield_score')}")
        ctx.yield_data = y
    except Exception as e:
        md.append(f"_⚠️ Yield plan failed: {e}_")
        ctx.yield_data = {}