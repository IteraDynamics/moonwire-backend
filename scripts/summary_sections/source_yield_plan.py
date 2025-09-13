# scripts/summary_sections/source_yield_plan.py
from src.analytics.source_yield import compute_source_yield
from .common import is_demo_mode

# optional demo seeder
try:
    from .common import generate_demo_yield_plan_if_needed
except Exception:
    def generate_demo_yield_plan_if_needed(x): return x

def _has_known(origins):
    return any(o.get("origin") and o["origin"] != "unknown" for o in (origins or []))

def append(md, ctx):
    try:
        min_ev = 1 if ctx.is_demo else 5
        yd = compute_source_yield(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7,
            min_events=min_ev,
            alpha=0.7,
        )

        # demo seed if empty or unknown-only
        yd = generate_demo_yield_plan_if_needed(yd)

        # if still unknown-only, hide it and just say no plan
        if not _has_known(yd.get("origins")):
            md.append("\n### 📈 Source Yield Plan (last 7 days)")
            md.append("_No yield plan available (not enough recent activity)._")
            ctx.yield_data = None
            return

        # filter unknown from display (keep raw)
        display_origins = [o for o in yd.get("origins", []) if o.get("origin") != "unknown"]
        display_budget = [b for b in yd.get("budget_plan", []) if b.get("origin") != "unknown"]

        md.append("\n### 📈 Source Yield Plan (last 7 days)")
        if not display_budget:
            md.append("_No yield plan available (not enough recent activity)._")
        else:
            md.append("**Rate-limit budget plan:**")
            for item in display_budget:
                md.append(f"- `{item['origin']}` → **{item['pct']}%**")

            md.append("\n**Raw Origin Stats:**")
            for o in display_origins:
                md.append(f"- `{o['origin']}`: {o['flags']} flags, {o['triggers']} triggers → score={o['yield_score']}")

        # keep raw in context for downstream sections (candidate picking)
        ctx.yield_data = yd

    except Exception as e:
        md.append(f"\n_⚠️ Yield plan failed: {e}_")
        ctx.yield_data = None