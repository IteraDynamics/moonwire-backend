# scripts/summary_sections/source_yield_plan.py
from __future__ import annotations
from typing import List

from src.analytics.source_yield import compute_source_yield
from .common import SummaryContext, generate_demo_yield_plan_if_needed

def append(md: List[str], ctx: SummaryContext) -> None:
    md.append("\n### 📈 Source Yield Plan (last 7 days)")
    try:
        min_ev = 1 if ctx.is_demo else 5
        yd = compute_source_yield(
            flags_path=ctx.logs_dir / "retraining_log.jsonl",
            triggers_path=ctx.logs_dir / "retraining_triggered.jsonl",
            days=7,
            min_events=min_ev,
            alpha=0.7,
        )
        yd = generate_demo_yield_plan_if_needed(yd)
        ctx.yield_data = yd

        if not yd.get("budget_plan"):
            md.append("_No yield plan available (not enough recent activity)._")
            return

        md.append("**Rate-limit budget plan:**")
        for item in yd.get("budget_plan", []):
            md.append(f"- `{item['origin']}` → **{item['pct']}%**")

        md.append("\n**Raw Origin Stats:**")
        for o in yd.get("origins", []):
            md.append(f"- `{o['origin']}`: {o['flags']} flags, {o['triggers']} triggers → score={o['yield_score']}")
    except Exception as e:
        md.append(f"_⚠️ Yield plan failed: {e}_")