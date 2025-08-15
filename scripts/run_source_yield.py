#!/usr/bin/env python3

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.source_yield import compute_source_yield
from src.paths import LOGS_DIR

def render_markdown(result: dict) -> str:
    lines = []
    lines.append(f"### 📈 Source Yield Plan — {result['window_days']}d Window\n")

    if not result["budget_plan"]:
        lines.append("_No eligible origins to display — try lowering `min_events`._\n")
        return "\n".join(lines)

    lines.append("**Rate-limit budget plan:**\n")
    for item in result["budget_plan"]:
        lines.append(f"- `{item['origin']}` → **{item['pct']}%**")

    lines.append("\n**Raw Origin Stats:**")
    for o in result["origins"]:
        lines.append(f"- `{o['origin']}`: {o['flags']} flags, {o['triggers']} triggers → score={o['yield_score']}")

    return "\n".join(lines)


if __name__ == "__main__":
    result = compute_source_yield(
        flags_path=LOGS_DIR / "retraining_log.jsonl",
        triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
        days=7,
        min_events=5,
        alpha=0.7
    )

    md = render_markdown(result)
    print(md)