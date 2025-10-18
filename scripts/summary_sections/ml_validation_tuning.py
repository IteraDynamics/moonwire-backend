# scripts/summary_sections/ml_validation_tuning.py
from __future__ import annotations
import json
from typing import List
from .common import SummaryContext  # this exists in your repo (used by other sections)

def _fmt_pct(x, digits=0):
    try:
        return f"{float(x)*100:.{digits}f}%"
    except Exception:
        return "n/a"

def append(md: List[str], ctx: SummaryContext) -> None:
    try:
        with open("models/backtest_summary.json", "r", encoding="utf-8") as f:
            bt = json.load(f)
    except Exception:
        md.append("\n> ⚠️ Skipping ML Validation & Threshold Tuning (v0.9.1)")
        return

    agg = bt.get("aggregate", {})
    per = bt.get("per_symbol", {})
    try:
        with open("models/signal_thresholds.json", "r", encoding="utf-8") as f:
            params = json.load(f)
    except Exception:
        params = {}

    md.append("\n🚀 🤖 **ML Validation & Threshold Tuning (v0.9.1)**")
    md.append(
        f"\nagg: win={_fmt_pct(agg.get('win_rate', 0.0), 0)} | "
        f"PF={agg.get('profit_factor','n/a'):.2f} | "
        f"MaxDD={agg.get('max_drawdown','n/a'):.1%} | "
        f"signals/day={agg.get('signals_per_day','n/a'):.2f}"
    )

    # per symbol
    if per:
        parts = []
        for sym, m in per.items():
            parts.append(f"{sym}({_fmt_pct(m.get('win_rate',0.0),0)}, PF {m.get('profit_factor',0.0):.2f})")
        md.append("\nper-symbol: " + ", ".join(parts))

    if params:
        md.append(
            f"\nchosen: conf_min={params.get('conf_min')}," 
            f" debounce={params.get('debounce_min')}m, horizon={params.get('horizon_h')}h"
        )

    # link bundle artifacts by name (they’ll show in the “Artifacts” section)
    md.append("\nvisuals: `ml_roc_pr_curve.png`, `bt_equity_curve.png`, more in artifacts/")