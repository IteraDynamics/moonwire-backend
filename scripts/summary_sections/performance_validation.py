# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .common import SummaryContext


def _load_metrics(models_dir: Path) -> dict:
    p = models_dir / "performance_metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _fmt(v, pct=False, nd=2):
    if v is None:
        return "n/a"
    try:
        if pct:
            return f"{v*100:.1f}%"
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def append(md: List[str], ctx: SummaryContext) -> None:
    m = _load_metrics(ctx.models_dir)
    if not m:
        md.append("### Signal Performance Validation (v0.9.0)\n> ⚠️ No performance metrics available")
        return

    agg = m.get("aggregate", {})
    sym = m.get("by_symbol", {})

    parts = [
        "### Signal Performance Validation (v0.9.0 • {win}h {mode})".format(
            win=m.get("window_hours", 0), mode=m.get("mode", "backtest")
        ),
        "trades={t} | Sharpe={S} | Sortino={So} | MaxDD={DD} | Win={WR} | PF={PF}".format(
            t=agg.get("trades", 0),
            S=_fmt(agg.get("sharpe")),
            So=_fmt(agg.get("sortino")),
            DD=_fmt(agg.get("max_drawdown"), pct=True),
            WR=_fmt(agg.get("win_rate"), pct=True),
            PF=_fmt(agg.get("profit_factor")),
        ),
    ]

    if sym:
        bys = []
        for k, v in sym.items():
            bys.append(f"{k}(S={_fmt(v.get('sharpe'))}, WR={_fmt(v.get('win_rate'), pct=True)})")
        parts.append("by symbol: " + ", ".join(bys))

    parts.append("artifacts: perf_equity_curve.png • perf_drawdown.png • perf_returns_hist.png")

    md.append("\n".join(parts))