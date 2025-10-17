# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from scripts.summary_sections.common import _iso

def _fmt_pct(x):
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"

def _fmt_ratio(x):
    if x is None:
        return "n/a"
    return f"{x:.2f}"

def append(md: List[str], ctx) -> None:
    """
    Appends a compact CI block if models/performance_metrics.json exists.
    If not present, adds a one-line warning.
    """
    models_dir = getattr(ctx, "models_dir", Path("models"))
    arts_dir = getattr(ctx, "artifacts_dir", Path("artifacts"))

    j = models_dir / "performance_metrics.json"
    md.append("🚀 Signal Performance Validation (v0.9.0)")
    if not j.exists():
        md.append("⚠️ No performance metrics available")
        return

    try:
        metrics = json.loads(j.read_text(encoding="utf-8"))
    except Exception as e:
        md.append(f"⚠️ Unable to read metrics: {e}")
        return

    agg = metrics.get("aggregate", {})
    trades = agg.get("trades", 0)
    sharpe = _fmt_ratio(agg.get("sharpe"))
    sortino = _fmt_ratio(agg.get("sortino"))
    maxdd = agg.get("max_drawdown")
    wr = _fmt_pct(agg.get("win_rate"))
    pf = _fmt_ratio(agg.get("profit_factor"))

    maxdd_str = "n/a"
    if isinstance(maxdd, (int, float)):
        maxdd_str = f"{maxdd*100:.1f}%"

    # header line
    mode = metrics.get("mode", "backtest")
    win_hours = metrics.get("window_hours", None)
    hdr = f"({mode}" + (f" • {win_hours}h backtest" if (mode == "backtest" and win_hours) else ")")
    if hdr.endswith("backtest"):
        hdr += ")"
    md.append(f"trades={trades} | Sharpe={sharpe} | Sortino={sortino} | MaxDD={maxdd_str} | Win={wr} | PF={pf}")

    # by-symbol short line
    bys = metrics.get("by_symbol", {})
    if bys:
        parts = []
        for sym, row in bys.items():
            s_s = _fmt_ratio(row.get("sharpe"))
            w_s = _fmt_pct(row.get("win_rate"))
            parts.append(f"{sym}(S={s_s}, WR={w_s})")
        md.append("by symbol: " + ", ".join(parts))

    # artifacts list (best-effort)
    arts = []
    for name in ("perf_equity_curve.png", "perf_drawdown.png", "perf_returns_hist.png", "perf_by_symbol_bar.png"):
        if (arts_dir / name).exists():
            arts.append(name)
    if arts:
        md.append("artifacts: " + " • ".join(arts))
