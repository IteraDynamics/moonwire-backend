#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List

ART_DIR = Path("artifacts")
MODELS_DIR = Path("models")

def _read_metrics():
    p = MODELS_DIR / "performance_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def append(md_lines: List[str], ctx) -> None:
    metrics = _read_metrics()
    if not metrics:
        md_lines.append("🚀 Signal Performance Validation (v0.9.0)\n")
        md_lines.append("⚠️ No performance metrics available\n")
        return

    agg = metrics.get("aggregate", {})
    trades = agg.get("trades", 0)
    sharpe = agg.get("sharpe", float("nan"))
    sortino = agg.get("sortino", float("nan"))
    maxdd = agg.get("max_drawdown", float("nan"))
    win = agg.get("win_rate", float("nan"))
    pf = agg.get("profit_factor", float("nan"))

    md_lines.append(f"🚀 Signal Performance Validation (v0.9.0 • {metrics.get('mode','backtest')})\n")
    md_lines.append(
        f"trades={trades} │ Sharpe={_fmt(sharpe)} │ Sortino={_fmt(sortino)} │ "
        f"MaxDD={_fmt_pct(maxdd)} │ Win={_fmt_pct(win)} │ PF={_fmt(pf)}\n"
    )
    # Links to artifacts if present
    art = []
    if (ART_DIR / "perf_equity_curve.png").exists(): art.append("perf_equity_curve.png")
    if (ART_DIR / "perf_drawdown.png").exists(): art.append("perf_drawdown.png")
    if (ART_DIR / "perf_returns_hist.png").exists(): art.append("perf_returns_hist.png")
    if art:
        md_lines.append(f"artifacts: " + " • ".join(art) + "\n")

def _fmt(x):
    try:
        if x is None or (isinstance(x, float) and (x != x)):  # NaN
            return "n/a"
        return f"{x:.2f}"
    except Exception:
        return "n/a"

def _fmt_pct(x):
    try:
        if x is None or (isinstance(x, float) and (x != x)):
            return "n/a"
        return f"{x*100:.1f}%"
    except Exception:
        return "n/a"
