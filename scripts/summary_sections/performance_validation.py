# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Any

from .common import SummaryContext, _fmt_pct

METRICS_PATH = Path("models/performance_metrics.json")
ARTS = [
    ("artifacts/perf_equity_curve.png", "equity_curve.png"),
    ("artifacts/perf_drawdown.png", "perf_drawdown.png"),
    ("artifacts/perf_returns_hist.png", "perf_returns_hist.png"),
    ("artifacts/perf_by_symbol_bar.png", "perf_by_symbol_bar.png"),
]

def _load_metrics() -> Any:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None

def _fmt_float(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"

def _fmt_pct_neg(dd):
    # Always show drawdown as a negative percentage if provided
    if dd is None:
        return "n/a"
    try:
        v = float(dd)
        if v > 0:
            v = -abs(v)
        else:
            v = -abs(v)  # ensure negative sign for display
        return _fmt_pct(v, nd=1)
    except Exception:
        return "n/a"

def _mode_label(m):
    m = (m or "backtest").lower()
    return "backtest" if m not in ("live", "paper", "paper_trading") else "live"

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    CI-friendly summary card for performance validation.
    Assumes models/performance_metrics.json exists (best-effort).
    """
    data = _load_metrics()
    mode = _mode_label((data or {}).get("mode") or os.getenv("MW_PERF_MODE", "backtest"))
    window_h = (data or {}).get("window_hours") or os.getenv("MW_PERF_LOOKBACK_H", "72")

    # Single header decoration (avoid duplicate emojis/headers)
    md.append(f"### Signal Performance Validation (v0.9.0 • {window_h}h {mode})")

    if not data:
        md.append("⚠️ No performance metrics available")
        return

    agg = data.get("aggregate") or {}
    trades = agg.get("trades", 0)
    sharpe = _fmt_float(agg.get("sharpe"))
    sortino = _fmt_float(agg.get("sortino"))
    maxdd = _fmt_pct_neg(agg.get("max_drawdown"))
    win = _fmt_pct(agg.get("win_rate"), nd=1) if agg.get("win_rate") is not None else "n/a"
    pf = _fmt_float(agg.get("profit_factor"))

    md.append(
        f"trades={trades} │ Sharpe={sharpe} │ Sortino={sortino} │ MaxDD={maxdd} │ Win={win} │ PF={pf}"
    )

    bysym = data.get("by_symbol") or {}
    if bysym:
        parts = []
        for sym, row in bysym.items():
            s = _fmt_float(row.get("sharpe"))
            wr = _fmt_pct(row.get("win_rate"), nd=1) if row.get("win_rate") is not None else "n/a"
            parts.append(f"{sym}(S={s}, WR={wr})")
        md.append("by symbol: " + ", ".join(parts))

    # Artifact links (names only; your summary decorator already prettifies images)
    existing = []
    for p, label in ARTS:
        if Path(p).exists():
            existing.append(label)
    if existing:
        md.append("artifacts: " + " • ".join(existing))