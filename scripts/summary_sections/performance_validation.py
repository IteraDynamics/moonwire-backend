# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

def _fmt_pct(x):
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"

def _fmt_ratio(x):
    if x is None:
        return "n/a"
    return f"{x:.2f}"

def append(md: List[str], ctx) -> None:
    # prevent duplicate section if build_all calls twice
    cache = getattr(ctx, "caches", None)
    if isinstance(cache, dict) and cache.get("perf_added"):
        return
    if isinstance(cache, dict):
        cache["perf_added"] = True

    models_dir = getattr(ctx, "models_dir", Path("models"))
    arts_dir = getattr(ctx, "artifacts_dir", Path("artifacts"))

    md.append("🚀 Signal Performance Validation (v0.9.0)")
    j = models_dir / "performance_metrics.json"
    if not j.exists():
        md.append("⚠️ No performance metrics available")
        return

    try:
        metrics = json.loads(j.read_text(encoding="utf-8"))
    except Exception as e:
        md.append(f"⚠️ Unable to read metrics: {e}")
        return

    agg = metrics.get("aggregate", {})
    trades = int(agg.get("trades", 0) or 0)
    sharpe = _fmt_ratio(agg.get("sharpe"))
    sortino = _fmt_ratio(agg.get("sortino"))
    maxdd = agg.get("max_drawdown")
    wr = _fmt_pct(agg.get("win_rate"))
    pf = agg.get("profit_factor")
    pf_str = "∞" if pf is not None and pf > 1e6 else _fmt_ratio(pf)

    maxdd_str = "n/a"
    if isinstance(maxdd, (int, float)):
        maxdd_str = f"{maxdd*100:.1f}%"

    md.append(f"trades={trades} │ Sharpe={sharpe} │ Sortino={sortino} │ MaxDD={maxdd_str} │ Win={wr} │ PF={pf_str}")

    bys = metrics.get("by_symbol", {})
    if bys:
        parts = []
        for sym, row in bys.items():
            s_s = _fmt_ratio(row.get("sharpe"))
            w_s = _fmt_pct(row.get("win_rate"))
            parts.append(f"{sym}(S={s_s}, WR={w_s})")
        md.append("by symbol: " + ", ".join(parts))

    # List plots (the summary renderer converts png lines to images)
    for name in ("perf_equity_curve.png", "perf_drawdown.png", "perf_returns_hist.png", "perf_by_symbol_bar.png"):
        p = arts_dir / name
        if p.exists():
            md.append(str(p))