# scripts/summary_sections/performance_validation.py
"""
CI section: Signal Performance Validation (v0.9.0)
Reads models/performance_metrics.json and renders a compact block with links.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .common import SummaryContext


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def append(md: List[str], ctx: SummaryContext) -> None:
    models_dir: Path = getattr(ctx, "models_dir", Path("models"))
    artifacts_dir: Path = getattr(ctx, "artifacts_dir", Path("artifacts"))

    j = models_dir / "performance_metrics.json"
    if not j.exists():
        md.append("### 🚀 Signal Performance Validation (v0.9.0)")
        md.append("⚠️ No performance metrics available")
        return

    try:
        data = json.loads(j.read_text())
    except Exception as e:
        md.append("### 🚀 Signal Performance Validation (v0.9.0)")
        md.append(f"❌ Could not read metrics: {e}")
        return

    agg = data.get("aggregate", {})
    mode = data.get("mode", "backtest")
    window = data.get("window_hours", 72)

    md.append(f"### 🚀 Signal Performance Validation (v0.9.0 • {mode})")
    line = (
        f"trades={agg.get('trades', 0)} │ "
        f"Sharpe={agg.get('sharpe', 0.0):.2f} │ "
        f"Sortino={agg.get('sortino', 0.0):.2f} │ "
        f"MaxDD={_fmt_pct(abs(agg.get('max_drawdown', 0.0)))} │ "
        f"Win={_fmt_pct(agg.get('win_rate', 0.0))} │ "
        f"PF={agg.get('profit_factor', 0.0):.2f}"
    )
    md.append(line)

    bys = data.get("by_symbol", {})
    if bys:
        parts = []
        for sym, m in bys.items():
            parts.append(f"{sym}(S={m.get('sharpe', 0.0):.2f}, WR={_fmt_pct(m.get('win_rate', 0.0))})")
        md.append("by symbol: " + ", ".join(parts))

    # artifact hints
    eq = artifacts_dir / "perf_equity_curve.png"
    dd = artifacts_dir / "perf_drawdown.png"
    rh = artifacts_dir / "perf_returns_hist.png"
    md.append(f"artifacts: {eq.name} • {dd.name} • {rh.name}")