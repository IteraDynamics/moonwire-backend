# scripts/summary_sections/performance_validation.py
import json
from pathlib import Path


def append(md, ctx):
    """Append performance validation block to CI markdown."""
    path = ctx.models_dir / "performance_metrics.json"
    if not path.exists():
        md.append("🚀 Signal Performance Validation (v0.9.0)\n⚠️ No performance metrics available")
        return

    metrics = json.loads(path.read_text())
    agg = metrics
    fmt_pct = lambda x: f"{x*100:.1f}%"

    md.append(f"🚀 Signal Performance Validation (v0.9.0 • {metrics.get('mode','backtest')})")
    md.append(
        f"trades={agg.get('trades',0)} | "
        f"Sharpe={agg.get('sharpe',0):.2f} | "
        f"Sortino={agg.get('sortino',0):.2f} | "
        f"MaxDD={fmt_pct(agg.get('max_drawdown',0))} | "
        f"Win={fmt_pct(agg.get('win_rate',0))} | "
        f"PF={agg.get('profit_factor',0):.2f}"
    )

    # Artifacts
    md.append("artifacts: perf_equity_curve.png • perf_drawdown.png • perf_returns_hist.png")