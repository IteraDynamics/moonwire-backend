import json
from pathlib import Path
from typing import List

MODELS_DIR = Path("models")
ART_DIR = Path("artifacts")

def _fmt_pct(x):
    if x is None:
        return "n/a"
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "n/a"

def append(md_lines: List[str], ctx) -> List[str]:
    p = MODELS_DIR / "performance_metrics.json"
    if not p.exists():
        md_lines.append("🚀 Signal Performance Validation (v0.9.0)\n")
        md_lines.append("⚠️ No performance metrics available\n\n")
        return md_lines

    data = json.loads(p.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {}) or {}
    brk = data.get("by_symbol", {}) or {}

    trades = agg.get("trades", 0)
    sharpe = agg.get("sharpe", float("nan"))
    sortino = agg.get("sortino", float("nan"))
    maxdd = agg.get("max_drawdown", 0.0)  # negative
    wr = agg.get("win_rate", float("nan"))
    pf = agg.get("profit_factor", float("nan"))

    md_lines.append(f"🚀 Signal Performance Validation (v0.9.0 • {data.get('mode','backtest')})\n")
    md_lines.append(
        f"trades={trades} │ Sharpe={sharpe:.2f} │ Sortino={sortino:.2f} │ "
        f"MaxDD={_fmt_pct(maxdd)} │ Win={_fmt_pct(wr)} │ PF={pf:.2f}\n"
    )

    # compact per-symbol
    if brk:
        parts = []
        for s, m in brk.items():
            parts.append(f"{s}(S={m.get('sharpe', float('nan')):.2f}, WR={_fmt_pct(m.get('win_rate'))})")
        md_lines.append("by symbol: " + ", ".join(parts) + "\n")

    # artifacts list
    arts = []
    for fn in ["perf_equity_curve.png","perf_drawdown.png","perf_returns_hist.png","perf_by_symbol_bar.png"]:
        if (ART_DIR / fn).exists():
            arts.append(fn)
    if arts:
        md_lines.append("artifacts: " + " • ".join(arts) + "\n")

    md_lines.append("\n")
    return md_lines
