# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from .common import SummaryContext, ensure_dir

def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        # try a float with 2dp; fallback to str
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Compact CI section that summarizes v0.9.0 signal performance validation.

    Inputs (best-effort):
      - models/performance_metrics.json    # produced by validation runner
      - artifacts/performance_validation_summary.png  # visual, optional
    """
    title = "🚀 Signal Performance Validation (v0.9.0)"
    metrics_path = Path(ctx.models_dir) / "performance_metrics.json"
    img_path = Path(ctx.artifacts_dir) / "performance_validation_summary.png"

    data = _load_json(metrics_path)
    if not data:
        md.append(f"\n{title}\n⚠️ No performance metrics available")
        # If a plot exists anyway, surface it
        if img_path.exists():
            md.append(f"Visual {img_path}")
        return

    # Expected keys (soft): trades, sharpe, sortino, max_dd, win_rate, profit_factor, by_symbol
    trades = data.get("trades")
    sharpe = data.get("sharpe")
    sortino = data.get("sortino")
    max_dd = data.get("max_drawdown") or data.get("max_dd")
    win = data.get("win_rate") or data.get("winrate")
    pf = data.get("profit_factor") or data.get("pf")
    by_symbol = data.get("by_symbol") or {}

    # Build compact line
    head = (
        f"trades={trades if trades is not None else 0} │ "
        f"Sharpe={_fmt(sharpe)} │ "
        f"Sortino={_fmt(sortino)} │ "
        f"MaxDD={_fmt(max_dd)}% │ "
        f"Win={_fmt(win)}% │ "
        f"PF={_fmt(pf)}"
    )

    # by-symbol summary
    sym_bits = []
    if isinstance(by_symbol, dict):
        for sym, s in by_symbol.items():
            s_sharpe = _fmt((s or {}).get("sharpe"))
            s_wr = _fmt((s or {}).get("win_rate"))
            sym_bits.append(f"{sym.upper()}(S={s_sharpe}, WR={s_wr}%)")
    elif isinstance(by_symbol, list):
        for s in by_symbol:
            sym = (s or {}).get("symbol", "")
            s_sharpe = _fmt((s or {}).get("sharpe"))
            s_wr = _fmt((s or {}).get("win_rate"))
            if sym:
                sym_bits.append(f"{str(sym).upper()}(S={s_sharpe}, WR={s_wr}%)")

    md.append(f"\n{title}")
    md.append(head)
    if sym_bits:
        md.append("by symbol: " + ", ".join(sym_bits))

    # If the image exists, add a line that contains the path;
    # the outer markdown writer converts any line containing 'png'
    # into an embedded image with a proper GitHub raw URL.
    if img_path.exists():
        md.append(f"Visual {img_path}")
