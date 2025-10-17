# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from .common import SummaryContext

def _fmt_num(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def append(md: List[str], ctx: SummaryContext) -> None:
    title = "🚀 Signal Performance Validation (v0.9.0)"
    metrics_path = Path(ctx.models_dir) / "performance_metrics.json"
    img_path = Path(ctx.artifacts_dir) / "performance_validation_summary.png"

    data = _read_json(metrics_path)
    if not data:
        md.append(f"\n{title}\n⚠️ No performance metrics available")
        if img_path.exists():
            md.append(f"Visual {img_path}")
        return

    trades = int(data.get("trades", 0) or 0)
    sharpe = data.get("sharpe")
    sortino = data.get("sortino")
    max_dd = data.get("max_drawdown") or data.get("max_dd")
    win = data.get("win_rate") or data.get("winrate")
    pf = data.get("profit_factor") or data.get("pf")
    by_symbol = data.get("by_symbol") or {}

    md.append(f"\n{title}")

    if trades <= 0:
        md.append("⚠️ No backtestable trades found in the lookback window.")
        if img_path.exists():
            md.append(f"Visual {img_path}")
        return

    head = (
        f"trades={trades} │ "
        f"Sharpe={_fmt_num(sharpe)} │ "
        f"Sortino={_fmt_num(sortino)} │ "
        f"MaxDD={_fmt_num(max_dd)}% │ "
        f"Win={_fmt_num(win)}% │ "
        f"PF={_fmt_num(pf)}"
    )
    md.append(head)

    # by-symbol only when trades>0
    bits: List[str] = []
    if isinstance(by_symbol, dict):
        for sym, s in by_symbol.items():
            s = s or {}
            bits.append(f"{sym.upper()}(S={_fmt_num(s.get('sharpe'))}, WR={_fmt_num(s.get('win_rate'))}%)")
    elif isinstance(by_symbol, list):
        for s in by_symbol:
            s = s or {}
            sym = (s.get("symbol") or "").upper()
            if sym:
                bits.append(f"{sym}(S={_fmt_num(s.get('sharpe'))}, WR={_fmt_num(s.get('win_rate'))}%)")
    if bits:
        md.append("by symbol: " + ", ".join(bits))

    if img_path.exists():
        md.append(f"Visual {img_path}")
