# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List

from .common import SummaryContext

def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False

def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        if x is None or _is_nan(float(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"

def _fmt_pct_signed(x: Any, nd: int = 1) -> str:
    """Preserve sign for drawdown; x is a fraction (e.g., -0.052)."""
    try:
        if x is None or _is_nan(float(x)):
            return "n/a"
        return f"{float(x)*100:.{nd}f}%"
    except Exception:
        return "n/a"

def _load_metrics(models_dir: Path) -> Dict[str, Any]:
    j = models_dir / "performance_metrics.json"
    if not j.exists():
        return {}
    try:
        return json.loads(j.read_text(encoding="utf-8"))
    except Exception:
        return {}

def build_section(ctx: SummaryContext) -> List[str]:
    """
    Always render a section — never raise. If metrics are missing or invalid,
    print a friendly note instead of causing the registry to skip.
    """
    lines: List[str] = []
    try:
        models_dir = Path("models")
        arts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        model_version = os.getenv("MODEL_VERSION", "v0.9.0")

        lines.append(f"### 🚀 Signal Performance Validation ({model_version})")

        data = _load_metrics(models_dir)
        agg = data.get("aggregate", {}) if isinstance(data, dict) else {}
        by_sym = data.get("by_symbol", {}) if isinstance(data, dict) else {}

        trades = agg.get("trades")
        sharpe = agg.get("sharpe")
        sortino = agg.get("sortino")
        max_dd = agg.get("max_drawdown")   # fraction; keep sign
        win_rate = agg.get("win_rate")     # fraction
        pf = agg.get("profit_factor")

        if not isinstance(trades, (int, float)) or trades is None:
            lines.append("⚠️ No performance metrics available")
        else:
            sharpe_s = _fmt_num(sharpe, 2)
            sortino_s = _fmt_num(sortino, 2)
            maxdd_s = _fmt_pct_signed(max_dd, 1)
            win_s = _fmt_pct_signed(win_rate, 1).replace("%", "%")  # reuse pct fmt
            pf_s = _fmt_num(pf, 2)

            lines.append(
                f"trades={int(trades)} │ Sharpe={sharpe_s} │ Sortino={sortino_s} │ "
                f"MaxDD={maxdd_s} │ Win={win_s} │ PF={pf_s}"
            )

            if isinstance(by_sym, dict) and by_sym:
                parts = []
                for sym, d in by_sym.items():
                    s_sharpe = _fmt_num(d.get("sharpe"), 2)
                    s_wr = _fmt_pct_signed(d.get("win_rate"), 1)
                    parts.append(f"{sym}(S={s_sharpe}, WR={s_wr})")
                if parts:
                    lines.append("by symbol: " + ", ".join(parts))

        # Mention plots so the outer writer will inline them as images.
        plot_names = [
            "perf_equity_curve.png",
            "perf_drawdown.png",
            "perf_returns_hist.png",
            "perf_by_symbol_bar.png",
        ]
        for name in plot_names:
            p = arts_dir / name
            if p.exists():
                lines.append(str(p))

        return lines

    except Exception:
        # Never propagate — keep the section, say it’s unavailable
        lines.clear()
        lines.append(f"### 🚀 Signal Performance Validation ({os.getenv('MODEL_VERSION', 'v0.9.0')})")
        lines.append("⚠️ Performance metrics unavailable")
        return lines
