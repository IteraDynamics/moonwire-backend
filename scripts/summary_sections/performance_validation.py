# scripts/summary_sections/performance_validation.py
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List

from .common import SummaryContext

NUMERIC_SENTINELS = {None, float("nan")}

def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False

def _fmt_num(x: Any, nd=2) -> str:
    if x in NUMERIC_SENTINELS or _is_nan(x):
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"

def _fmt_pct_signed(x: Any, nd=1) -> str:
    """Keep the sign for drawdown; expects a fraction (e.g., -0.052)."""
    if x in NUMERIC_SENTINELS or _is_nan(x):
        return "n/a"
    try:
        v = float(x) * 100.0
        sign = "-" if v < 0 else "+"
        # ensure we always show sign; if you prefer no + sign, drop the branch
        # but per request we want negative to show clearly
        return f"{v:.{nd}f}%".replace("+", "") if v < 0 else f"{v:.{nd}f}%"
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
    Renders the 'Signal Performance Validation' block.
    Looks for models/performance_metrics.json written by the validator or the demo seeder.
    """
    lines: List[str] = []
    models_dir = Path("models")
    arts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    model_version = os.getenv("MODEL_VERSION", "v0.9.0")

    data = _load_metrics(models_dir)
    agg = data.get("aggregate", {}) if isinstance(data, dict) else {}
    by_sym = data.get("by_symbol", {}) if isinstance(data, dict) else {}

    trades = agg.get("trades")
    sharpe = agg.get("sharpe")
    sortino = agg.get("sortino")
    max_dd = agg.get("max_drawdown")  # fraction, negative preferred (e.g., -0.052)
    win_rate = agg.get("win_rate")    # fraction (0..1)
    pf = agg.get("profit_factor")

    lines.append(f"### 🚀 Signal Performance Validation ({model_version})")

    if not trades:
        lines.append("⚠️ No performance metrics available")
        return lines

    # formatters
    sharpe_s = _fmt_num(sharpe, 2)
    sortino_s = _fmt_num(sortino, 2)
    maxdd_s = _fmt_pct_signed(max_dd, 1)
    win_s = "n/a" if (win_rate in NUMERIC_SENTINELS or _is_nan(win_rate)) else f"{float(win_rate)*100:.1f}%"
    pf_s = _fmt_num(pf, 2)

    lines.append(
        f"trades={trades} │ Sharpe={sharpe_s} │ Sortino={sortino_s} │ "
        f"MaxDD={maxdd_s} │ Win={win_s} │ PF={pf_s}"
    )

    # by symbol quick line
    if isinstance(by_sym, dict) and by_sym:
        parts = []
        for sym, d in by_sym.items():
            s_sharpe = _fmt_num(d.get("sharpe"), 2)
            s_wr = d.get("win_rate")
            s_wr_s = "n/a" if (s_wr in NUMERIC_SENTINELS or _is_nan(s_wr)) else f"{float(s_wr)*100:.1f}%"
            parts.append(f"{sym}(S={s_sharpe}, WR={s_wr_s})")
        if parts:
            lines.append("by symbol: " + ", ".join(parts))

    # Mention plots so outer writer inlines images
    # (Your _write_md detects 'png' tokens and turns into image markdown)
    for png in [
        arts_dir / "perf_equity_curve.png",
        arts_dir / "perf_drawdown.png",
        arts_dir / "perf_returns_hist.png",
        arts_dir / "perf_by_symbol_bar.png",
    ]:
        if png.exists():
            lines.append(str(png))

    return lines