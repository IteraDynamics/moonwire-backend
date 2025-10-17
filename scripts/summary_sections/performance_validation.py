# scripts/summary_sections/performance_validation.py
from __future__ import annotations
import json, math, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

from .common import SummaryContext, ensure_dir, _iso

def _fmt(x: Any, fmt: str = "{:.2f}") -> str:
    """Format numbers defensively; return 'n/a' for None/NaN/inf."""
    try:
        if x is None:
            return "n/a"
        if isinstance(x, (int, float)):
            if math.isnan(x) or math.isinf(x):
                return "n/a"
            return fmt.format(x)
        # strings or other types
        return str(x)
    except Exception:
        return "n/a"

def _fmt_pct(x: Any) -> str:
    """Format as percentage with one decimal place."""
    try:
        if x is None:
            return "n/a"
        if isinstance(x, (int, float)):
            if math.isnan(x) or math.isinf(x):
                return "n/a"
            return f"{x:.1f}%"
        return str(x)
    except Exception:
        return "n/a"

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Performance Validation section (v0.9.0)
    Produces:
      - artifacts/performance_metrics.json (overall + by_symbol)
      - neat 2-line CI summary
    Never hard-fails CI; prints a readable error instead.
    """
    try:
        arts = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        ensure_dir(arts)

        # ---- Load or synthesize validation results (demo-friendly) ----
        # If your pipeline writes a real file, replace the synthetic block below
        # with a loader that populates the same 'perf' structure.
        perf: Dict[str, Any] = {
            "trades": 3,
            "sharpe": 0.56,
            "sortino": 2.47,
            "max_drawdown": -0.03,   # as decimal (-3%)
            "win_rate": 0.667,       # as decimal
            "profit_factor": 4.50,
            "by_symbol": {
                "BTC": {"sharpe": float("nan"), "win_rate": 1.00},
                "ETH": {"sharpe": float("nan"), "win_rate": 0.00},
                "SOL": {"sharpe": float("nan"), "win_rate": 1.00},
            }
        }

        # ---- Build outputs for artifacts & markdown ----
        trades = perf.get("trades")
        sharpe = perf.get("sharpe")
        sortino = perf.get("sortino")
        mdd = perf.get("max_drawdown")  # decimal
        wr = perf.get("win_rate")       # decimal
        pf = perf.get("profit_factor")

        # overall metrics dict written to JSON
        overall_metrics = {
            "trades": trades,
            "sharpe": None if (sharpe is None or (isinstance(sharpe, float) and (math.isnan(sharpe) or math.isinf(sharpe)))) else sharpe,
            "sortino": None if (sortino is None or (isinstance(sortino, float) and (math.isnan(sortino) or math.isinf(sortino)))) else sortino,
            "max_drawdown_pct": None if mdd is None else round(float(mdd) * 100.0, 2),
            "win_rate_pct": None if wr is None else round(float(wr) * 100.0, 2),
            "profit_factor": None if (pf is None or (isinstance(pf, float) and (math.isnan(pf) or math.isinf(pf)))) else pf,
        }

        # by-symbol (convert decimals to percents in JSON)
        by_symbol_metrics: Dict[str, Dict[str, Any]] = {}
        for sym, s in (perf.get("by_symbol") or {}).items():
            s_sharpe = s.get("sharpe")
            s_wr = s.get("win_rate")
            by_symbol_metrics[sym] = {
                "sharpe": None if (s_sharpe is None or (isinstance(s_sharpe, float) and (math.isnan(s_sharpe) or math.isinf(s_sharpe)))) else s_sharpe,
                "win_rate_pct": None if s_wr is None else round(float(s_wr) * 100.0, 1),
            }

        out = {
            "generated_at": _iso(datetime.now(timezone.utc)),
            "window_hours": int(os.getenv("MW_PERF_WINDOW_H", "72")),
            "overall": overall_metrics,
            "by_symbol": by_symbol_metrics,
            "demo": bool(ctx.is_demo),
        }
        (arts / "performance_metrics.json").write_text(json.dumps(out, indent=2))

        # ---- CI summary lines (clean & resilient) ----
        md.append(
            f"trades={overall_metrics.get('trades')} │ "
            f"Sharpe={_fmt(overall_metrics.get('sharpe'))} │ "
            f"Sortino={_fmt(overall_metrics.get('sortino'))} │ "
            f"MaxDD={_fmt(overall_metrics.get('max_drawdown_pct'), '{:.1f}%')} │ "
            f"Win={_fmt(overall_metrics.get('win_rate_pct'), '{:.1f}%')} │ "
            f"PF={_fmt(overall_metrics.get('profit_factor'))}"
        )

        bysym_parts = []
        for sym, s in by_symbol_metrics.items():
            s_sharpe = _fmt(s.get("sharpe"))
            s_wr = _fmt(s.get("win_rate_pct"), "{:.1f}%")
            bysym_parts.append(f"{sym}(S={s_sharpe}, WR={s_wr})")
        md.append("by symbol: " + ", ".join(bysym_parts))

    except Exception as e:
        md.append(f"\n> ❌ Performance Validation failed: {e}")