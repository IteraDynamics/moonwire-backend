# scripts/summary_sections/live_backtest_section.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import os


def _is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")


def _fmt_float(x, nd=2, default="0.00"):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return default


def render(md: List[str], backtest: Optional[Dict[str, Any]] = None, window_hours: int = 24) -> None:
    """
    Render the 'Live Backtest' section.

    Parameters
    ----------
    md : list[str]
        The markdown buffer to append to.
    backtest : dict | None
        If provided, should look like:
        {
          "threshold": 0.5,                 # optional
          "overall": {"precision":..., "recall":..., "tp":..., "fp":..., "fn":...},  # optional
          "origins": {
             "twitter": {"precision":..., "recall":..., "tp":..., "fp":..., "fn":...},
             "reddit":  {...},
             ...
          }  # or "by_origin" instead of "origins"
        }
        If None/empty, we only print a demo fallback (in DEMO_MODE) or a 'no activity' line.
    window_hours : int
        Label for the header (cosmetic only).
    """
    md.append(f"\n### 🧪 Live Backtest ({int(window_hours)}h)")

    bt = backtest or {}
    # threshold selection: prefer explicit in backtest, else env override
    try:
        thr = bt.get("threshold")
        if thr is None:
            _env = os.getenv("TL_DECISION_THRESHOLD")
            thr = float(_env) if _env is not None else None
        thr_str = f" @thr={float(thr):.2f}" if thr is not None else ""
    except Exception:
        thr_str = ""

    # overall line (optional)
    overall = bt.get("overall") or {}
    if overall:
        try:
            md.append(
                f"- overall: precision={_fmt_float(overall.get('precision'))} | "
                f"recall={_fmt_float(overall.get('recall'))} "
                f"(tp={int(overall.get('tp', 0))}, fp={int(overall.get('fp', 0))}, "
                f"fn={int(overall.get('fn', 0))}){thr_str}"
            )
        except Exception:
            # if malformed, just skip
            pass

    # per-origin lines
    origins = bt.get("origins") or bt.get("by_origin") or {}
    printed = 0
    try:
        for org, stats in sorted(origins.items()):
            tp = int(stats.get("tp", 0) or 0)
            fp = int(stats.get("fp", 0) or 0)
            fn = int(stats.get("fn", 0) or 0)
            if (tp + fp + fn) == 0:
                continue
            if org == "unknown" and (tp + fp + fn) == 0:
                continue
            prec = _fmt_float(stats.get("precision"))
            rec  = _fmt_float(stats.get("recall"))
            md.append(f"- {org}: precision={prec} | recall={rec}")
            printed += 1
    except Exception:
        # ignore per-origin if structure is off
        pass

    # empty-state handling
    if printed == 0 and not overall:
        if _is_demo_mode():
            md.append("- twitter: precision=0.50 | recall=0.33 (demo)")
            md.append("- reddit: precision=0.40 | recall=0.25 (demo)")
        else:
            md.append("_No activity in the window._")
