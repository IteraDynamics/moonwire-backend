# scripts/summary_sections/live_backtest.py
from __future__ import annotations
import os
from scripts.summary_sections.common import SummaryContext

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🧪 Live Backtest (24h)")
    bt = (ctx.caches.get("live_backtest") or ctx.caches.get("backtest") or {}) or {}

    # Optional threshold display
    try:
        _th = bt.get("threshold")
        if _th is None:
            _th_env = os.getenv("TL_DECISION_THRESHOLD")
            _th = float(_th_env) if _th_env is not None else None
        _th_str = f" @thr={float(_th):.2f}" if _th is not None else ""
    except Exception:
        _th_str = ""

    overall = bt.get("overall") or {}
    if overall:
        try:
            md.append(
                f"- overall: precision={float(overall.get('precision', 0.0)):.2f} | "
                f"recall={float(overall.get('recall', 0.0)):.2f} "
                f"(tp={int(overall.get('tp', 0))}, fp={int(overall.get('fp', 0))}, fn={int(overall.get('fn', 0))}){_th_str}"
            )
        except Exception:
            pass

    by_origin = (bt.get("origins") or bt.get("by_origin") or {}) or {}
    printed = 0
    for org, stats in sorted(by_origin.items()):
        tp = int(stats.get("tp", 0) or 0)
        fp = int(stats.get("fp", 0) or 0)
        fn = int(stats.get("fn", 0) or 0)
        if (tp + fp + fn) == 0:
            continue
        if org == "unknown" and (tp + fp + fn) == 0:
            continue
        try:
            prec = float(stats.get("precision", 0.0) or 0.0)
            rec  = float(stats.get("recall", 0.0) or 0.0)
            md.append(f"- {org}: precision={prec:.2f} | recall={rec:.2f}")
            printed += 1
        except Exception:
            continue

    if printed == 0 and not overall:
        if os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
            md.append("- twitter: precision=0.50 | recall=0.33 (demo)")
            md.append("- reddit: precision=0.40 | recall=0.25 (demo)")
        else:
            md.append("_No activity in the window._")