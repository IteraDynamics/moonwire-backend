# scripts/perf/performance_metrics.py
from __future__ import annotations

import math
from typing import Dict, List, Any
import numpy as np

_EPS = 1e-12
_MAX_ABS_RATIO = 5.0  # clip extreme ratios (keeps CI output sane)

def _clean(x: float | None) -> float | None:
    if x is None:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    # clip absurd magnitudes to prevent ugly CI output
    if abs(x) > _MAX_ABS_RATIO:
        return None
    return float(x)

def compute_metrics(equity_series: np.ndarray, returns_series: np.ndarray, trades: List[Any]) -> Dict[str, Any]:
    equity_series = np.asarray(equity_series, dtype=float)
    returns_series = np.asarray(returns_series, dtype=float)

    sharpe = None
    sortino = None
    if returns_series.size > 2:
        mu = float(np.mean(returns_series))
        sd = float(np.std(returns_series))
        if sd > _EPS:
            sharpe = mu / sd
        downside = returns_series[returns_series < 0.0]
        if downside.size >= 2:
            ds = float(np.std(downside))
            if ds > _EPS:
                sortino = mu / ds

    maxdd = None
    if equity_series.size > 1:
        peak = np.maximum.accumulate(equity_series)
        dd = (equity_series - peak) / np.maximum(peak, _EPS)
        maxdd = float(np.min(dd))

    wins = losses = 0
    gw = gl = 0.0
    t_rets = []
    for t in trades:
        pnl = float(getattr(t, "pnl", 0.0))
        r = getattr(t, "pnl_pct", None)
        t_rets.append(0.0 if r is None else float(r))
        if pnl >= 0:
            wins += 1
            gw += pnl
        else:
            losses += 1
            gl += -pnl

    win_rate = None
    profit_factor = None
    avg_trade = None
    total = wins + losses
    if total > 0:
        win_rate = wins / total
        avg_trade = float(np.mean(t_rets))
        # Robust PF:
        # - If there are zero losses, PF is undefined → return None
        # - If losses extremely tiny relative to gains, treat as unstable → None
        if losses == 0:
            profit_factor = None
        else:
            if gl <= max(_EPS, 1e-9 * max(gw, 1.0)):
                profit_factor = None
            else:
                profit_factor = gw / gl

    exposure_pct = 1.0 if returns_series.size > 0 else None
    cagr = None  # not meaningful in short demo windows

    return {
        "sharpe": _clean(sharpe),
        "sortino": _clean(sortino),
        "max_drawdown": _clean(maxdd),
        "calmar": None,
        "win_rate": _clean(win_rate),
        "profit_factor": _clean(profit_factor),
        "avg_trade": _clean(avg_trade),
        "exposure_pct": _clean(exposure_pct),
        "cagr": _clean(cagr),
    }