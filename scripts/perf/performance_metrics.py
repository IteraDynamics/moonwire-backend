# scripts/perf/performance_metrics.py
from __future__ import annotations

import math
from typing import Dict, List, Any

import numpy as np

def _safe(x: float) -> float:
    if x is None or math.isnan(x) or math.isinf(x):
        return None
    return float(x)

def compute_metrics(equity_series: np.ndarray, returns_series: np.ndarray, trades: List[Any]) -> Dict[str, Any]:
    """
    equity_series: sequence of equity levels (>=1 point)
    returns_series: simple returns aligned with equity diff (len = len(equity)-1)
    trades: list of objects with .pnl and .pnl_pct
    """
    equity_series = np.asarray(equity_series, dtype=float)
    returns_series = np.asarray(returns_series, dtype=float)

    # Sharpe / Sortino on per-step returns (assume steps roughly ~minutes,
    # but we only need relative values for CI; no annualization here)
    rf = 0.0
    excess = returns_series - rf
    sharpe = None
    sortino = None
    if excess.size > 1 and np.std(excess) > 1e-12:
        sharpe = np.mean(excess) / (np.std(excess) + 1e-12)
    downside = excess[excess < 0.0]
    if downside.size > 0:
        sortino = np.mean(excess) / (np.std(downside) + 1e-12)

    # Max drawdown
    maxdd = None
    if equity_series.size > 0:
        peak = np.maximum.accumulate(equity_series)
        dd = (equity_series - peak) / (peak + 1e-12)
        maxdd = float(np.min(dd))

    # Profit factor, win rate, average trade
    wins, losses = 0, 0
    gross_win, gross_loss = 0.0, 0.0
    trade_returns = []
    for t in trades:
        r = getattr(t, "pnl_pct", None)
        trade_returns.append(r if r is not None else 0.0)
        pnl = getattr(t, "pnl", 0.0)
        if pnl >= 0:
            wins += 1
            gross_win += pnl
        else:
            losses += 1
            gross_loss += -pnl
    win_rate = None
    profit_factor = None
    avg_trade = None
    n = wins + losses
    if n > 0:
        win_rate = wins / n
        avg_trade = float(np.mean(trade_returns))
        if gross_loss > 0:
            profit_factor = gross_win / gross_loss
        elif gross_win > 0 and gross_loss == 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

    # Exposure estimate (% of steps with positions) — rough proxy in this simplified sim
    exposure_pct = None
    if returns_series.size > 0:
        # assume always in market when we simulated (since we exit at horizon per signal)
        exposure_pct = 1.0

    # CAGR — only meaningful if backtest window >= ~30d; we can compute generic CAGR anyway
    cagr = None
    if equity_series.size > 1 and equity_series[0] > 0:
        total_return = equity_series[-1] / equity_series[0]
        # assume ~N steps ~minutes; rough yearly scaling is not reliable in demo;
        # keep None unless someone runs real longer windows.
        cagr = None

    return {
        "sharpe": _safe(sharpe),
        "sortino": _safe(sortino),
        "max_drawdown": _safe(maxdd),
        "calmar": _safe(((-1.0 * (0 if maxdd in (None, 0) else maxdd)) and ( ( (equity_series[-1]/equity_series[0]) - 1.0 ) / abs(maxdd) )) if (maxdd not in (None, 0) and equity_series.size>1) else None),
        "win_rate": _safe(win_rate),
        "profit_factor": _safe(profit_factor),
        "avg_trade": _safe(avg_trade),
        "exposure_pct": _safe(exposure_pct),
        "cagr": _safe(cagr),
    }
