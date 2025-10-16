# scripts/perf/performance_metrics.py
from __future__ import annotations
import math, numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Sequence

EPS = 1e-9


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _annualization_factor(timestamps: List[str]) -> float:
    """Estimate sampling frequency for Sharpe/Sortino scaling."""
    if len(timestamps) < 2:
        return 1.0
    ts = np.array([np.datetime64(t) for t in timestamps])
    dt_sec = np.median((ts[1:] - ts[:-1]).astype("timedelta64[s]").astype(float))
    if dt_sec <= 0:
        return 1.0
    samples_per_year = (365.25 * 24 * 3600) / dt_sec
    return math.sqrt(samples_per_year)


def _max_drawdown(equity: Sequence[float]) -> float:
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / np.maximum(peaks, EPS)
    return float(dd.min())  # negative fraction


def compute_metrics(equity_series: List[Dict[str, float]], trades: List[Dict]) -> Dict:
    """
    Compute Sharpe, Sortino, MaxDD, Win rate, Profit factor, Avg trade.
    equity_series: [{ts, equity}]
    trades: [{pnl, pnl_frac, ...}]
    """
    ts = [r["ts"] for r in equity_series]
    eq = np.array([r["equity"] for r in equity_series], dtype=float)
    if len(eq) >= 2:
        rets = (eq[1:] / np.maximum(eq[:-1], EPS)) - 1.0
    else:
        rets = np.array([], dtype=float)

    if rets.size == 0:
        sharpe = sortino = 0.0
    else:
        mu = rets.mean()
        sigma = max(rets.std(ddof=1), EPS)
        downside = rets[rets < 0]
        downside_std = max(downside.std(ddof=1) if downside.size else 0.0, EPS)
        af = _annualization_factor(ts)
        sharpe = (mu / sigma) * af
        sortino = (mu / downside_std) * af

    mdd = _max_drawdown(eq)
    n_trades = len(trades)
    if n_trades:
        wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        win_rate = wins / n_trades
        gross_profit = sum(max(t.get("pnl", 0.0), 0.0) for t in trades)
        gross_loss = sum(-min(t.get("pnl", 0.0), 0.0) for t in trades)
        profit_factor = gross_profit / max(gross_loss, EPS) if gross_profit > 0 else 0.0
        avg_trade = (gross_profit - gross_loss) / n_trades
    else:
        win_rate = profit_factor = avg_trade = 0.0

    return {
        "generated_at": _utc_now(),
        "trades": n_trades,
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "max_drawdown": round(float(mdd), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 2),
        "avg_trade": round(float(avg_trade), 6),
    }