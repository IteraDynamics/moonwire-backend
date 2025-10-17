import math
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional

# equity_series: list[(ts: str|datetime, equity: float)]
# returns_series: np.array of per-period returns (simple returns, not log)
# trades: list[dict] with "pnl", "pnl_pct", "closed": bool

def _to_dt(ts):
    if isinstance(ts, dt.datetime):
        return ts
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)

def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / np.where(peaks == 0, 1.0, peaks)
    return float(dd.min())  # negative number

def _sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 365*24) -> float:
    if returns.size < 2:
        return float("nan")
    ex = returns - (rf / periods_per_year)
    s = ex.mean() / (ex.std(ddof=1) + 1e-12)
    return float(s * math.sqrt(periods_per_year))

def _sortino(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 365*24) -> float:
    if returns.size < 2:
        return float("nan")
    ex = returns - (rf / periods_per_year)
    downside = ex[ex < 0.0]
    denom = downside.std(ddof=1) if downside.size > 1 else 0.0
    if denom == 0.0:
        return float("inf") if ex.mean() > 0 else float("-inf")
    return float((ex.mean() / denom) * math.sqrt(periods_per_year))

def _profit_factor(trades: List[Dict]) -> float:
    gains = sum(max(t.get("pnl", 0.0), 0.0) for t in trades if t.get("closed"))
    losses = -sum(min(t.get("pnl", 0.0), 0.0) for t in trades if t.get("closed"))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)

def _win_rate(trades: List[Dict]) -> float:
    closed = [t for t in trades if t.get("closed")]
    if not closed:
        return float("nan")
    wins = sum(1 for t in closed if t.get("pnl", 0.0) > 0.0)
    return wins / len(closed)

def _avg_trade(trades: List[Dict]) -> float:
    closed = [t for t in trades if t.get("closed")]
    if not closed:
        return float("nan")
    return float(np.mean([t.get("pnl_pct", 0.0) for t in closed]))

def _exposure_pct(trades: List[Dict], start: dt.datetime, end: dt.datetime) -> float:
    if start >= end:
        return 0.0
    total_secs = (end - start).total_seconds()
    live_secs = 0.0
    for t in trades:
        et = t.get("entry_ts")
        xt = t.get("exit_ts")
        if not et or not xt:
            continue
        et = _to_dt(et)
        xt = _to_dt(xt)
        live_secs += max(0.0, (xt - et).total_seconds())
    return (live_secs / total_secs) if total_secs > 0 else 0.0

def _cagr(equity_series: List[Tuple[str, float]]) -> Optional[float]:
    if len(equity_series) < 2:
        return None
    t0 = _to_dt(equity_series[0][0])
    t1 = _to_dt(equity_series[-1][0])
    days = (t1 - t0).days
    if days < 30:
        return None
    v0 = float(equity_series[0][1])
    v1 = float(equity_series[-1][1])
    if v0 <= 0 or v1 <= 0:
        return None
    years = days / 365.0
    return (v1 / v0) ** (1.0 / years) - 1.0

def compute_metrics(
    equity_series: List[Tuple[str, float]],
    returns_series: np.ndarray,
    trades: List[Dict],
    risk_free: float = 0.0,
    periods_per_year: int = 365*24,
) -> Dict:
    equity_np = np.array([v for _, v in equity_series], dtype=float)
    maxdd = _max_drawdown(equity_np)

    agg = {
        "trades": int(sum(1 for t in trades if t.get("closed"))),
        "sharpe": _sharpe(returns_series, rf=risk_free, periods_per_year=periods_per_year),
        "sortino": _sortino(returns_series, rf=risk_free, periods_per_year=periods_per_year),
        "max_drawdown": maxdd,  # negative fraction
        "win_rate": _win_rate(trades),
        "profit_factor": _profit_factor(trades),
        "avg_trade": _avg_trade(trades),
    }

    # Exposure & CAGR if possible
    if equity_series:
        start = _to_dt(equity_series[0][0])
        end = _to_dt(equity_series[-1][0])
        agg["exposure_pct"] = _exposure_pct(trades, start, end)
        agg["cagr"] = _cagr(equity_series)

    return agg
