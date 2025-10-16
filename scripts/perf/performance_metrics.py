# scripts/perf/performance_metrics.py
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List

from scripts.summary_sections.common import _iso


def _to_dt(ts: str) -> datetime:
    if ts.endswith("Z"):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return datetime.fromisoformat(ts)


def _safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _safe_mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(var, 0.0))


def _downside_std(xs: List[float]) -> float:
    neg = [min(x, 0.0) for x in xs]
    if not any(neg):
        return 0.0
    mu = _safe_mean(xs)
    dd = [min(x - mu, 0.0) for x in xs]
    if len(dd) < 2:
        return 0.0
    var = sum(d * d for d in dd) / (len(dd) - 1)
    return math.sqrt(max(var, 0.0))


def _max_drawdown(equity: List[float]) -> float:
    peak = -1e18
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            mdd = min(mdd, dd)
    return mdd


def _profit_factor(trades: List[Dict[str, Any]]) -> float:
    gains = sum(max(t.get("pnl_frac", 0.0), 0.0) for t in trades)
    losses = -sum(min(t.get("pnl_frac", 0.0), 0.0) for t in trades)
    if losses <= 0:
        return gains if gains > 0 else 0.0
    return gains / losses


def compute_metrics(equity_series: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute finance-grade metrics from equity curve and trades.
    Expects equity_series sorted by ts with at least one initial point.
    """
    eq = [float(p["equity"]) for p in equity_series]
    ts = [_to_dt(p["ts"]) for p in equity_series]

    # Simple step returns from equity
    rets: List[float] = []
    for i in range(1, len(eq)):
        if eq[i - 1] != 0:
            rets.append(eq[i] / eq[i - 1] - 1.0)

    # Annualization (assume steps are irregular; approximate by per-step mean/std then scale by sqrt(N))
    n = len(rets)
    mu = _safe_mean(rets)
    sd = _std(rets)
    dsd = _downside_std(rets)

    sharpe = (mu / sd * math.sqrt(n)) if sd > 1e-12 and n > 1 else 0.0
    sortino = (mu / dsd * math.sqrt(n)) if dsd > 1e-12 and n > 1 else 0.0
    mdd = _max_drawdown(eq)

    # Wins / trades
    wins = sum(1 for t in trades if t.get("pnl_frac", 0.0) > 0)
    total = len(trades)
    win_rate = (wins / total) if total > 0 else 0.0
    pf = _profit_factor(trades)

    # CAGR (only meaningful if span >= ~30 days)
    cagr = None
    if ts and (ts[-1] - ts[0]).days >= 30 and eq[0] > 0:
        years = (ts[-1] - ts[0]).days / 365.25
        cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if years > 0 else None

    out = {
        "generated_at": _iso(datetime.now(timezone.utc).replace(microsecond=0)),
        "trades": total,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "cagr": cagr,
    }
    return out