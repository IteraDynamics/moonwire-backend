# scripts/perf/performance_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, isfinite
from typing import Dict, List, Iterable, Optional, Tuple
from datetime import datetime, timezone

import statistics as stats


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_std(xs: List[float]) -> float:
    return stats.pstdev(xs) if len(xs) > 1 else 0.0


def _downside_std(xs: List[float]) -> float:
    downs = [x for x in xs if x < 0]
    return stats.pstdev(downs) if len(downs) > 1 else 0.0


def _max_drawdown(equity: List[Tuple[str, float]]) -> float:
    """
    equity: list of (ts_iso, equity_value) sorted by time
    returns peak-to-trough drawdown as a **ratio** (negative number, e.g. -0.12)
    """
    if not equity:
        return 0.0
    peak = equity[0][1]
    max_dd = 0.0
    for _, v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _period_days(start_iso: str, end_iso: str) -> float:
    s = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    e = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    return max((e - s).total_seconds() / 86400.0, 0.0)


def compute_metrics(
    equity_series: List[Tuple[str, float]],
    returns_series: List[float],
    trades: List[Dict],
    *,
    risk_free: float = 0.0,
    mode: str = "backtest",
) -> Dict:
    """
    equity_series: [(ts_iso, equity_value)], sorted
    returns_series: fractional returns per step (we treat per-trade returns here)
    trades: list of trade dicts: {entry_ts, exit_ts, symbol, side, entry, exit, pnl, pnl_pct}

    Returns dict with aggregate + per-symbol metrics.
    """
    generated_at = _utc_now_iso()
    agg_trades = len(trades)

    # Basic aggregates from returns
    sharpe = None
    sortino = None
    win_rate = None
    profit_factor = None
    avg_trade = None
    exposure_pct = None
    cagr = None
    max_dd = _max_drawdown(equity_series)

    if returns_series:
        mu = stats.fmean(returns_series)
        sigma = _safe_std(returns_series)
        d_sigma = _downside_std(returns_series)

        if sigma > 0:
            sharpe = (mu - risk_free) / sigma
        if d_sigma > 0:
            sortino = (mu - risk_free) / d_sigma

        wins = [r for r in returns_series if r > 0]
        losses = [-r for r in returns_series if r < 0]
        win_rate = len(wins) / len(returns_series) if returns_series else None
        profit_factor = (sum(wins) / sum(losses)) if losses else None
        avg_trade = mu

    # Exposure: time in market / backtest window
    if equity_series and trades:
        t_start, _ = equity_series[0]
        t_end, _ = equity_series[-1]
        total_min = _period_days(t_start, t_end) * 24 * 60
        # sum of trade durations in minutes
        mins = 0.0
        for t in trades:
            try:
                s = datetime.fromisoformat(t["entry_ts"].replace("Z", "+00:00"))
                e = datetime.fromisoformat(t["exit_ts"].replace("Z", "+00:00"))
                mins += max((e - s).total_seconds() / 60.0, 0.0)
            except Exception:
                pass
        exposure_pct = (mins / total_min) if total_min > 0 else None

        # CAGR only if ~30d+ period and backtest mode
        days = _period_days(t_start, t_end)
        if mode == "backtest" and days >= 30:
            try:
                v0 = equity_series[0][1]
                v1 = equity_series[-1][1]
                if v0 > 0 and isfinite(v1):
                    cagr = (v1 / v0) ** (365.0 / days) - 1.0
            except Exception:
                cagr = None

    # Per-symbol rollups (using trades)
    by_symbol: Dict[str, Dict] = {}
    for t in trades:
        sym = t.get("symbol", "UNK")
        d = by_symbol.setdefault(sym, {"trades": 0, "wins": 0, "rets": [], "gains": 0.0, "losses": 0.0})
        d["trades"] += 1
        r = float(t.get("pnl_pct", 0.0))
        d["rets"].append(r)
        if r > 0:
            d["wins"] += 1
            d["gains"] += r
        elif r < 0:
            d["losses"] += -r

    per_symbol_out: Dict[str, Dict] = {}
    for sym, d in by_symbol.items():
        rets = d["rets"]
        s_mu = stats.fmean(rets) if rets else 0.0
        s_sigma = _safe_std(rets)
        s_dsigma = _downside_std(rets)
        s_sharpe = (s_mu - risk_free) / s_sigma if s_sigma > 0 else None
        s_sortino = (s_mu - risk_free) / s_dsigma if s_dsigma > 0 else None
        s_wr = (d["wins"] / d["trades"]) if d["trades"] else None
        s_pf = (d["gains"] / d["losses"]) if d["losses"] > 0 else None
        per_symbol_out[sym] = {
            "trades": d["trades"],
            "sharpe": s_sharpe,
            "sortino": s_sortino,
            "win_rate": s_wr,
            "profit_factor": s_pf,
            "avg_trade": s_mu,
            "cagr": None,  # not meaningful per-symbol over short windows here
        }

    aggregate = {
        "trades": agg_trades,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "exposure_pct": exposure_pct,
        "cagr": cagr,
    }

    return {
        "generated_at": generated_at,
        "aggregate": aggregate,
        "by_symbol": per_symbol_out,
    }