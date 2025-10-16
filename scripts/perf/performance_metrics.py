# scripts/perf/performance_metrics.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Dict, Any, Optional, Tuple


# ---- Helpers -----------------------------------------------------------------

_DT_FMT_GUESSES = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
)

def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in _DT_FMT_GUESSES:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # try naive ISO fallback
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _epsilon() -> float:
    # Small floor to avoid division blow-ups
    return 1e-8

def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def _stdev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(var, 0.0))

def _downside_stdev(xs: List[float]) -> float:
    # downside volatility uses only negative deviations from 0
    negs = [min(x, 0.0) for x in xs]
    # if all non-negative, stdev should be ~0 (protect with epsilon later)
    return _stdev(negs)

def _annualization_from_series(returns_series: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Best-effort annualization / period inference.
    Returns (ann_factor, rf_per_period_factor_days), where:
      - ann_factor multiplies Sharpe/Sortino (sqrt of periods per year)
      - rf_days is the length of each period in days (for risk-free adjustment)
    If timestamps are missing, return (1.0, None) meaning "no annualization".
    """
    # We expect either [{"ts": iso, "ret": x}, ...] or [{"ts": iso, "value": x}] or raw floats elsewhere.
    ts = []
    for r in returns_series:
        ts_val = r.get("ts") if isinstance(r, dict) else None
        if ts_val:
            dt = _parse_dt(ts_val)
            if dt:
                ts.append(dt)
    if len(ts) < 2:
        return 1.0, None

    ts_sorted = sorted(ts)
    total_days = (ts_sorted[-1] - ts_sorted[0]).total_seconds() / 86400.0
    n = len(ts_sorted)
    if total_days <= 0 or n < 2:
        return 1.0, None

    period_days = total_days / (n - 1)
    # periods per year
    ppy = max(365.0 / max(period_days, _epsilon()), 1.0)
    ann_factor = math.sqrt(ppy)
    return ann_factor, period_days


# ---- Public API ---------------------------------------------------------------

def compute_metrics(
    equity_series: Iterable[Dict[str, Any]],
    returns_series: Iterable[Any],
    trades: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute finance-grade metrics from equity / returns / trades.

    Inputs:
      - equity_series: iterable of {"ts": ISO, "equity": float}
      - returns_series: iterable of either floats or {"ts": ISO, "ret": float}
      - trades: iterable of {"pnl": float, "pnl_pct": float, "side": "long"/"short", "entry_ts": ISO?, "exit_ts": ISO?}

    Outputs (minimal stable surface):
      {
        "trades": <int>,
        "sharpe": <float>,
        "sortino": <float>,
        "max_drawdown": <float>,            # as fraction, negative value (e.g., -0.12)
        "win_rate": <float>,                # 0..1
        "profit_factor": <float> or null,
        "avg_trade": <float>,               # mean pnl (currency)
        "avg_trade_pct": <float>,           # mean pnl_pct (fraction)
        "exposure_pct": <float or null>,    # 0..1 if entry/exit present
        "cagr": <float or null>,            # if enough span (>=30d), else null
      }
    """
    # Normalize inputs
    eq = []
    for e in equity_series:
        if isinstance(e, dict) and "equity" in e:
            eq.append({"ts": e.get("ts"), "equity": float(e["equity"])})
    rets = []
    for r in returns_series:
        if isinstance(r, dict) and "ret" in r:
            rets.append({"ts": r.get("ts"), "ret": float(r["ret"])})
        elif isinstance(r, dict) and "value" in r:
            rets.append({"ts": r.get("ts"), "ret": float(r["value"])})
        else:
            # raw float fallback
            try:
                rets.append({"ts": None, "ret": float(r)})
            except Exception:
                pass
    tr = []
    for t in trades:
        d = {k: t.get(k) for k in ("pnl", "pnl_pct", "side", "entry_ts", "exit_ts")}
        if d["pnl"] is not None:
            d["pnl"] = float(d["pnl"])
        if d["pnl_pct"] is not None:
            d["pnl_pct"] = float(d["pnl_pct"])
        tr.append(d)

    # Metrics from returns
    returns = [x["ret"] for x in rets if isinstance(x.get("ret"), (int, float))]
    mean_ret = _mean(returns) if returns else 0.0
    vol = max(_stdev(returns), _epsilon())
    dvol = max(_downside_stdev(returns), _epsilon())

    ann_factor, period_days = _annualization_from_series(rets)
    # Risk-free (annual) -> per-period
    rf_annual = float(os.getenv("MW_PERF_RISK_FREE", "0.0"))
    if period_days is None:
        rf_per_period = 0.0
    else:
        rf_per_period = rf_annual * (max(period_days, 0.0) / 365.0)

    # Sharpe/Sortino
    sharpe = ((mean_ret - rf_per_period) / vol) * ann_factor if returns else float("nan")
    sortino = ((mean_ret - rf_per_period) / dvol) * ann_factor if returns else float("nan")

    # Max drawdown from equity
    def _max_dd(es: List[Dict[str, Any]]) -> float:
        if not es:
            return float("nan")
        run_max = -float("inf")
        max_dd = 0.0
        for row in es:
            v = float(row["equity"])
            run_max = max(run_max, v)
            dd = (v / run_max) - 1.0 if run_max > 0 else 0.0
            max_dd = min(max_dd, dd)
        return max_dd  # negative number
    max_dd = _max_dd(eq)

    # Win rate, profit factor, averages
    wins = [t for t in tr if (t.get("pnl") or 0.0) > 0]
    losses = [t for t in tr if (t.get("pnl") or 0.0) < 0]
    win_rate = (len(wins) / len(tr)) if tr else float("nan")
    gross_profit = sum(x["pnl"] for x in wins) if wins else 0.0
    gross_loss = abs(sum(x["pnl"] for x in losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > _epsilon() else (float("inf") if gross_profit > 0 else float("nan"))

    avg_trade = _mean([t["pnl"] for t in tr if t.get("pnl") is not None]) if tr else float("nan")
    avg_trade_pct = _mean([t["pnl_pct"] for t in tr if t.get("pnl_pct") is not None]) if tr else float("nan")

    # Exposure % (best-effort)
    def _exposure_pct(ts: List[Dict[str, Any]]) -> Optional[float]:
        spans = []
        for t in ts:
            et = _parse_dt(t.get("entry_ts") or "")
            xt = _parse_dt(t.get("exit_ts") or "")
            if et and xt and xt > et:
                spans.append((et, xt))
        if not spans:
            return None
        start = min(s for s, _ in spans)
        end = max(e for _, e in spans)
        total = (end - start).total_seconds()
        if total <= 0:
            return None
        held = sum((e - s).total_seconds() for s, e in spans)
        return max(0.0, min(1.0, held / total))

    exposure = _exposure_pct(tr)

    # CAGR if we have >=30d span
    def _cagr(es: List[Dict[str, Any]]) -> Optional[float]:
        if not es:
            return None
        dts = [(_parse_dt(r.get("ts") or ""), float(r["equity"])) for r in es]
        dts = [(dt, v) for dt, v in dts if dt]
        if len(dts) < 2:
            return None
        dts.sort(key=lambda x: x[0])
        days = (dts[-1][0] - dts[0][0]).days
        if days < 30:
            return None
        start_v, end_v = dts[0][1], dts[-1][1]
        if start_v <= 0 or end_v <= 0:
            return None
        years = days / 365.0
        return (end_v / start_v) ** (1.0 / years) - 1.0

    cagr = _cagr(eq)

    return {
        "trades": len(tr),
        "sharpe": float(sharpe) if not math.isnan(sharpe) else None,
        "sortino": float(sortino) if not math.isnan(sortino) else None,
        "max_drawdown": float(max_dd) if not math.isnan(max_dd) else None,
        "win_rate": float(win_rate) if not math.isnan(win_rate) else None,
        "profit_factor": float(profit_factor) if not math.isnan(profit_factor) else None,
        "avg_trade": float(avg_trade) if not math.isnan(avg_trade) else None,
        "avg_trade_pct": float(avg_trade_pct) if not math.isnan(avg_trade_pct) else None,
        "exposure_pct": exposure,
        "cagr": cagr,
    }