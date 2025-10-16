# scripts/perf/performance_metrics.py
from __future__ import annotations
import math, json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple
import numpy as np

def _to_np(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    return a if a.size else np.asarray([0.0])

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def max_drawdown(equity: Sequence[float]) -> Tuple[float, float]:
    eq = _to_np(equity)
    if eq.size < 2:
        return 0.0, 0.0
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks.clip(min=1e-12)
    min_dd = dd.min()
    return float(min_dd), float(min_dd)  # pct (negative), same twice for compat

def sharpe(returns: Sequence[float], rf: float = 0.0, periods_per_year: int = 365*24) -> float:
    r = _to_np(returns) - rf / periods_per_year
    if r.std(ddof=1) == 0:
        return 0.0
    return float((r.mean() / r.std(ddof=1)) * math.sqrt(periods_per_year))

def sortino(returns: Sequence[float], rf: float = 0.0, periods_per_year: int = 365*24) -> float:
    r = _to_np(returns) - rf / periods_per_year
    downside = r[r < 0]
    denom = downside.std(ddof=1) if downside.size else 0.0
    if denom == 0:
        return 0.0
    return float((r.mean() / denom) * math.sqrt(periods_per_year))

def calmar(equity: Sequence[float], years: float) -> float:
    if years <= 0:
        return 0.0
    eq = _to_np(equity)
    if eq[0] <= 0:
        return 0.0
    ret = (eq[-1] / eq[0]) - 1.0
    mdd, _ = max_drawdown(eq)
    return float((ret / years) / abs(mdd) if mdd != 0 else 0.0)

def profit_factor(trade_pnls: Sequence[float]) -> float:
    gains = sum(x for x in trade_pnls if x > 0)
    losses = -sum(x for x in trade_pnls if x < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)

def win_rate(trade_pnls: Sequence[float]) -> float:
    t = list(trade_pnls)
    if not t:
        return 0.0
    wins = sum(1 for x in t if x > 0)
    return float(wins / len(t))

def cagr(equity: Sequence[float], hours: float) -> float | None:
    if hours < 24 * 30:  # ≥ ~30d only
        return None
    eq = _to_np(equity)
    if eq[0] <= 0:
        return None
    years = hours / (24 * 365)
    return float((eq[-1] / eq[0]) ** (1 / years) - 1.0)

def compute_metrics(
    equity_series: Sequence[float],
    returns_series: Sequence[float],
    trades: List[Dict],
    *,
    risk_free: float = 0.0,
    hours_span: float = 72.0,
) -> Dict:
    eq = _to_np(equity_series)
    rets = _to_np(returns_series)
    trade_pnls = [float(t.get("pnl", 0.0)) for t in trades]
    wr = win_rate(trade_pnls)
    pf = profit_factor(trade_pnls)
    mdd, _ = max_drawdown(eq if eq.size else [1.0, 1.0])
    sh = sharpe(rets, rf=risk_free)
    so = sortino(rets, rf=risk_free)
    cal = calmar(eq if eq.size else [1.0, 1.0], years=hours_span/(24*365))
    cagr_v = cagr(eq, hours_span)
    avg_tr = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    exposure = float(sum(1 for t in trades if t.get("open", False)) / len(trades)) if trades else 0.0
    return {
        "generated_at": _utc_now(),
        "trades": len(trades),
        "sharpe": sh,
        "sortino": so,
        "max_drawdown": float(mdd),
        "win_rate": wr,
        "profit_factor": pf,
        "avg_trade": avg_tr,
        "exposure": exposure,
        "calmar": cal,
        "cagr": cagr_v,
    }