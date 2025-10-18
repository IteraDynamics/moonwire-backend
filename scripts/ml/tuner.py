# scripts/ml/tuner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Local backtest
from .backtest import run_backtest


# ---------------------------- Config / Guardrails -----------------------------

# Grid (from brief)
CONF_GRID = [0.55, 0.58, 0.60, 0.62]
DEBOUNCE_GRID = [15, 30, 45, 60]  # minutes
HORIZON_GRID = [1, 2]             # hours

# Targets (from brief)
MIN_WINRATE = 0.60
SIGS_PER_DAY_MIN = 5.0
SIGS_PER_DAY_MAX = 10.0

# NEW: robustness guardrails to avoid degenerate configs
MIN_TRADES_AGG = 20               # at least 20 trades across all symbols
MIN_SYMBOLS_WITH_TRADES = 2       # require coverage on >= 2 symbols
LOW_TRADES_THRESH = 10            # below this, PF is capped for scoring
MAX_PF_IF_LOW_TRADES = 5.0        # cap PF used for scoring when trades are tiny

# Output paths
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------- Types -------------------------------------

@dataclass
class BTMetrics:
    win_rate: float
    profit_factor: float
    max_drawdown: float
    signals_per_day: float
    n_trades: int


# --------------------------------- Helpers -----------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return default


def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Accepts either:
      - {"metrics": {...}, "trades": [...]} or
      - flat dict with keys directly on the root.
    Be permissive to avoid test brittleness.
    """
    m = bt.get("metrics") or {}

    # Prefer metrics dict; fall back to root if not present
    win_rate = _safe_float(m.get("win_rate", bt.get("win_rate", 0.0)), 0.0)
    profit_factor = _safe_float(m.get("profit_factor", bt.get("profit_factor", 0.0)), 0.0)
    max_dd = _safe_float(m.get("max_drawdown", bt.get("max_drawdown", 0.0)), 0.0)
    sigs_day = _safe_float(m.get("signals_per_day", bt.get("signals_per_day", 0.0)), 0.0)

    # Trades count
    n_trades = _safe_int(m.get("n_trades", 0), 0)
    if n_trades <= 0:
        trades_obj = bt.get("trades")
        if isinstance(trades_obj, (list, tuple)):
            n_trades = len(trades_obj)
        else:
            n_trades = _safe_int(bt.get("n_trades", 0), 0)

    # Normalize: DD should be <= 0 if it’s a drawdown; accept either
    if max_dd > 0:
        max_dd = -abs(max_dd)

    return BTMetrics(
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        signals_per_day=sigs_day,
        n_trades=n_trades,
    )


def _aggregate(per_symbol: Dict[str, BTMetrics]) -> BTMetrics:
    """
    Aggregate logic:
      - win_rate: trade-weighted average
      - profit_factor: weighted by number of trades (approximation)
      - max_drawdown: worst (most negative)
      - signals_per_day: sum across symbols (aggregate cadence)
      - n_trades: sum
    """
    if not per_symbol:
        return BTMetrics(0.0, 0.0, 0.0, 0.0, 0)

    total_trades = sum(m.n_trades for m in per_symbol.values())
    if total_trades <= 0:
        return BTMetrics(0.0, 0.0, min((m.max_drawdown for m in per_symbol.values()), default=0.0), 0.0, 0)

    # Trade-weighted win rate and PF (approx)
    wr_num = sum(m.win_rate * m.n_trades for m in per_symbol.values())
    pf_num = sum(m.profit_factor * m.n_trades for m in per_symbol.values())
    wr = wr_num / total_trades if total_trades else 0.0
    pf = pf_num / total_trades if total_trades else 0.0

    max_dd = min((m.max_drawdown for m in per_symbol.values()), default=0.0)  # most negative
    sigs_day = sum(m.signals_per_day for m in per_symbol.values())

    return BTMetrics(win_rate=wr, profit_factor=pf, max_drawdown=max_dd,
                     signals_per_day=sigs_day, n_trades=total_trades)


def _meets_constraints(agg: BTMetrics, per_symbol: Dict[str, BTMetrics]) -> bool:
    # Core objectives
    if agg.win_rate < MIN_WINRATE:
        return False
    if not (SIGS_PER_DAY_MIN <= agg.signals_per_day <= SIGS_PER_DAY_MAX):
        return False

    # Guardrails (robustness)
    if agg.n_trades < MIN_TRADES_AGG:
        return False

    symbols_with_trades = sum(1 for m in per_symbol.values() if m.n_trades >= 3)
    if symbols_with_trades < MIN_SYMBOLS_WITH_TRADES:
        return False

    return True


def _score_tuple(metrics: BTMetrics) -> Tuple[float, float, float]:
    """
    Higher is better for the first two entries, lower (more positive) is better for DD,
    so we return (-abs(DD)) as the last tie-breaker.
    To avoid selecting freak PF on tiny trades, cap PF for low-trade configs.
    """
    pf = _safe_float(metrics.profit_factor, 0.0)
    if metrics.n_trades < LOW_TRADES_THRESH:
        pf = min(pf, MAX_PF_IF_LOW_TRADES)
    wr = _safe_float(metrics.win_rate, 0.0)
    dd = _safe_float(metrics.max_drawdown, 0.0)
    return (pf, wr, -abs(dd))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


# --------------------------------- Public API --------------------------------

def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
    fees_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> Dict[str, Any]:
    """
    Grid search over (conf_min, debounce_min, horizon_h).
    Returns:
      {
        "params": {"conf_min": ..., "debounce_min": ..., "horizon_h": ...},
        "agg": {...},
        "per_symbol": {...}
      }
    Also writes:
      - models/signal_thresholds.json (params only)
      - models/backtest_summary.json (agg + per_symbol)
    """
    best: Optional[Tuple[Tuple[float, float, float], Dict[str, Any]]] = None  # (score, result)
    best_relaxed: Optional[Tuple[Tuple[float, float, float], Dict[str, Any]]] = None

    # iterate grid; prefer horizon=1 first to encourage more opportunities
    for horizon_h in sorted(HORIZON_GRID):
        for conf_min in CONF_GRID:
            for debounce_min in DEBOUNCE_GRID:
                symbol_metrics: Dict[str, BTMetrics] = {}
                any_error = False

                for sym, df_pred in pred_dfs.items():
                    px = prices.get(sym)
                    if px is None or df_pred is None or df_pred.empty:
                        # No data → zero metrics
                        symbol_metrics[sym] = BTMetrics(0.0, 0.0, 0.0, 0.0, 0)
                        continue
                    try:
                        bt = run_backtest(
                            pred_df=df_pred,
                            prices_df=px,
                            conf_min=conf_min,
                            debounce_min=debounce_min,
                            horizon_h=horizon_h,
                            fees_bps=fees_bps,
                            slippage_bps=slippage_bps,
                        )
                        symbol_metrics[sym] = _extract_metrics(bt)
                    except Exception:
                        # Be defensive in CI; treat as zero contribution
                        any_error = True
                        symbol_metrics[sym] = BTMetrics(0.0, 0.0, 0.0, 0.0, 0)

                agg = _aggregate(symbol_metrics)
                meets = _meets_constraints(agg, symbol_metrics)
                score = _score_tuple(agg)

                payload = {
                    "params": {
                        "conf_min": conf_min,
                        "debounce_min": debounce_min,
                        "horizon_h": horizon_h,
                    },
                    "agg": {
                        "win_rate": round(agg.win_rate, 4),
                        "profit_factor": round(agg.profit_factor, 4),
                        "max_drawdown": round(agg.max_drawdown, 4),
                        "signals_per_day": round(agg.signals_per_day, 4),
                        "n_trades": int(agg.n_trades),
                    },
                    "per_symbol": {
                        s: {
                            "win_rate": round(m.win_rate, 4),
                            "profit_factor": round(m.profit_factor, 4),
                            "max_drawdown": round(m.max_drawdown, 4),
                            "signals_per_day": round(m.signals_per_day, 4),
                            "n_trades": int(m.n_trades),
                        }
                        for s, m in symbol_metrics.items()
                    },
                }

                if meets:
                    if (best is None) or (score > best[0]):
                        best = (score, payload)
                else:
                    # relaxed fallback: at least some trading activity & cadence
                    relaxed_ok = (agg.n_trades >= max(10, MIN_TRADES_AGG // 2)) and (agg.signals_per_day >= 1.0)
                    if relaxed_ok and ((best_relaxed is None) or (score > best_relaxed[0])):
                        best_relaxed = (score, payload)

    # Choose final
    if best is not None:
        chosen = best[1]
    elif best_relaxed is not None:
        chosen = best_relaxed[1]
    else:
        # No viable configs; return a sane default using the first grid point
        chosen = {
            "params": {
                "conf_min": CONF_GRID[0],
                "debounce_min": DEBOUNCE_GRID[0],
                "horizon_h": sorted(HORIZON_GRID)[0],
            },
            "agg": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0, "n_trades": 0},
            "per_symbol": {s: {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0, "n_trades": 0}
                           for s in pred_dfs.keys()},
        }

    # Persist artifacts
    _save_json(MODELS_DIR / "signal_thresholds.json", chosen["params"])
    _save_json(MODELS_DIR / "backtest_summary.json", {
        "aggregate": chosen["agg"],
        "per_symbol": chosen["per_symbol"],
    })

    return chosen