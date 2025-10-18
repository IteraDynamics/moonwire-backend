# scripts/ml/tuner.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_SUMMARY_PATH = MODELS_DIR / "backtest_summary.json"
THRESHOLDS_PATH = MODELS_DIR / "signal_thresholds.json"
TRADES_LOG_PATH = LOGS_DIR / "trades.jsonl"

from scripts.ml.backtest import run_backtest  # type: ignore


@dataclass
class BTMetrics:
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    signals_per_day: float = 0.0
    wins: int = 0
    losses: int = 0


# -----------------------
# Helpers
# -----------------------

def _as_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, (list, tuple)):
            return len(x)
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _trade_to_dict(t: Any) -> Dict[str, Any]:
    """Normalize a trade (dict or object) to a JSON-serializable dict."""
    if isinstance(t, dict):
        return t
    # object-like: try common attrs; fall back to __dict__ (if safe)
    keys = [
        "ts", "entry_ts", "exit_ts", "symbol", "side",
        "entry_px", "exit_px", "pnl", "pnl_pct", "size",
        "conf", "horizon_h", "fees", "slippage",
    ]
    d: Dict[str, Any] = {}
    for k in keys:
        if hasattr(t, k):
            d[k] = getattr(t, k)
    # if still empty, last resort
    if not d and hasattr(t, "__dict__"):
        try:
            d = dict(t.__dict__)
        except Exception:
            d = {}
    return d


def _extract_pnl(trade: Any) -> float:
    """Get pnl from a dict or object trade."""
    if isinstance(trade, dict):
        return _as_float(trade.get("pnl", 0.0), 0.0)
    if hasattr(trade, "pnl"):
        return _as_float(getattr(trade, "pnl"), 0.0)
    # try pnl_pct as a proxy
    if isinstance(trade, dict):
        return _as_float(trade.get("pnl_pct", 0.0), 0.0)
    if hasattr(trade, "pnl_pct"):
        return _as_float(getattr(trade, "pnl_pct"), 0.0)
    return 0.0


def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Extract consistent metrics from run_backtest() outputs.
    Supports nested 'metrics' dict, and trades as list[int/dict/object] or count.
    """
    m = bt.get("metrics", {})

    # n_trades
    n_trades = _as_int(m.get("n_trades", bt.get("trades", 0)))
    if n_trades == 0:
        n_trades = _as_int(bt.get("trades", 0), 0)

    # simple scalars
    win_rate = _as_float(bt.get("win_rate", m.get("win_rate", 0.0)), 0.0)
    profit_factor = _as_float(bt.get("profit_factor", m.get("profit_factor", 0.0)), 0.0)
    max_drawdown = _as_float(bt.get("max_drawdown", m.get("max_drawdown", 0.0)), 0.0)
    spd = _as_float(bt.get("signals_per_day", m.get("signals_per_day", 0.0)), 0.0)

    wins = _as_int(m.get("wins", bt.get("wins", 0)), 0)
    losses = _as_int(m.get("losses", bt.get("losses", 0)), 0)

    # If trades list present, recompute wins/losses if missing
    trades_obj = bt.get("trades", None)
    if isinstance(trades_obj, (list, tuple)) and len(trades_obj) > 0:
        if wins == 0 and losses == 0:
            w = 0
            l = 0
            for t in trades_obj:
                pnl = _extract_pnl(t)
                if pnl > 0:
                    w += 1
                elif pnl < 0:
                    l += 1
            wins, losses = w, l
        if n_trades == 0:
            n_trades = len(trades_obj)
        if n_trades > 0 and win_rate == 0.0 and (wins + losses) > 0:
            win_rate = wins / max(1, wins + losses)

    return BTMetrics(
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        signals_per_day=spd,
        wins=wins,
        losses=losses,
    )


def _weighted_average(values: List[float], weights: List[int]) -> float:
    if not values or not weights or sum(weights) == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w[w < 0] = 0
    denom = w.sum()
    if denom <= 0:
        return 0.0
    return float((v * w).sum() / denom)


def _aggregate_metrics(per_symbol: Dict[str, BTMetrics]) -> Dict[str, float]:
    if not per_symbol:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "signals_per_day": 0.0,
        }
    n_trades = [m.n_trades for m in per_symbol.values()]
    wrs = [m.win_rate for m in per_symbol.values()]
    pfs = [m.profit_factor for m in per_symbol.values()]
    mdds = [m.max_drawdown for m in per_symbol.values()]
    spd = [m.signals_per_day for m in per_symbol.values()]

    return {
        "n_trades": int(sum(n_trades)),
        "win_rate": _weighted_average(wrs, n_trades),
        "profit_factor": _weighted_average(pfs, n_trades) if sum(n_trades) > 0 else float(np.mean(pfs)),
        "max_drawdown": float(np.mean(mdds)) if mdds else 0.0,
        "signals_per_day": float(np.sum(spd)),
    }


def _penalty(agg: Dict[str, float], target_wr: float = 0.60, min_spd: float = 5.0, max_spd: float = 10.0) -> float:
    wr = agg.get("win_rate", 0.0)
    spd = agg.get("signals_per_day", 0.0)
    pf = agg.get("profit_factor", 0.0)
    mdd = abs(agg.get("max_drawdown", 0.0))

    wr_gap = max(0.0, target_wr - wr)
    spd_gap = 0.0
    if spd < min_spd:
        spd_gap = (min_spd - spd)
    elif spd > max_spd:
        spd_gap = (spd - max_spd)

    score = 10.0 * wr_gap + 2.0 * spd_gap + 1.0 * mdd + 1.0 * max(0.0, 1.0 - pf)
    return score


def _mk_params_grid() -> Iterable[Tuple[float, int, int]]:
    for c in [0.55, 0.58, 0.60, 0.62]:
        for d in [15, 30, 45, 60]:
            for h in [1, 2]:
                yield (c, d, h)


def _env_or_default(key: str, default: Any) -> Any:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        if isinstance(default, int):
            return int(v)
        if isinstance(default, float):
            return float(v)
        return v
    except Exception:
        return default


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------
# Public API
# -----------------------

def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
    write_summary: bool = True,
) -> Dict[str, Any]:
    """
    Grid-search thresholds with constraints:
      - win_rate >= 0.60
      - 5 <= signals/day <= 10 (aggregate)
    Tie-break by PF (higher), then |MDD| (lower), then WR (higher).
    Writes:
      - models/backtest_summary.json
      - models/signal_thresholds.json
      - logs/trades.jsonl (best config, if any trades)
    """
    fees_bps = _env_or_default("MW_FEES_BPS", 1.0)
    slippage_bps = _env_or_default("MW_SLIPPAGE_BPS", 2.0)

    tried: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] | None = None
    best_agg: Dict[str, float] | None = None
    best_per_symbol: Dict[str, Dict[str, float]] | None = None
    best_pen = float("inf")
    best_trades_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    symbols = sorted(pred_dfs.keys())

    for conf_min, debounce_min, horizon_h in _mk_params_grid():
        per_symbol_metrics: Dict[str, BTMetrics] = {}
        trades_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

        for sym in symbols:
            pred_df = pred_dfs[sym]
            price_df = prices.get(sym)
            if price_df is None or pred_df is None or pred_df.empty:
                per_symbol_metrics[sym] = BTMetrics()
                trades_by_symbol[sym] = []
                continue

            bt = run_backtest(
                pred_df=pred_df,
                prices_df=price_df,
                conf_min=conf_min,
                debounce_min=debounce_min,
                horizon_h=horizon_h,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
            )

            # Normalize trades for later logging
            trades = bt.get("trades", [])
            norm_trades: List[Dict[str, Any]] = []
            if isinstance(trades, (list, tuple)):
                for t in trades:
                    norm_trades.append(_trade_to_dict(t))
            trades_by_symbol[sym] = norm_trades

            per_symbol_metrics[sym] = _extract_metrics(bt)

        # Aggregate & score
        agg = _aggregate_metrics(per_symbol_metrics)
        pen = _penalty(agg, target_wr=0.60, min_spd=5.0, max_spd=10.0)

        tried.append(
            {
                "conf_min": conf_min,
                "debounce_min": debounce_min,
                "horizon_h": horizon_h,
                "aggregate": agg,
                "per_symbol": {k: vars(v) for k, v in per_symbol_metrics.items()},
                "penalty": pen,
            }
        )

        feasible = (
            agg.get("win_rate", 0.0) >= 0.60
            and 5.0 <= agg.get("signals_per_day", 0.0) <= 10.0
        )

        def _tie_key(a: Dict[str, float]) -> Tuple[float, float, float]:
            return (a.get("profit_factor", 0.0), -abs(a.get("max_drawdown", 0.0)), a.get("win_rate", 0.0))

        choose = False
        if best_params is None:
            choose = True
        else:
            if feasible and best_agg is not None:
                best_feasible = (
                    best_agg.get("win_rate", 0.0) >= 0.60
                    and 5.0 <= best_agg.get("signals_per_day", 0.0) <= 10.0
                )
                if feasible and not best_feasible:
                    choose = True
                elif feasible and best_feasible:
                    choose = _tie_key(agg) > _tie_key(best_agg)  # type: ignore
            else:
                choose = pen < best_pen

        if choose:
            best_params = {"conf_min": conf_min, "debounce_min": debounce_min, "horizon_h": horizon_h}
            best_agg = agg
            best_per_symbol = {k: vars(v) for k, v in per_symbol_metrics.items()}
            best_pen = pen
            best_trades_by_symbol = trades_by_symbol

    if best_params is None:
        best_params = {"conf_min": 0.55, "debounce_min": 15, "horizon_h": 1}
        best_agg = {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0}
        best_per_symbol = {sym: vars(BTMetrics()) for sym in symbols}

    result = {
        "params": best_params,
        "agg": best_agg,
        "per_symbol": best_per_symbol,
        "grid": tried,
    }

    if write_summary:
        _write_json(BACKTEST_SUMMARY_PATH, result)

    _write_json(
        THRESHOLDS_PATH,
        {
            "conf_min": best_params["conf_min"],
            "debounce_min": best_params["debounce_min"],
            "horizon_h": best_params["horizon_h"],
            "source": "tuner",
            "status": "finalized",
        },
    )

    if os.getenv("MW_WRITE_BT_LOGS", "1") == "1":
        try:
            with open(TRADES_LOG_PATH, "w", encoding="utf-8") as f:
                total = 0
                for sym, trades in best_trades_by_symbol.items():
                    if trades:
                        for rec in trades:
                            rec = dict(rec)
                            rec["symbol"] = sym
                            f.write(json.dumps(rec) + "\n")
                            total += 1
                if total == 0:
                    f.write(json.dumps({"note": "No trades met thresholds in best configuration."}) + "\n")
        except Exception:
            pass

    return result