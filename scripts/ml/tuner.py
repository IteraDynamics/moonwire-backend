# scripts/ml/tuner.py
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make sure "project root" is on path whether run as module or script
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lazy imports from our package (do not fail at import time for tests that mock pieces)
try:
    from scripts.ml.backtest import run_backtest  # type: ignore
except Exception:  # pragma: no cover
    run_backtest = None  # type: ignore


MODELS = ROOT / "models"
LOGS = ROOT / "logs"
ARTIFACTS = ROOT / "artifacts"
for d in (MODELS, LOGS, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------

def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple)):
            return len(x)
        return int(x)
    except Exception:
        return default


def _get(o: Any, key: str, default: Any = None) -> Any:
    """dict.get but tolerates objects with attributes (dataclass Trades)."""
    if isinstance(o, dict):
        return o.get(key, default)
    try:
        return getattr(o, key, default)
    except Exception:
        return default


def _max_drawdown_from_equity(equity_points: Iterable[Dict[str, Any]]) -> float:
    """
    Compute max drawdown from an equity curve iterable of {'ts', 'equity'}.
    Returns negative number (e.g., -0.12 for -12%).
    """
    peak = -float("inf")
    max_dd = 0.0
    for pt in equity_points:
        eq = _as_float(_get(pt, "equity", 1.0), 1.0)
        if eq > peak:
            peak = eq
        dd = (eq / peak) - 1.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd  # already negative or 0


@dataclass
class BTMetrics:
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    signals_per_day: float = 0.0


def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Accepts flexible shapes from run_backtest and extracts consistent metrics.
    Supports:
      - metrics at root or under 'metrics'
      - 'trades' can be a list of dicts or objects, or a count
    """
    m = bt.get("metrics", {})

    # n_trades — prefer explicit, else infer
    n_trades = _as_int(m.get("n_trades", bt.get("trades", 0)))
    if n_trades == 0:
        n_trades = _as_int(bt.get("trades", 0), 0)

    win_rate = _as_float(bt.get("win_rate", m.get("win_rate", 0.0)), 0.0)
    profit_factor = _as_float(bt.get("profit_factor", m.get("profit_factor", 0.0)), 0.0)
    max_drawdown = _as_float(bt.get("max_drawdown", m.get("max_drawdown", 0.0)), 0.0)
    spd = _as_float(bt.get("signals_per_day", m.get("signals_per_day", 0.0)), 0.0)

    # If any core metric missing, try to compute from trades/equity
    trades_obj = bt.get("trades", None)
    equity = bt.get("equity", None)

    wins = _as_int(m.get("wins", bt.get("wins", 0)), 0)
    losses = _as_int(m.get("losses", bt.get("losses", 0)), 0)

    # Compute wins/losses/profit factor from trades if possible
    if isinstance(trades_obj, (list, tuple)) and len(trades_obj) > 0:
        w = 0
        l = 0
        pos_pnl = 0.0
        neg_pnl = 0.0
        for t in trades_obj:
            pnl = _as_float(_get(t, "pnl", _get(t, "pnl_pct", 0.0)), 0.0)
            if pnl > 0:
                w += 1
                pos_pnl += pnl
            elif pnl < 0:
                l += 1
                neg_pnl += abs(pnl)
        if wins == 0 and losses == 0:
            wins, losses = w, l
        if profit_factor == 0.0 and (pos_pnl > 0 or neg_pnl > 0):
            profit_factor = (pos_pnl / neg_pnl) if neg_pnl > 0 else (pos_pnl if pos_pnl > 0 else 0.0)
        if n_trades == 0:
            n_trades = w + l

    if win_rate == 0.0 and (wins + losses) > 0:
        win_rate = wins / float(wins + losses)

    if max_drawdown == 0.0 and isinstance(equity, list) and len(equity) > 0:
        max_drawdown = _max_drawdown_from_equity(equity)

    # signals_per_day — keep whatever the backtest provides; else rough fallback
    if spd == 0.0:
        # Prefer explicit 'days' in metrics if present
        days = _as_float(_get(m, "days", _get(bt, "days", 0.0)), 0.0)
        if days > 0:
            spd = n_trades / days

    return BTMetrics(
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        signals_per_day=spd,
    )


# ---------- Grid & Objective ----------

def _grid() -> Tuple[List[float], List[int], List[int]]:
    """
    Widened search to help reach 4–12 signals/day:
    - Lower conf_min and shorter debounce open up more entries.
    """
    confs = [0.50, 0.52, 0.55, 0.58, 0.60]
    debounces = [5, 10, 15, 30]
    horizons = [1, 2]
    return confs, debounces, horizons


def _objective_ok(agg: Dict[str, Any]) -> bool:
    """
    Keep WR>=60% target. Allow 4–12 signals/day so the tuner can actually
    reach volume first; we’ll then order by PF and MDD in tie-breaks.
    """
    wr = float(agg.get("win_rate", 0.0))
    spd = float(agg.get("signals_per_day", 0.0))
    return (wr >= 0.60) and (4.0 <= spd <= 12.0)


# ---------- Aggregation ----------

def _aggregate(per_symbol: Dict[str, BTMetrics]) -> Dict[str, Any]:
    """
    Aggregate across symbols:
      - win_rate: wins/total
      - profit_factor: sum(pos_pnl)/sum(neg_pnl) (approx via PF~ weighted)
      - max_drawdown: take the worst (most negative)
      - signals_per_day: sum across symbols
    Since we rarely have pnl legs here, we approximate PF as the average of PFs
    weighted by n_trades.
    """
    total_trades = sum(m.n_trades for m in per_symbol.values())
    if total_trades == 0:
        return dict(win_rate=0.0, profit_factor=0.0, max_drawdown=0.0, signals_per_day=0.0)

    # Win rate weighted by trades
    wr_num = 0.0
    for m in per_symbol.values():
        wr_num += m.win_rate * m.n_trades
    agg_wr = wr_num / total_trades

    # Profit factor weighted by trades (proxy)
    pf_num = 0.0
    for m in per_symbol.values():
        pf_num += m.profit_factor * m.n_trades
    agg_pf = (pf_num / total_trades) if total_trades > 0 else 0.0

    # Worst drawdown
    agg_mdd = min((m.max_drawdown for m in per_symbol.values()), default=0.0)

    # Signals/day: sum across symbols
    agg_spd = sum(m.signals_per_day for m in per_symbol.values())

    return dict(
        win_rate=round(agg_wr, 4),
        profit_factor=round(agg_pf, 4),
        max_drawdown=round(agg_mdd, 4),
        signals_per_day=round(agg_spd, 2),
    )


def _rank_key(rec: Dict[str, Any]) -> Tuple:
    """
    Rank feasible candidates by:
      1) higher profit_factor
      2) lower max_drawdown (more negative = worse)
      3) higher win_rate
      4) higher signals/day (within the objective band)
    """
    agg = rec["aggregate"]
    return (
        -_as_float(agg.get("profit_factor", 0.0)),
        _as_float(agg.get("max_drawdown", 0.0)),  # less negative (closer to 0) is better
        -_as_float(agg.get("win_rate", 0.0)),
        -_as_float(agg.get("signals_per_day", 0.0)),
    )


# ---------- Public API ----------

def tune_thresholds(
    pred_dfs: Dict[str, Any],
    prices: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a small grid search over (conf_min, debounce_min, horizon_h) and pick
    thresholds that satisfy the objective (WR>=0.60 & 4<=signals/day<=12).
    Returns a dict with:
      {
        "params": {"conf_min": ..., "debounce_min": ..., "horizon_h": ...},
        "aggregate": {...},
        "per_symbol": {...}
      }
    Also writes:
      - models/signal_thresholds.json
      - models/backtest_summary.json
    """
    if run_backtest is None:
        # Should not happen in CI; but be safe.
        params = {"conf_min": 0.55, "debounce_min": 15, "horizon_h": 1}
        out = {"params": params, "aggregate": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0}, "per_symbol": {}}
        (MODELS / "signal_thresholds.json").write_text(json.dumps(params, indent=2))
        (MODELS / "backtest_summary.json").write_text(json.dumps({"aggregate": out["aggregate"], "per_symbol": {}}, indent=2))
        return out

    confs, debounces, horizons = _grid()

    # Allow env overrides for quick tuning experiments
    env_confs = os.environ.get("MW_TUNE_CONFS")
    if env_confs:
        confs = [float(x) for x in env_confs.split(",") if x.strip()]
    env_deb = os.environ.get("MW_TUNE_DEBOUNCES")
    if env_deb:
        debounces = [int(x) for x in env_deb.split(",") if x.strip()]
    env_h = os.environ.get("MW_TUNE_HORIZONS")
    if env_h:
        horizons = [int(x) for x in env_h.split(",") if x.strip()]

    candidates: List[Dict[str, Any]] = []

    symbols = list(pred_dfs.keys())
    for conf in confs:
        for db in debounces:
            for hz in horizons:
                per_sym_metrics: Dict[str, BTMetrics] = {}
                for sym in symbols:
                    pred_df = pred_dfs[sym]
                    price_df = prices[sym]
                    try:
                        bt = run_backtest(
                            pred_df,
                            price_df,
                            conf_min=conf,
                            debounce_min=db,
                            horizon_h=hz,
                            fees_bps=float(os.environ.get("MW_FEES_BPS", "1")),
                            slippage_bps=float(os.environ.get("MW_SLIPPAGE_BPS", "2")),
                        )
                    except Exception:
                        bt = {}
                    per_sym_metrics[sym] = _extract_metrics(bt)

                agg = _aggregate(per_sym_metrics)
                candidates.append(
                    {
                        "params": {"conf_min": conf, "debounce_min": db, "horizon_h": hz},
                        "aggregate": agg,
                        "per_symbol": {k: m.__dict__ for k, m in per_sym_metrics.items()},
                    }
                )

    # 1) Try to find any that meet the objective
    feasible = [c for c in candidates if _objective_ok(c["aggregate"])]

    # 2) If none feasible, choose the best by PF, then MDD, then WR, then SPD
    if feasible:
        feasible.sort(key=_rank_key)
        best = feasible[0]
    else:
        # Prefer combos with >0 trades first (sum of per-symbol trades)
        def total_trades(c: Dict[str, Any]) -> int:
            return sum(int(c["per_symbol"][s]["n_trades"]) for s in c["per_symbol"].keys())

        nonzero = [c for c in candidates if total_trades(c) > 0]
        if nonzero:
            nonzero.sort(key=_rank_key)
            best = nonzero[0]
        else:
            # Degenerate: pick a friendly default
            best = {
                "params": {"conf_min": 0.55, "debounce_min": 15, "horizon_h": 1},
                "aggregate": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0},
                "per_symbol": {s: BTMetrics().__dict__ for s in symbols},
            }

    # Write artifacts
    (MODELS / "signal_thresholds.json").write_text(json.dumps(best["params"], indent=2))
    backtest_summary = {
        "aggregate": best["aggregate"],
        "per_symbol": best["per_symbol"],
    }
    (MODELS / "backtest_summary.json").write_text(json.dumps(backtest_summary, indent=2))

    return best


# ---------- CLI (optional) ----------

if __name__ == "__main__":  # pragma: no cover
    # Minimal manual run if invoked directly (expects a prior pipeline to have produced preds)
    print("tuner.py is intended to be called from the train/predict pipeline.")