# scripts/perf/replay_shadow_to_paper.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging

from scripts.perf.paper_trader import Ctx as PTX, run_paper_trader  # your existing paper trader
from src.paths import PAPER_TRADING_PARAMS_PATH

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Load centralized paper trading config
# -------------------------------------------------------------------
def _load_paper_trading_config() -> Dict[str, Any]:
    """Load paper trading parameters from config file with sensible defaults."""
    defaults = {
        "deadband": 0.08,
        "min_flip_min": 360,
        "lookback_h": 720,
        "conf_min": 0.60
    }

    try:
        if PAPER_TRADING_PARAMS_PATH.exists():
            with PAPER_TRADING_PARAMS_PATH.open("r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"Loaded paper trading params from {PAPER_TRADING_PARAMS_PATH}")
                return {**defaults, **config}  # Config overrides defaults
    except Exception as e:
        logger.warning(f"Failed to load paper trading params from {PAPER_TRADING_PARAMS_PATH}: {e}. Using defaults.")

    return defaults

_PT_CONFIG = _load_paper_trading_config()

# -------------------------------------------------------------------
# Config (env-overridable, with config file as fallback)
# -------------------------------------------------------------------
SHADOW_PATH = Path(os.getenv("SHADOW_LOG", "logs/signal_inference_shadow.jsonl"))
OUT_TRADES  = Path(os.getenv("PAPER_TRADES_LOG", "logs/paper_trades.jsonl"))

# Global fallbacks (used when a shadow row does not carry "gov")
# Priority: env var > config file > hardcoded default
DEFAULT_CONF_MIN   = float(os.getenv("REPLAY_CONF_MIN", str(_PT_CONFIG.get("conf_min", 0.60))))
DEFAULT_DEADBAND   = float(os.getenv("REPLAY_DEADBAND", os.getenv("MW_PERF_DEADBAND", str(_PT_CONFIG.get("deadband", 0.08)))))
DEFAULT_MIN_FLIP_M = int(os.getenv("REPLAY_MIN_FLIP_MIN", os.getenv("MW_PERF_MIN_FLIP_MIN", str(_PT_CONFIG.get("min_flip_min", 360)))))

LOOKBACK_H = int(os.getenv("REPLAY_LOOKBACK_H", os.getenv("MW_PERF_LOOKBACK_H", str(_PT_CONFIG.get("lookback_h", 720)))))
ALLOWED    = {s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()}

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _parse_ts(v) -> Optional[datetime]:
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip bad lines
                continue

def _gov_float(row: Dict[str, Any], key: str, fallback: float) -> float:
    gov = row.get("gov") or {}
    try:
        return float(gov.get(key, fallback))
    except Exception:
        return fallback

def _conf_min_from_row(row: Dict[str, Any]) -> float:
    return _gov_float(row, "conf_min", DEFAULT_CONF_MIN)

def _deadband_from_row(row: Dict[str, Any]) -> float:
    # distance from 0.5 below which we drop the row
    return _gov_float(row, "deadband", DEFAULT_DEADBAND)

def _min_flip_from_row(row: Dict[str, Any]) -> int:
    # minimum minutes between taking opposite directions for same symbol
    val = _gov_float(row, "min_flip_min", float(DEFAULT_MIN_FLIP_M))
    try:
        return int(val)
    except Exception:
        return DEFAULT_MIN_FLIP_M

# -------------------------------------------------------------------
# Selection & conversion
# -------------------------------------------------------------------
@dataclass
class _ShadowRow:
    ts: datetime
    symbol: str
    direction: str
    confidence: float
    raw: Dict[str, Any]

def _in_time_window(ts: Optional[datetime], win_start: datetime, win_end: datetime) -> bool:
    return bool(ts) and (win_start <= ts <= win_end)

def _qualify(row: Dict[str, Any], win_start: datetime, win_end: datetime) -> Optional[_ShadowRow]:
    sym = str(row.get("symbol", "")).upper()
    if sym not in ALLOWED:
        return None

    ts = _parse_ts(row.get("ts"))
    if not _in_time_window(ts, win_start, win_end):
        return None

    if not bool(row.get("ml_ok", False)):
        return None

    direction = str(row.get("ml_dir", "")).lower()
    if direction not in {"long", "short"}:
        return None

    try:
        conf = float(row.get("ml_conf", 0.0) or 0.0)
    except Exception:
        return None

    conf_min = _conf_min_from_row(row)
    if conf < conf_min:
        return None

    deadband = _deadband_from_row(row)
    if abs(conf - 0.5) < deadband:
        return None

    return _ShadowRow(ts=ts, symbol=sym, direction=direction, confidence=conf, raw=row)

def _to_signal(sr: _ShadowRow) -> Dict[str, Any]:
    ts_iso = sr.ts.isoformat().replace("+00:00", "Z")
    return {
        "id": f"shadow_{ts_iso}_{sr.symbol}_{sr.direction}",
        "ts": ts_iso,
        "symbol": sr.symbol,
        "direction": sr.direction,
        "confidence": sr.confidence,
        "price": None,
        "source": "shadow",
        "model_version": "bundle/current",
        "outcome": None,
    }

def _apply_min_flip(selected: List[_ShadowRow]) -> List[_ShadowRow]:
    """
    Enforce per-symbol min_flip_min using each row's own gov (if present) or global default.
    We must evaluate rows in chronological order.
    """
    selected = sorted(selected, key=lambda r: (r.symbol, r.ts))
    kept: List[_ShadowRow] = []
    last: Dict[str, Tuple[datetime, str, int]] = {}  # sym -> (ts, dir, min_flip_min)

    for r in selected:
        mf_this = _min_flip_from_row(r.raw)
        state = last.get(r.symbol)
        if state is None:
            kept.append(r)
            last[r.symbol] = (r.ts, r.direction, mf_this)
            continue

        last_ts, last_dir, last_mf = state
        # Use the stricter (max) of the two mins when deciding to flip
        gate_min = max(last_mf, mf_this)
        if r.direction != last_dir:
            delta_min = (r.ts - last_ts).total_seconds() / 60.0
            if delta_min < gate_min:
                # skip churny opposite signal
                continue

        kept.append(r)
        last[r.symbol] = (r.ts, r.direction, mf_this)

    return kept

# -------------------------------------------------------------------
# Main replay
# -------------------------------------------------------------------
def replay() -> Dict[str, Any]:
    now = _now_utc()
    win_start = now - timedelta(hours=LOOKBACK_H)

    rows = list(_read_jsonl(SHADOW_PATH))
    considered = 0
    selected: List[_ShadowRow] = []

    for r in rows:
        sr = _qualify(r, win_start, now)
        if sr is None:
            continue
        considered += 1
        selected.append(sr)

    # anti-churn (min_flip) pass
    selected = _apply_min_flip(selected)

    # write the canonical signals file for the paper trader
    sig_path = Path("logs") / "signal_history.jsonl"
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    with sig_path.open("w", encoding="utf-8") as f:
        for sr in sorted(selected, key=lambda x: x.ts):
            f.write(json.dumps(_to_signal(sr)) + "\n")

    # run paper trader
    ctx = PTX()
    ctx.symbols = list(ALLOWED)
    ctx.lookback_h = LOOKBACK_H
    ctx.force_demo = False
    ctx.demo_mode = False
    out = run_paper_trader(ctx, mode="replay-shadow")

    # small summary to stdout
    summary = {
        "shadow_path": str(SHADOW_PATH),
        "trades_out": str(OUT_TRADES),
        "total_rows": len(rows),
        "eligible_rows": considered,
        "written_trades": out.get("aggregate", {}).get("trades", 0),
        "per_symbol": {k: v.get("trades", 0) for k, v in out.get("by_symbol", {}).items()},
    }
    print("[replay_shadow_to_paper]", json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    replay()