# scripts/perf/replay_shadow_to_paper.py
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.perf.paper_trader import Ctx as PTX, run_paper_trader

# ---- Inputs / knobs ----
SHADOW_PATH = Path(os.getenv("SHADOW_LOG", "logs/signal_inference_shadow.jsonl"))
OUT_TRADES  = Path(os.getenv("PAPER_TRADES_LOG", "logs/paper_trades.jsonl"))

DEFAULT_CONF_MIN = float(os.getenv("REPLAY_CONF_MIN", "0.60"))
LOOKBACK_H       = int(os.getenv("REPLAY_LOOKBACK_H", os.getenv("MW_PERF_LOOKBACK_H", "72")))
ALLOWED          = {s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()}

# Optional extra gates (disabled by default unless you set them)
DEADBAND         = float(os.getenv("MW_PERF_DEADBAND", "0.00"))    # skip if |p-0.5| < deadband (e.g., 0.05)
MIN_FLIP_MIN     = int(os.getenv("MW_PERF_MIN_FLIP_MIN", "0"))     # ignore rapid long<->short flips within N minutes

# ---- Helpers ----
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
                continue

def _conf_min_from_row(row: Dict[str, Any]) -> float:
    gov = row.get("gov") or {}
    try:
        return float(gov.get("conf_min", DEFAULT_CONF_MIN))
    except Exception:
        return DEFAULT_CONF_MIN

def _baseline_qualifies(row: Dict[str, Any], win_start: datetime) -> Tuple[bool, Optional[datetime], Optional[str], Optional[float]]:
    """
    Baseline filter: window, allowed symbol, ml_ok, confidence >= conf_min.
    Returns (ok, ts, dir, conf) for re-use by later gates.
    """
    sym = str(row.get("symbol", "")).upper()
    if sym not in ALLOWED:
        return False, None, None, None

    ts = _parse_ts(row.get("ts")) or _now_utc()
    if ts < win_start:
        return False, None, None, None

    if not bool(row.get("ml_ok", False)):
        return False, None, None, None

    conf_v = row.get("ml_conf")
    if not isinstance(conf_v, (int, float)):
        return False, None, None, None
    conf = float(conf_v)

    # respect per-row governance conf_min if present
    if conf < _conf_min_from_row(row):
        return False, None, None, None

    dir_ = str(row.get("ml_dir", "")).lower()
    if dir_ not in {"long", "short"}:
        return False, None, None, None

    return True, ts, dir_, conf

def _to_signal(symbol: str, ts: datetime, direction: str, conf: float) -> Dict[str, Any]:
    """
    Convert to canonical signal line paper_trader expects.
    """
    ts_iso = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": f"shadow_{ts_iso}_{symbol}_{direction}",
        "ts": ts_iso,
        "symbol": symbol,
        "direction": direction,               # "long" | "short"
        "confidence": float(conf),
        "price": None,                        # price not needed; paper_trader uses price book
        "source": "shadow",
        "model_version": "bundle/current",
        "outcome": None,
    }

# ---- Main ----
def replay() -> Dict[str, Any]:
    win_start = _now_utc() - timedelta(hours=LOOKBACK_H)

    # 1) read rows and sort by timestamp (oldest -> newest)
    raw_rows = list(_read_jsonl(SHADOW_PATH))
    def _safe_ts(r): 
        t = _parse_ts(r.get("ts"))
        return t or _now_utc()
    raw_rows.sort(key=_safe_ts)

    considered_in_window = 0
    selected: List[Dict[str, Any]] = []

    # state for flip-cooldown
    last_kept: Dict[str, Tuple[datetime, str]] = {}  # symbol -> (ts, dir)

    for r in raw_rows:
        ok, ts, direction, conf = _baseline_qualifies(r, win_start)
        if not ok:
            continue

        considered_in_window += 1
        sym = str(r.get("symbol", "")).upper()

        # (A) deadband: skip indecisive probs near 0.5
        if DEADBAND > 0.0 and abs(conf - 0.5) < DEADBAND:
            continue

        # (B) flip cooldown: if direction flips too soon after previous kept trade for same symbol, skip
        if MIN_FLIP_MIN > 0 and sym in last_kept:
            last_ts, last_dir = last_kept[sym]
            if direction != last_dir and (ts - last_ts).total_seconds() < MIN_FLIP_MIN * 60:
                continue

        selected.append(_to_signal(sym, ts, direction, conf))
        last_kept[sym] = (ts, direction)

    # 2) write canonical signals file for paper_trader
    sig_path = Path("logs") / "signal_history.jsonl"
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    with sig_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    # 3) run paper trader on these signals
    ctx = PTX()
    ctx.symbols = list(ALLOWED)
    ctx.lookback_h = LOOKBACK_H
    ctx.force_demo = False
    ctx.demo_mode = False

    out = run_paper_trader(ctx, mode="replay-shadow")

    # 4) summary
    summary = {
        "shadow_path": str(SHADOW_PATH),
        "trades_out": str(OUT_TRADES),
        "total_rows": len(raw_rows),
        "considered_rows": considered_in_window,
        "written_trades": out.get("aggregate", {}).get("trades", 0),
        "per_symbol": {k: v.get("trades", 0) for k, v in out.get("by_symbol", {}).items()},
    }
    print("[replay_shadow_to_paper]", json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    replay()