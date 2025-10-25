# scripts/perf/replay_shadow_to_paper.py
from __future__ import annotations
import json, os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.perf.paper_trader import Ctx as PTX, run_paper_trader

SHADOW_PATH = Path(os.getenv("SHADOW_LOG", "logs/signal_inference_shadow.jsonl"))
OUT_DIR     = Path("logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Global fallbacks (used if coin override missing)
GLOBAL_LOOKBACK_H   = int(os.getenv("MW_PERF_LOOKBACK_H", "72"))
GLOBAL_HORIZON_MIN  = int(os.getenv("MW_PERF_HORIZON_MIN", "240"))
GLOBAL_FEES_BPS     = float(os.getenv("MW_PERF_FEES_BPS", "1"))
GLOBAL_SLIPPAGE_BPS = float(os.getenv("MW_PERF_SLIPPAGE_BPS", "2"))
GLOBAL_DEADBAND     = float(os.getenv("MW_PERF_DEADBAND", "0.00"))  # |p-0.5| < deadband → drop
GLOBAL_CONF_MIN     = float(os.getenv("REPLAY_CONF_MIN", "0.60"))
GLOBAL_MIN_FLIP_MIN = int(os.getenv("MW_PERF_MIN_FLIP_MIN", "0"))

ALLOWED   = {s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()}
GOV_FILE  = Path("models/governance_params.json")

def _now() -> datetime: return datetime.now(timezone.utc).replace(microsecond=0)

def _parse_ts(v) -> Optional[datetime]:
    try:
        s = str(v)
        if s.endswith("Z"): s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists(): return []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln: continue
        try:
            yield json.loads(ln)
        except Exception:
            continue

def _load_gov() -> Dict[str, Dict[str, Any]]:
    try:
        if GOV_FILE.exists():
            return json.loads(GOV_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _coin_knob(gov: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        v = gov.get(key, default)
        return type(default)(v) if v is not None else default
    except Exception:
        return default

def _filter_shadow_for_coin(rows: List[Dict[str,Any]], coin: str, win_start: datetime, gov: Dict[str,Any]) -> List[Dict[str,Any]]:
    """Apply window, conf_min, deadband, min_flip to coin rows."""
    conf_min    = _coin_knob(gov, "conf_min", GLOBAL_CONF_MIN)
    deadband    = _coin_knob(gov, "deadband", GLOBAL_DEADBAND)
    min_flip    = _coin_knob(gov, "min_flip_min", GLOBAL_MIN_FLIP_MIN)

    selected: List[Dict[str,Any]] = []
    last_dir: Optional[str] = None
    last_ts: Optional[datetime] = None

    for j in rows:
        if str(j.get("symbol","")).upper() != coin: continue
        ts = _parse_ts(j.get("ts"))
        if not ts or ts < win_start: continue
        if not j.get("ml_ok", False): continue
        conf = j.get("ml_conf")
        if not isinstance(conf, (int,float)): continue
        if conf < conf_min: continue
        if abs(float(conf) - 0.5) < float(deadband): continue

        d = str(j.get("ml_dir","")).lower()
        if d not in {"long","short"}: continue

        # min-flip guard
        if last_dir is not None and last_ts is not None:
            if d != last_dir and (ts - last_ts).total_seconds() < min_flip * 60:
                continue

        selected.append({
            "id": f"shadow_{ts.isoformat()}_{coin}_{d}",
            "ts": ts.isoformat().replace("+00:00","Z"),
            "symbol": coin,
            "direction": d,
            "confidence": float(conf),
            "price": None,
            "source": "shadow",
            "model_version": "bundle/current",
            "outcome": None,
        })
        last_dir, last_ts = d, ts

    return selected

def _run_one_coin(coin: str, signals: List[Dict[str,Any]], gov: Dict[str,Any]) -> Tuple[str, Dict[str,Any]]:
    # Write coin-specific signal_history the trader expects
    sig_path = OUT_DIR / f"signal_history_{coin}.jsonl"
    with sig_path.open("w", encoding="utf-8") as f:
        for r in signals: f.write(json.dumps(r) + "\n")

    # Build a ctx with per-coin overrides
    ctx = PTX()
    ctx.symbols        = [coin]
    ctx.lookback_h     = GLOBAL_LOOKBACK_H  # window for price data fetch
    ctx.horizon_min    = _coin_knob(gov, "horizon_min", GLOBAL_HORIZON_MIN)
    ctx.fees_bps       = _coin_knob(gov, "fees_bps", GLOBAL_FEES_BPS)
    ctx.slippage_bps   = _coin_knob(gov, "slippage_bps", GLOBAL_SLIPPAGE_BPS)
    ctx.force_demo     = False
    ctx.demo_mode      = False
    # Let paper_trader read the coin file (most versions read logs/signal_history.jsonl).
    # We swap the expected path via env to avoid changing the trader internals.
    os.environ["SIGNAL_HISTORY_PATH"] = str(sig_path)

    res = run_paper_trader(ctx, mode="replay-shadow")
    return coin, res

def replay() -> Dict[str, Any]:
    rows = list(_read_jsonl(SHADOW_PATH))
    if not rows:
        print("[replay] no shadow rows present at", SHADOW_PATH)
        return {"aggregate": {"trades": 0}, "by_symbol": {}, "demo": False}

    gov_all = _load_gov()
    now = _now()
    t0  = now - timedelta(hours=GLOBAL_LOOKBACK_H)

    # Per-coin runs
    per: Dict[str, Dict[str,Any]] = {}
    for coin in sorted(ALLOWED):
        gov = gov_all.get(coin, {})
        sigs = _filter_shadow_for_coin(rows, coin, t0, gov)
        if not sigs:
            per[coin] = {"trades": 0}
            continue
        coin_key, res = _run_one_coin(coin, sigs, gov)
        per[coin_key] = res.get("by_symbol", {}).get(coin_key, {"trades": 0})

    # Aggregate (simple roll-up of key stats)
    agg_trades = sum(v.get("trades",0) for v in per.values())
    aggregate = {
        "trades": agg_trades,
        # You can compute more blended stats here if needed
    }

    out = {
        "generated_at": now.replace(microsecond=0).isoformat().replace("+00:00","Z"),
        "mode": "replay-shadow",
        "window_hours": GLOBAL_LOOKBACK_H,
        "capital": float(os.getenv("MW_PERF_CAPITAL","100000")),
        "by_symbol": per,
        "aggregate": aggregate,
        "demo": False,
    }
    # Persist standard summary for your workflow summary step
    Path("models").mkdir(parents=True, exist_ok=True)
    (Path("models")/"performance_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[replay_shadow_to_paper]", json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    replay()