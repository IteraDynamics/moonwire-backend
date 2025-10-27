# src/signal_generator.py
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, List

from src.signal_filter import is_signal_valid
from src.cache_instance import cache
from src.sentiment_blended import blend_sentiment_scores
from src.dispatcher import dispatch_alerts
from src.jsonl_writer import atomic_jsonl_append
from src.observability import failure_tracker
from src.paths import SHADOW_LOG_PATH, GOVERNANCE_PARAMS_PATH

logger = logging.getLogger(__name__)

# --- optional ML inference (soft dependency) -------------------------------
_ML_INFER_FN = None
try:
    # Expects: def infer_asset_signal(symbol: str, *, models_dir: Path | None = None) -> Dict[str, Any]
    from src.ml.infer import infer_asset_signal as _ML_INFER_FN  # type: ignore
except Exception:
    _ML_INFER_FN = None

# --- governance params loader ----------------------------------------------
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def load_governance_params(symbol: str) -> Dict[str, Any]:
    """
    Returns per-symbol governance knobs. Defaults are conservative.
    """
    default = {"conf_min": 0.60, "debounce_min": 15}
    data = _read_json(GOVERNANCE_PARAMS_PATH, {})
    row = data.get(symbol) or {}
    return {
        "conf_min": float(row.get("conf_min", default["conf_min"])),
        "debounce_min": int(row.get("debounce_min", default["debounce_min"])),
    }

# --- shadow logging ---------------------------------------------------------
def _shadow_write(payload: Dict[str, Any]) -> None:
    """
    Append a single JSON line to the shadow log with atomic writes.
    Uses observable failure tracking - never throws.
    """
    try:
        payload = dict(payload)
        if "ts" not in payload:
            payload["ts"] = _utcnow_iso()
        atomic_jsonl_append(SHADOW_LOG_PATH, payload)
    except Exception as e:
        # Don't break signal flow, but track failures for monitoring
        failure_tracker.record_failure("shadow_write", e, {
            "symbol": payload.get("symbol"),
            "reason": payload.get("reason")
        })

# --- ML inference wrapper (safe) -------------------------------------------
def _infer_ml(asset: str) -> Dict[str, Any]:
    """
    Try ML inference for `asset`. Always returns a dict with keys:
      - ok: bool
      - dir: "long"/"short"/None
      - conf: float|None
      - reason: str (why it failed or 'ok')
      - raw: optional raw model output
    """
    # Feature flag to *completely* disable ML (even shadow)
    if str(os.getenv("MW_INFER_ENABLE", "1")).lower() not in {"1", "true", "yes"}:
        return {"ok": False, "dir": None, "conf": None, "reason": "ml_disabled"}

    if _ML_INFER_FN is None:
        return {"ok": False, "dir": None, "conf": None, "reason": "ml_unavailable"}

    try:
        # Let the infer module use its default bundle location (models/current)
        out = _ML_INFER_FN(asset)  # expected: {"direction": "...", "confidence": 0.xx, ...}
        if not isinstance(out, dict):
            return {"ok": False, "dir": None, "conf": None, "reason": "ml_bad_return_type"}
        if out.get("error"):
            return {"ok": False, "dir": None, "conf": None, "reason": f"ml_error:{out['error']}"}

        direction = out.get("direction")
        conf = out.get("confidence")

        if direction not in {"long", "short"} or not isinstance(conf, (int, float)):
            return {"ok": False, "dir": None, "conf": None, "reason": "ml_missing_keys", "raw": out}

        return {"ok": True, "dir": direction, "conf": float(conf), "reason": "ok", "raw": out}
    except Exception as e:
        return {"ok": False, "dir": None, "conf": None, "reason": f"ml_exception:{type(e).__name__}"}

# --- heuristic fallback (your existing logic) -------------------------------
def _heuristic_confidence(price_change: float, sentiment: float) -> float:
    """
    Your original heuristic: ((price_change / 10) + sentiment) / 2, clipped [0,1].
    """
    try:
        conf = ((price_change / 10.0) + float(sentiment)) / 2.0
        return float(max(0.0, min(1.0, conf)))
    except Exception:
        return 0.0

def label_confidence(score: float) -> str:
    if score >= 0.66:
        return "High Confidence"
    elif score >= 0.33:
        return "Medium Confidence"
    else:
        return "Low Confidence"

# --- main entry -------------------------------------------------------------
def generate_signals():
    """
    Default: use existing heuristic for **live** signals.
    Always: write **ML shadow** inference per asset.
    Flip to live-ML by setting MW_INFER_LIVE=1.
    """
    logger.info("Running signal generation", extra={
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    stablecoins = {"USDC", "USDT", "DAI", "TUSD", "BUSD"}
    valid_signals: List[dict] = []

    # feature-flag: shadow only? (never dispatch anything) -> useful in CI
    shadow_only = str(os.getenv("MW_INFER_SHADOW_ONLY", "0")).lower() in {"1", "true", "yes"}
    live_ml = str(os.getenv("MW_INFER_LIVE", "0")).lower() in {"1", "true", "yes"}

    try:
        sentiment_scores = blend_sentiment_scores()
        # cache.keys() may contain helpers; scrub the suffixes you used before
        assets = [k for k in cache.keys() if not k.endswith('_signals') and not k.endswith('_sentiment')]

        for asset in assets:
            if asset in stablecoins:
                continue

            data = cache.get_signal(asset)
            if not isinstance(data, dict):
                _shadow_write({"symbol": asset, "reason": "bad_signal_type", "got": str(type(data))})
                continue

            price_change = data.get("price_change_24h")
            volume = data.get("volume_now")
            if price_change is None or volume is None:
                _shadow_write({"symbol": asset, "reason": "missing_fields"})
                continue

            sentiment = float(sentiment_scores.get(asset, 0.0))

            # --- ML SHADOW inference (always attempted unless globally disabled)
            ml = _infer_ml(asset)
            gov = load_governance_params(asset)
            shadow_payload = {
                "symbol": asset,
                "reason": "shadow",
                "ml_ok": ml.get("ok", False),
                "ml_dir": ml.get("dir"),
                "ml_conf": ml.get("conf"),
                "ml_reason": ml.get("reason"),
                "gov": gov,
                "heuristic_sentiment": sentiment,
                "heuristic_price_change_24h": price_change,
            }
            _shadow_write(shadow_payload)

            # --- choose live path
            if live_ml and ml.get("ok"):
                # live ML path: direction/conf from model, gate by governance conf_min
                direction = ml["dir"]
                confidence = float(ml["conf"] or 0.0)
                if confidence < float(gov["conf_min"]):
                    _shadow_write({
                        "symbol": asset,
                        "reason": "live_ml_below_conf_min",
                        "conf": confidence,
                        "conf_min": gov["conf_min"]
                    })
                    continue

                signal = {
                    "asset": asset,
                    "price_change": price_change,
                    "volume": volume,
                    "sentiment": sentiment,
                    "confidence_score": confidence,
                    "confidence_label": label_confidence(confidence),
                    "direction": direction,
                    "timestamp": datetime.now(timezone.utc),
                    "governance": gov,
                    "inference": "ml_live",
                }

                if shadow_only:
                    _shadow_write({"symbol": asset, "reason": "shadow_only_live_ml_candidate", "dir": direction, "conf": confidence})
                    continue

                if is_signal_valid(signal):
                    dispatch_alerts(asset, signal, cache)
                    valid_signals.append(signal)
                else:
                    _shadow_write({"symbol": asset, "reason": "live_ml_rejected_by_filter", "dir": direction, "conf": confidence})

            else:
                # heuristic path (current production behavior)
                confidence = _heuristic_confidence(price_change, sentiment)
                direction = "long" if confidence >= 0.5 else "short"

                signal = {
                    "asset": asset,
                    "price_change": price_change,
                    "volume": volume,
                    "sentiment": sentiment,
                    "confidence_score": confidence,
                    "confidence_label": label_confidence(confidence),
                    "direction": direction,
                    "timestamp": datetime.now(timezone.utc),
                    "inference": "heuristic",
                }

                if shadow_only:
                    _shadow_write({"symbol": asset, "reason": "shadow_only_heuristic_candidate", "dir": direction, "conf": confidence})
                    continue

                if is_signal_valid(signal):
                    dispatch_alerts(asset, signal, cache)
                    valid_signals.append(signal)
                else:
                    _shadow_write({"symbol": asset, "reason": "heuristic_rejected_by_filter", "dir": direction, "conf": confidence})

    except Exception as e:
        _shadow_write({"symbol": None, "reason": f"generator_exception:{type(e).__name__}", "detail": str(e)})

    return valid_signals

# --- CI/cron probe (REAL inference) ----------------------------------------
def shadow_probe(symbols: Iterable[str] | str = ("BTC", "ETH", "SOL"), reason: str = "shadow-cron") -> None:
    """
    Probe that **actually runs ML inference** per symbol and appends one JSONL record.
    Keeps your workflow call signature compatible: shadow_probe([...], reason="...").
    """
    # Normalize symbols to a list
    if isinstance(symbols, str):
        symbols = [symbols]
    ts = datetime.now(timezone.utc).isoformat()

    wrote = 0
    for sym in symbols:
        gov = load_governance_params(sym)
        # Run ML; if disabled/unavailable, _infer_ml reports the reason
        ml = _infer_ml(sym)
        rec = {
            "symbol": sym,
            "reason": reason if ml.get("ok") else ml.get("reason", reason),
            "ml_ok": bool(ml.get("ok")),
            "ml_dir": ml.get("dir"),
            "ml_conf": ml.get("conf"),
            "gov": gov,
            "ts": ts,
        }
        _shadow_write(rec)
        wrote += 1
    logger.info(f"Shadow probe completed", extra={
        "records_written": wrote,
        "log_path": str(SHADOW_LOG_PATH)
    })

# --- tiny CLI hook for CI probes (optional) --------------------------------
if __name__ == "__main__":
    # Allow a cheap probe run: write at least one line even if cache is empty
    _shadow_write({"symbol": "BTC", "reason": "ci_probe"})