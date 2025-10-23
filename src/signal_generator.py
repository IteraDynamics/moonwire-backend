# src/signal_generator.py
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from src.signal_filter import is_signal_valid
from src.cache_instance import cache
from src.sentiment_blended import blend_sentiment_scores
from src.dispatcher import dispatch_alerts

# --- optional ML inference (soft dependency) -------------------------------
_ML_INFER_FN = None
try:
    # If you already have a helper in src/ml/infer.py, expose a simple function:
    # def infer_asset_signal(asset: str) -> Dict[str, Any]: {"direction": "long"/"short", "confidence": 0.73, ...}
    from src.ml.infer import infer_asset_signal as _ML_INFER_FN  # type: ignore
except Exception:
    _ML_INFER_FN = None

# --- governance params loader ----------------------------------------------
GOV_PATH = Path("models/governance_params.json")

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
    data = _read_json(GOV_PATH, {})
    row = data.get(symbol) or {}
    return {
        "conf_min": float(row.get("conf_min", default["conf_min"])),
        "debounce_min": int(row.get("debounce_min", default["debounce_min"])),
    }

# --- shadow logging ---------------------------------------------------------
SHADOW_LOG = Path("logs/signal_inference_shadow.jsonl")
SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)

def _shadow_write(payload: Dict[str, Any]) -> None:
    """
    Append a single JSON line to the shadow log. Never throw.
    """
    try:
        payload = dict(payload)
        if "ts" not in payload:
            payload["ts"] = _utcnow_iso()
        line = json.dumps(payload, default=str)
        with SHADOW_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # don't break signal flow for logging errors
        pass

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
        out = _ML_INFER_FN(asset)  # expected: {"direction": "...", "confidence": 0.xx, ...}
        if not isinstance(out, dict):
            return {"ok": False, "dir": None, "conf": None, "reason": "ml_bad_return_type"}

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
    print(f"[{datetime.utcnow()}] Running signal generation...")

    stablecoins = {"USDC", "USDT", "DAI", "TUSD", "BUSD"}
    valid_signals: list[dict] = []

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
                    "timestamp": datetime.utcnow(),
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
                    "timestamp": datetime.utcnow(),
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

# --- CI/cron probe ----------------------------------------------------------
def shadow_probe(symbols: Optional[Iterable[str]] = None, reason: str = "ci_probe") -> None:
    """
    CI/cron probe: writes one shadow line per symbol.
    - `symbols` can be an iterable of strings or a single comma-separated string.
      If None, falls back to env MW_SHADOW_SYMBOLS (default "BTC").
    - `reason` is stored into the log (e.g., "shadow-cron").
    """
    if symbols is None:
        raw = os.getenv("MW_SHADOW_SYMBOLS", "BTC")
        # allow comma-separated string or single token
        if isinstance(raw, str):
            symbols = [s.strip() for s in raw.split(",") if s.strip()]
        else:
            symbols = ["BTC"]

    # If someone passed a single string directly, normalize to list
    if isinstance(symbols, str):
        symbols = [symbols]

    ts = _utcnow_iso()
    wrote = 0
    for sym in symbols:
        gov = load_governance_params(sym)
        _shadow_write({
            "symbol": sym,
            "reason": reason,
            "ml_ok": False,      # probe isn't running ML; just proving plumbing
            "ml_dir": None,
            "ml_conf": None,
            "gov": gov,
            "ts": ts,
        })
        wrote += 1
    print(f"[shadow_probe] wrote {wrote} record(s) to {SHADOW_LOG}")

# --- tiny CLI hook for CI probes (optional) --------------------------------
if __name__ == "__main__":
    # Allow a cheap probe run: write at least one line even if cache is empty
    _shadow_write({"symbol": "BTC", "reason": "ci_probe"})