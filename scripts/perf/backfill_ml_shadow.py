# scripts/perf/backfill_ml_shadow.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable
from datetime import datetime, timedelta, timezone

import numpy as np

# Training/inference helpers
from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features

# Optional: governance loader (for conf_min)
try:
    from src.signal_generator import load_governance_params as _load_gov
except Exception:
    _load_gov = None  # fallback later

# ---------------- Env / paths ----------------
ROOT = Path(".").resolve()
BUNDLE_ROOT = ROOT / "models" / "current"
SHADOW_LOG = ROOT / "logs" / "signal_inference_shadow.jsonl"

SYMBOLS = [s.strip().upper() for s in (os.getenv("MW_BACKFILL_SYMBOLS") or "BTC,ETH,SOL").split(",") if s.strip()]
LOOKBACK_DAYS = int(os.getenv("MW_BACKFILL_DAYS", "180") or "180")
INCLUDE_ALL = str(os.getenv("MW_BACKFILL_INCLUDE_ALL", "true")).lower() in {"1", "true", "yes"}
CONF_MIN_OVERRIDE_ENV = os.getenv("MW_BACKFILL_CONF_MIN_OVERRIDE", "").strip()

# ---------------- Bundle loader (per-symbol aware + self-heal) ----------------
def _load_bundle_for_symbol(symbol: str) -> Tuple[Any, List[str], Dict[str, Any], Path]:
    """
    Try in order:
      1) models/current/{SYMBOL}/
      2) models/current/default/
      3) models/current/           (legacy flat)
    Returns: (model, feature_order, manifest, bundle_dir)
    """
    import joblib
    search_dirs = [BUNDLE_ROOT / symbol, BUNDLE_ROOT / "default", BUNDLE_ROOT]
    last_err = None

    for base in search_dirs:
        man_p = base / "manifest.json"
        feat_p = base / "features.json"
        mdl_p = base / "model.joblib"
        if man_p.exists() and mdl_p.exists():
            try:
                manifest = json.loads(man_p.read_text(encoding="utf-8"))
                # features: prefer features.json, else manifest fields
                feats: List[str] = []
                if feat_p.exists():
                    fj = json.loads(feat_p.read_text(encoding="utf-8"))
                    if isinstance(fj, dict):
                        feats = fj.get("feature_order") or fj.get("features") or []
                    elif isinstance(fj, list):
                        feats = fj
                if not feats:
                    feats = manifest.get("feature_order") or manifest.get("features") or manifest.get("feature_list") or []
                if not isinstance(feats, list) or not feats:
                    raise RuntimeError("feature list missing in bundle")

                model = joblib.load(mdl_p)
                return model, [str(f) for f in feats], manifest, base
            except Exception as e:
                last_err = e
                # try next candidate
                continue

    if last_err:
        raise FileNotFoundError(f"inference bundle not found for {symbol} under {BUNDLE_ROOT}; last_err={type(last_err).__name__}: {last_err}")
    raise FileNotFoundError(f"inference bundle not found for {symbol} under {BUNDLE_ROOT}")

def _ensure_bundle_exists() -> None:
    """
    If no bundle exists under models/current/**, attempt to build one
    by invoking the training script.
    """
    has_any = False
    for p in [*BUNDLE_ROOT.rglob("model.joblib")]:
        has_any = True
        break
    if has_any:
        return
    # Try to build a minimal bundle
    try:
        print("[backfill] No bundle detected; training a minimal bundle via scripts.ml.train_predict …")
        from scripts.ml.train_predict import main as _train_main  # local import
        _train_main()
    except Exception as e:
        raise FileNotFoundError(f"Unable to build bundle automatically: {type(e).__name__}: {e}")

# ---------------- Feature / scoring helpers ----------------
def _vectorize(row: Dict[str, Any], feat_order: List[str]) -> np.ndarray:
    return np.array([[float(row.get(k, 0.0) or 0.0) for k in feat_order]], dtype=float)

def _score(model, row: Dict[str, Any], feat_order: List[str]) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(_vectorize(row, feat_order))[0, 1])
    if hasattr(model, "decision_function"):
        z = float(model.decision_function(_vectorize(row, feat_order))[0])
        return 1.0 / (1.0 + np.exp(-z))
    # fallback: try predict giving 0/1
    if hasattr(model, "predict"):
        y = float(model.predict(_vectorize(row, feat_order))[0])
        return float(max(0.0, min(1.0, y)))
    raise RuntimeError("model has neither predict_proba nor decision_function nor predict")

def _governance_conf_min(symbol: str) -> float:
    if CONF_MIN_OVERRIDE_ENV:
        try:
            return float(CONF_MIN_OVERRIDE_ENV)
        except Exception:
            pass
    if callable(_load_gov):
        try:
            g = _load_gov(symbol) or {}
            return float(g.get("conf_min", 0.6))
        except Exception:
            return 0.6
    return 0.6

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------- Main backfill ----------------
def _iter_rows_last_days(feat_df, days: int) -> Iterable[Dict[str, Any]]:
    """
    Yield dict-rows for the last N days from a feature DataFrame that has a 'ts' column.
    Assumes hourly data (but works for any frequency present).
    """
    if feat_df is None or len(feat_df) == 0:
        return
    # get ts as datetime and filter
    df = feat_df.copy()
    if "ts" in df.columns:
        # already present (our feature builder usually keeps it)
        df["ts"] = np.array(df["ts"], dtype="datetime64[ns]")
        df["ts"] = (df["ts"].astype("datetime64[ns]")).astype("datetime64[ms]")
        df["ts"] = df["ts"].astype("datetime64[ns]")
        # pandas already tz-aware in builder; we coerce to ISO later
    else:
        # if 'ts' was the index
        if df.index.name == "ts":
            df = df.reset_index()
        else:
            # nothing to iterate
            return

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_ns = np.datetime64(cutoff.isoformat().replace("+00:00", ""))
    df = df[df["ts"].values >= cutoff_ns]
    for _, r in df.iterrows():
        # convert row to dict
        d = {k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v) for k, v in r.to_dict().items()}
        yield d

def _write_shadow_line(f, rec: Dict[str, Any]) -> None:
    f.write(json.dumps(rec, default=str) + "\n")

def main():
    # 1) Ensure bundle (and feature order) exist
    _ensure_bundle_exists()

    # 2) Load prices & features once for the lookback window
    print(f"[backfill] Loading prices for {SYMBOLS} over {LOOKBACK_DAYS}d …")
    prices = load_prices(SYMBOLS, lookback_days=LOOKBACK_DAYS)
    feats_map = build_features(prices)  # dict[symbol] -> DataFrame (includes 'ts' + features)

    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0

    with SHADOW_LOG.open("a", encoding="utf-8") as out:
        for sym in SYMBOLS:
            # 3) Locate the appropriate bundle for this symbol
            try:
                model, feat_order, manifest, base = _load_bundle_for_symbol(sym)
            except Exception as e:
                # log one error line and continue other symbols
                _write_shadow_line(out, {
                    "symbol": sym, "reason": f"ml_exception:{type(e).__name__}",
                    "ml_ok": False, "ml_dir": None, "ml_conf": None,
                    "ts": _utcnow_iso()
                })
                continue

            # 4) Iterate feature rows and score
            df = feats_map.get(sym)
            if df is None or len(df) == 0:
                _write_shadow_line(out, {
                    "symbol": sym, "reason": "no_features",
                    "ml_ok": False, "ml_dir": None, "ml_conf": None,
                    "ts": _utcnow_iso()
                })
                continue

            conf_min = _governance_conf_min(sym)

            for row in _iter_rows_last_days(df, LOOKBACK_DAYS):
                # Build ts
                ts_val = row.get("ts")
                try:
                    # Convert to ISO8601 Z
                    ts_iso = (
                        ts_val.isoformat().replace("+00:00", "Z")
                        if hasattr(ts_val, "isoformat")
                        else str(ts_val)
                    )
                except Exception:
                    ts_iso = _utcnow_iso()

                # Score
                try:
                    p_long = _score(model, row, feat_order)
                    y_dir = "long" if p_long >= 0.5 else "short"
                    rec: Dict[str, Any] = {
                        "symbol": sym,
                        "reason": "backfill-ml",
                        "ml_ok": True,
                        "ml_dir": y_dir,
                        "ml_conf": float(p_long),
                        "ts": ts_iso,
                        "bundle": str(base),
                    }

                    # gating (unless include_all)
                    if not INCLUDE_ALL:
                        if CONF_MIN_OVERRIDE_ENV:
                            rec["conf_min_override"] = CONF_MIN_OVERRIDE_ENV
                        if p_long < conf_min:
                            # still write the row (so we can see it), but tag gated
                            rec["gated_below_conf_min"] = True
                            rec["conf_min"] = conf_min

                    _write_shadow_line(out, rec)
                    wrote += 1
                except Exception as e:
                    _write_shadow_line(out, {
                        "symbol": sym,
                        "reason": f"ml_exception:{type(e).__name__}",
                        "ml_ok": False, "ml_dir": None, "ml_conf": None,
                        "ts": ts_iso,
                        "bundle": str(base),
                    })

    print(f"[backfill] wrote {wrote} row(s) → {SHADOW_LOG}")

if __name__ == "__main__":
    main()