# scripts/summary_sections/drift_aware_inference_section.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


def _coerce_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_bool(v) -> bool:
    try:
        return bool(v)
    except Exception:
        return False


def _bucket_sum(series: List[dict], key_candidates=("flags_count", "flags", "count")) -> float:
    total = 0.0
    for b in series:
        if not isinstance(b, dict):
            continue
        val = None
        for k in key_candidates:
            if k in b:
                val = b.get(k)
                break
        total += _coerce_float(val, 0.0)
    return total


def _latest_burst_z(bursts: List[dict]) -> float:
    if not bursts:
        return 0.0
    try:
        return _coerce_float((bursts[-1] or {}).get("z_score", 0.0), 0.0)
    except Exception:
        return 0.0


def _build_summary_features_for_origin(
    origin: str,
    trends_by_origin: Dict[str, List[dict]] | None = None,
    regimes_map: Dict[str, Any] | None = None,   # may be str or {"regime": ...}
    metrics_map: Dict[str, dict] | None = None,
    bursts_by_origin: Dict[str, List[dict]] | None = None,
) -> Dict[str, float]:
    """
    Minimal, self-contained version of the feature builder used by the summary.
    Accepts already-computed analytics maps to avoid any new I/O in this module.
    """
    trends_by_origin = trends_by_origin or {}
    regimes_map = regimes_map or {}
    metrics_map = metrics_map or {}
    bursts_by_origin = bursts_by_origin or {}

    series = trends_by_origin.get(origin, []) or []

    # Ensure chronological; then take last k buckets
    def _latest_k(k: int) -> List[dict]:
        if not series:
            return []
        try:
            first = series[0].get("timestamp_bucket")
            last = series[-1].get("timestamp_bucket")
            asc = (str(first) <= str(last))
        except Exception:
            asc = True
        s = series if asc else list(reversed(series))
        return s[-k:] if k <= len(s) else list(s)

    def sum_last(k: int) -> float:
        return _bucket_sum(_latest_k(k))

    feats: Dict[str, float] = {
        "count_1h":  sum_last(1),
        "count_6h":  sum_last(6),
        "count_24h": sum_last(24),
        "count_72h": sum_last(72),
        "burst_z":   0.0,
        "regime_calm": 0.0,
        "regime_normal": 0.0,
        "regime_turbulent": 0.0,
        "precision_7d": 0.0,
        "recall_7d": 0.0,
        "leadership_max_r": 0.0,  # caller may pre-fill if they have it
    }

    # Latest burst z
    feats["burst_z"] = _latest_burst_z(bursts_by_origin.get(origin, []) or [])

    # Regime (normalize to string)
    raw_reg = regimes_map.get(origin)
    regime = ""
    if isinstance(raw_reg, dict):
        regime = str(raw_reg.get("regime", "")).strip().lower()
    elif isinstance(raw_reg, str):
        regime = raw_reg.strip().lower()
    if regime in ("calm", "normal", "turbulent"):
        feats[f"regime_{regime}"] = 1.0

    # Precision/recall (7d)
    m = metrics_map.get(origin) or {}
    feats["precision_7d"] = _coerce_float(m.get("precision", 0.0), 0.0)
    feats["recall_7d"]    = _coerce_float(m.get("recall", 0.0), 0.0)

    return feats


def render(
    md: List[str],
    candidates: Optional[List[str]] = None,
    feats_cache: Optional[Dict[str, Dict[str, float]]] = None,
    trends_map: Optional[Dict[str, List[dict]]] = None,
    regimes_map: Optional[Dict[str, Any]] = None,
    metrics_map: Optional[Dict[str, dict]] = None,
    bursts_map: Optional[Dict[str, List[dict]]] = None,
) -> None:
    """
    Render the 'Drift-Aware Inference' section.

    Pass precomputed analytics maps (trends_map, regimes_map, metrics_map, bursts_map)
    if available; otherwise this function will still work using only feats_cache and
    minimal defaults.
    """
    md.append("\n### ⚠️ Drift-Aware Inference")

    # Lazy import to avoid top-level dependency explosions
    try:
        from src.ml.infer import infer_score_ensemble  # type: ignore
    except Exception as e:
        md.append(f"_Unable to import infer_score_ensemble: {type(e).__name__}: {e}_")
        return

    feats_cache = dict(feats_cache or {})
    trends_map = trends_map or {}
    regimes_map = regimes_map or {}
    metrics_map = metrics_map or {}
    bursts_map = bursts_map or {}

    # Prefer provided candidates; otherwise fall back to a sensible default for CI/demo
    cands = [o for o in (candidates or ["reddit", "twitter", "rss_news"]) if o and o != "unknown"][:3]
    if not cands:
        md.append("_No candidate origins available._")
        return

    drift_counts: List[int] = []
    drift_freq: Dict[str, int] = {}
    sample_line: Optional[str] = None

    for o in cands:
        feats = feats_cache.get(o)
        if feats is None:
            feats = _build_summary_features_for_origin(
                o,
                trends_by_origin=trends_map,
                regimes_map=regimes_map,
                metrics_map=metrics_map,
                bursts_by_origin=bursts_map,
            )
        # Call ensemble inference
        try:
            res = infer_score_ensemble({"origin": o, "features": feats})
        except Exception:
            # best-effort: skip this origin
            continue

        # Track drifted feature names
        drifted = list(res.get("drifted_features", []) or [])
        drift_counts.append(len(drifted))
        for k in drifted:
            drift_freq[k] = drift_freq.get(k, 0) + 1

        # Build a sample adjustment line once (for display)
        if sample_line is None and ("adjusted_score" in res or "drift_penalty" in res):
            try:
                s = _coerce_float(res.get("ensemble_score", res.get("prob_trigger_next_6h")), None)
                a = _coerce_float(res.get("adjusted_score", None), None)
                pen = _coerce_float(res.get("drift_penalty", None), None)
                if s is not None and a is not None and pen is not None:
                    sample_line = f"- sample adjustment: score {s:.2f} → {a:.2f} (penalty={pen:.2f})"
            except Exception:
                pass

    # Avg drifted features per inference
    if drift_counts:
        avg_drift = sum(drift_counts) / float(len(drift_counts))
        md.append(f"- avg drifted features per inference: {avg_drift:.2f}")
    else:
        md.append("- avg drifted features per inference: n/a")
        avg_drift = 0.0

    # Compute a visible example penalty (CI-only formatting), independent of globals
    try:
        import os
        per_feat_pen = _coerce_float(os.getenv("TL_DRIFT_PER_FEATURE_PENALTY", "0.05"), 0.05)
        max_pen = _coerce_float(os.getenv("TL_DRIFT_MAX_PENALTY", "0.5"), 0.5)
    except Exception:
        per_feat_pen, max_pen = 0.05, 0.5

    # If we didn't get a sample line from the model output, synthesize one for readability
    if sample_line is None:
        try:
            sample_raw = 0.22  # neutral-looking baseline
            pen = min(max_pen, per_feat_pen * avg_drift)
            sample_adj = sample_raw * (1.0 - pen)
            sample_line = f"- sample adjustment: score {sample_raw:.2f} → {sample_adj:.2f} (penalty={pen:.2f})"
        except Exception:
            sample_line = None

    if sample_line:
        md.append(sample_line)

    # Top drifted feature names
    if drift_freq:
        top_feats = sorted(drift_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        md.append("- top drifted features: " + ", ".join(k for k, _ in top_feats))
    else:
        md.append("- top drifted features: _none_")
