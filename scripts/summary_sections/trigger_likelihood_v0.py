# scripts/summary_sections/trigger_likelihood_v0.py
from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime, timezone

from .common import (
    SummaryContext,
    pick_candidate_origins,
)
import os


__all__ = ["append", "_build_summary_features_for_origin"]


def _build_summary_features_for_origin(
    origin: str,
    *,
    trends_by_origin: Dict[str, List[Dict[str, Any]]] | None = None,
    regimes_map: Dict[str, Any] | None = None,   # may be str or {"regime": "..."}
    metrics_map: Dict[str, Dict[str, float]] | None = None,
    bursts_by_origin: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, float]:
    """
    Build a compact feature vector for one origin from summary analytics:

      - rolling counts over last 1/6/24/72 buckets (from flags_count / flags / count)
      - latest burst z-score
      - regime one-hot (calm|normal|turbulent)
      - precision_7d / recall_7d
      - leadership_max_r (caller may set later; default 0)

    This function is intentionally dependency-light and resilient to messy inputs.
    """
    series = (trends_by_origin or {}).get(origin, []) or []

    # Detect chronological order, then take last k buckets.
    def _series_latest_k(k: int) -> list:
        if not series:
            return []
        try:
            first = (series[0] or {}).get("timestamp_bucket")
            last  = (series[-1] or {}).get("timestamp_bucket")
            asc = (str(first) <= str(last))
        except Exception:
            asc = True
        s = series if asc else list(reversed(series))
        return s[-k:] if k <= len(s) else s

    def _flags_from_bucket(b: dict) -> float:
        v = b.get("flags_count")
        if v is None:
            v = b.get("flags")
        if v is None:
            v = b.get("count")
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0

    def sum_last(k: int) -> float:
        buckets = _series_latest_k(k)
        return float(sum(_flags_from_bucket(x) for x in buckets))

    feats = {
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
        "leadership_max_r": 0.0,
    }

    # Latest burst z (if any)
    try:
        bursts = (bursts_by_origin or {}).get(origin) or []
        if bursts:
            feats["burst_z"] = float((bursts[-1] or {}).get("z_score", 0.0) or 0.0)
    except Exception:
        pass

    # Regime (string or dict)
    try:
        raw_reg = (regimes_map or {}).get(origin)
        if isinstance(raw_reg, dict):
            regime = str(raw_reg.get("regime", "")).strip().lower()
        else:
            regime = str(raw_reg or "").strip().lower()
        if regime in ("calm", "normal", "turbulent"):
            feats[f"regime_{regime}"] = 1.0
    except Exception:
        pass

    # Precision/recall (7d)
    try:
        m = (metrics_map or {}).get(origin) or {}
        feats["precision_7d"] = float(m.get("precision", 0.0) or 0.0)
        feats["recall_7d"]    = float(m.get("recall", 0.0) or 0.0)
    except Exception:
        pass

    return feats


def _demo_rich_scores_enabled() -> bool:
    """Gate richer scoring using derived features via env DEMO_RICH_SCORES."""
    return os.getenv("DEMO_RICH_SCORES", "false").lower() in ("1", "true", "yes")


def append(md: List[str], ctx: SummaryContext) -> None:
    md.append("\n### 🤖 Trigger Likelihood v0 (next 6h)")

    # Import the model at call-time so importing this module stays cheap and robust.
    try:
        from src.ml.infer import score as infer_score, model_metadata
    except Exception:
        md.append("_No score available._")
        return

    # Metadata line (created_at, AUC, demo flag)
    try:
        _meta = model_metadata() or {}
        _metrics = _meta.get("metrics", {}) or {}
        _auc = _metrics.get("roc_auc_va") or _metrics.get("roc_auc_tr")
        bits = []
        if _meta.get("created_at"):
            bits.append(f"model@{_meta['created_at']}")
        if _auc is not None:
            try:
                bits.append(f"AUC={float(_auc):.2f}")
            except Exception:
                bits.append(f"AUC={_auc}")
        if _meta.get("demo"):
            bits.append("demo")
        if bits:
            md.append("- " + " • ".join(bits))
    except Exception:
        pass

    # Choose up to 3 candidate origins
    candidates = list(ctx.candidates or []) or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=3)
    now_bucket = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()

    # Optional "rich" path uses summary analytics as features
    use_rich = _demo_rich_scores_enabled()
    if use_rich:
        md.append("_rich features on_")

    # Build/collect maps (prefer caches from prior sections; compute on-the-fly if missing)
    trends_map: Dict[str, List[Dict[str, Any]]] = {}
    regimes_map: Dict[str, Any] = {}
    metrics_map: Dict[str, Dict[str, float]] = {}
    bursts_map: Dict[str, List[Dict[str, Any]]] = {}
    leadership_by_origin: Dict[str, float] = {}

    # Try to reuse cached analytics populated by earlier sections
    try:
        t = ctx.caches.get("origin_trends") or {}
        for item in (t.get("origins") or []):
            o = item.get("origin")
            if not o: continue
            series = (
                item.get("series")
                or item.get("buckets")
                or item.get("data")
                or item.get("timeline")
                or []
            )
            # Normalize so each bucket has 'flags_count'
            norm = []
            for b in series:
                if not isinstance(b, dict):
                    continue
                if "flags_count" not in b:
                    bb = dict(b)
                    if "flags" in bb:
                        bb["flags_count"] = bb.get("flags", 0)
                    elif "count" in bb:
                        bb["flags_count"] = bb.get("count", 0)
                    else:
                        bb["flags_count"] = 0
                    norm.append(bb)
                else:
                    norm.append(b)
            trends_map[o] = norm
    except Exception:
        pass

    try:
        vr = ctx.caches.get("volatility_regimes") or {}
        for row in (vr.get("origins") or []):
            o = row.get("origin")
            if o:
                regimes_map[o] = (row.get("regime") or "normal")
    except Exception:
        pass

    try:
        sm = ctx.caches.get("source_metrics_7d") or {}
        for r in (sm.get("origins") or []):
            o = r.get("origin")
            if o:
                metrics_map[o] = {
                    "precision": float(r.get("precision", 0.0) or 0.0),
                    "recall": float(r.get("recall", 0.0) or 0.0),
                }
    except Exception:
        pass

    try:
        bd = ctx.caches.get("bursts_7d") or {}
        for item in (bd.get("origins") or []):
            o = item.get("origin")
            if o:
                bursts_map[o] = list(item.get("bursts", []) or [])
    except Exception:
        pass

    try:
        ll = ctx.caches.get("lead_lag_7d") or {}
        for p in (ll.get("pairs") or []):
            leader = p.get("leader"); corr = p.get("correlation")
            if leader is None or corr is None:
                continue
            try:
                v = abs(float(corr))
            except Exception:
                continue
            leadership_by_origin[leader] = max(leadership_by_origin.get(leader, 0.0), v)
    except Exception:
        pass

    # If caches were empty and we still want rich features, try computing on-the-fly
    if use_rich and not trends_map:
        try:
            from src.analytics.origin_trends import compute_origin_trends
            _tr = compute_origin_trends(
                ctx.logs_dir / "retraining_log.jsonl",
                ctx.logs_dir / "retraining_triggered.jsonl",
                days=7, interval="hour",
            )
            for item in _tr.get("origins", []) or []:
                o = item.get("origin")
                if not o: continue
                series = (
                    item.get("series")
                    or item.get("buckets")
                    or item.get("data")
                    or item.get("timeline")
                    or []
                )
                norm = []
                for b in series:
                    if not isinstance(b, dict):
                        continue
                    if "flags_count" not in b:
                        bb = dict(b)
                        if "flags" in bb:
                            bb["flags_count"] = bb.get("flags", 0)
                        elif "count" in bb:
                            bb["flags_count"] = bb.get("count", 0)
                        else:
                            bb["flags_count"] = 0
                        norm.append(bb)
                    else:
                        norm.append(b)
                trends_map[o] = norm
        except Exception:
            pass

    if use_rich and not regimes_map:
        try:
            from src.analytics.volatility_regimes import compute_volatility_regimes
            _vr = compute_volatility_regimes(
                ctx.logs_dir / "retraining_log.jsonl",
                ctx.logs_dir / "retraining_triggered.jsonl",
                days=30, interval="hour", lookback=72,
            )
            for r in _vr.get("origins", []) or []:
                o = r.get("origin")
                if o:
                    regimes_map[o] = (r.get("regime") or "normal")
        except Exception:
            pass

    if use_rich and not metrics_map:
        try:
            from src.analytics.source_metrics import compute_source_metrics
            _sm = compute_source_metrics(
                ctx.logs_dir / "retraining_log.jsonl",
                ctx.logs_dir / "retraining_triggered.jsonl",
                days=7, min_count=1,
            )
            for r in _sm.get("origins", []) or []:
                o = r.get("origin")
                if o:
                    metrics_map[o] = {
                        "precision": float(r.get("precision", 0.0) or 0.0),
                        "recall": float(r.get("recall", 0.0) or 0.0),
                    }
            # cache for other sections if useful
            ctx.caches.setdefault("source_metrics_7d", _sm)
        except Exception:
            pass

    if use_rich and not bursts_map:
        try:
            from src.analytics.burst_detection import compute_bursts
            _bd = compute_bursts(
                ctx.logs_dir / "retraining_log.jsonl",
                ctx.logs_dir / "retraining_triggered.jsonl",
                days=7, interval="hour", z_thresh=2.0,
            )
            for item in _bd.get("origins", []) or []:
                o = item.get("origin")
                if o:
                    bursts_map[o] = list(item.get("bursts", []) or [])
            ctx.caches.setdefault("bursts_7d", _bd)
        except Exception:
            pass

    if use_rich and not leadership_by_origin:
        try:
            from src.analytics.lead_lag import compute_lead_lag
            _ll = compute_lead_lag(
                ctx.logs_dir / "retraining_log.jsonl",
                ctx.logs_dir / "retraining_triggered.jsonl",
                days=7, interval="hour", max_lag=24, use="flags",
            )
            for p in _ll.get("pairs", []) or []:
                leader = p.get("leader"); corr = p.get("correlation")
                if leader is None or corr is None:
                    continue
                try:
                    v = abs(float(corr))
                except Exception:
                    continue
                leadership_by_origin[leader] = max(leadership_by_origin.get(leader, 0.0), v)
            ctx.caches.setdefault("lead_lag_7d", _ll)
        except Exception:
            pass

    # Optional synthesized rich features in demo if everything was zero-ish
    feats_cache: Dict[str, Dict[str, float]] = {}
    nonzero_seen = False

    if use_rich:
        for o in candidates:
            feats = _build_summary_features_for_origin(
                o,
                trends_by_origin=trends_map,
                regimes_map=regimes_map,
                metrics_map=metrics_map,
                bursts_by_origin=bursts_map,
            )
            feats["leadership_max_r"] = float(leadership_by_origin.get(o, 0.0))
            feats_cache[o] = feats
            if any(abs(v or 0.0) > 1e-12 for v in feats.values()):
                nonzero_seen = True

        # If demo mode + all zeros → synthesize plausible variety
        if (os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")) and not nonzero_seen:
            patterns = [
                {"count_1h": 3, "count_6h": 9, "count_24h": 18, "count_72h": 54, "burst_z": 1.2, "regime": "turbulent", "precision_7d": 0.35, "recall_7d": 0.25, "leadership_max_r": 0.40},
                {"count_1h": 1, "count_6h": 4, "count_24h": 10, "count_72h": 30, "burst_z": 0.6, "regime": "normal",     "precision_7d": 0.20, "recall_7d": 0.15, "leadership_max_r": 0.20},
                {"count_1h": 0, "count_6h": 2, "count_24h": 6,  "count_72h": 18, "burst_z": 0.0, "regime": "calm",       "precision_7d": 0.10, "recall_7d": 0.08, "leadership_max_r": 0.05},
            ]
            for idx, o in enumerate(candidates):
                p = patterns[min(idx, len(patterns) - 1)]
                feats = feats_cache.get(o, {
                    "count_1h": 0.0, "count_6h": 0.0, "count_24h": 0.0, "count_72h": 0.0,
                    "burst_z": 0.0,
                    "regime_calm": 0.0, "regime_normal": 0.0, "regime_turbulent": 0.0,
                    "precision_7d": 0.0, "recall_7d": 0.0,
                    "leadership_max_r": 0.0,
                })
                feats.update({
                    "count_1h": float(p["count_1h"]),
                    "count_6h": float(p["count_6h"]),
                    "count_24h": float(p["count_24h"]),
                    "count_72h": float(p["count_72h"]),
                    "burst_z": float(p["burst_z"]),
                    "precision_7d": float(p["precision_7d"]),
                    "recall_7d": float(p["recall_7d"]),
                    "leadership_max_r": float(p["leadership_max_r"]),
                    "regime_calm": 0.0, "regime_normal": 0.0, "regime_turbulent": 0.0,
                })
                rk = f"regime_{p['regime']}"
                if rk in feats:
                    feats[rk] = 1.0
                feats_cache[o] = feats
            nonzero_seen = True
            md.append("_(demo) rich features synthesized for display_")

    # Scoring loop
    printed = 0
    for o in candidates:
        try:
            if use_rich and feats_cache.get(o):
                res = infer_score({"features": feats_cache[o]})
            else:
                res = infer_score({"origin": o, "timestamp": now_bucket})
            p = res.get("prob_trigger_next_6h")
            if isinstance(p, (int, float)):
                md.append(f"- {o}: **{round(float(p)*100,1)}%** chance of trigger in next 6h")

                # Show top contributions if available
                contribs = res.get("contributions")
                if isinstance(contribs, dict) and contribs:
                    top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                    md.append("  (" + ", ".join(f"{k}={v:+.2f}" for k, v in top) + ")")
                printed += 1
        except Exception:
            continue

    if printed == 0:
        # Deterministic fallback probe
        try:
            res = infer_score({"features": {"burst_z": 2.0}})
        except Exception:
            res = {"prob_trigger_next_6h": 0.0}
        md.append(f"- example (burst_z=2.0): **{round(float(res.get('prob_trigger_next_6h', 0))*100,1)}%**")

    # Small interpretability/coverage sub-block
    try:
        _m = model_metadata() or {}
        tfeat = _m.get("top_features") or []
        covsum = _m.get("feature_coverage_summary") or _m.get("feature_coverage") or {}
        low_cov = []
        if isinstance(covsum, dict):
            for k, v in list(covsum.items())[:]:
                try:
                    pct = float(v if isinstance(v, (int, float)) else v.get("nonzero_pct", 0.0))
                except Exception:
                    pct = 0.0
                if pct < 5.0:
                    low_cov.append(k)
        if tfeat:
            md.append("\n_top learned features_: " + ", ".join(f"{d['feature']}({d['coef']:+.2f})" for d in tfeat if isinstance(d, dict) and "feature" in d and "coef" in d))
        if low_cov:
            md.append("_low coverage_: " + ", ".join(sorted(set(low_cov))[:5]))
    except Exception:
        pass