# scripts/summary_sections/trigger_explainability.py
from __future__ import annotations
from typing import List, Dict, Any

from .common import SummaryContext, pick_candidate_origins

# Import the feature builder from the TL v0 section (where it now lives)
try:
    from .trigger_likelihood_v0 import _build_summary_features_for_origin
except Exception:  # ultra-safe fallback: define a no-op builder
    def _build_summary_features_for_origin(
        origin: str,
        *,
        trends_by_origin: Dict[str, List[Dict[str, Any]]] | None = None,
        regimes_map: Dict[str, Any] | None = None,
        metrics_map: Dict[str, Dict[str, float]] | None = None,
        bursts_by_origin: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, float]:
        return {}


def append(md: List[str], ctx: SummaryContext) -> None:
    md.append("\n### 🧠 Trigger Explainability")

    # Import ensemble scorer lazily so this module stays light to import
    try:
        from src.ml.infer import infer_score_ensemble
    except Exception as e:
        md.append(f"_Explainability unavailable: {type(e).__name__}_")
        return

    # Prefer the same candidates used in TL v0; otherwise pick top 3
    origins_list = list(ctx.candidates or []) or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=3)
    if not origins_list:
        md.append("_No candidate origins available._")
        return

    # Reuse cached analytics assembled by earlier sections when possible
    trends_map: Dict[str, List[Dict[str, Any]]] = {}
    regimes_map: Dict[str, Any] = {}
    metrics_map: Dict[str, Dict[str, float]] = {}
    bursts_map: Dict[str, List[Dict[str, Any]]] = {}
    dyn_thresholds: Dict[str, Dict[str, float]] = {}

    try:
        t = ctx.caches.get("origin_trends") or {}
        for item in (t.get("origins") or []):
            o = item.get("origin")
            if not o:
                continue
            series = (
                item.get("series")
                or item.get("buckets")
                or item.get("data")
                or item.get("timeline")
                or []
            )
            # Ensure 'flags_count' exists for downstream feature sums
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

    # Dynamic threshold information (if a prior section stored it)
    try:
        dyn_thresholds = ctx.caches.get("dyn_thresholds") or {}
    except Exception:
        dyn_thresholds = {}

    shown = 0
    for o in origins_list:
        if shown >= 2:
            break

        # Build features (best effort)
        try:
            feats = _build_summary_features_for_origin(
                o,
                trends_by_origin=trends_map,
                regimes_map=regimes_map,
                metrics_map=metrics_map,
                bursts_by_origin=bursts_map,
            )
        except Exception:
            feats = {}

        # Base threshold from dynamic map if present (used as a hint; ensemble may override)
        base_thr = None
        try:
            drec = dyn_thresholds.get(o) or {}
            # prefer value that was actually "used", else dynamic, else static
            if "used" in drec:
                base_thr = float(drec["used"])
            elif "dynamic" in drec:
                base_thr = float(drec["dynamic"])
            elif "static" in drec:
                base_thr = float(drec["static"])
        except Exception:
            base_thr = None

        payload = {"features": dict(feats or {})}
        if base_thr is not None:
            payload["base_threshold"] = base_thr

        try:
            res = infer_score_ensemble(payload)
        except Exception:
            md.append(f"- **{o}**: _no explanation available_")
            continue

        expl = res.get("explanation", {}) or {}

        # Pull numbers with safe defaults for formatting
        regime     = (expl.get("volatility_regime")
                      or res.get("volatility_regime")
                      or "normal")
        drift_pen  = float(res.get("drift_penalty", 0.0) or 0.0)
        adj_score  = float(res.get("adjusted_score", res.get("prob_trigger_next_6h", 0.0)) or 0.0)

        base_thr_v = res.get("base_threshold")
        try:
            base_thr_v = float(base_thr_v) if base_thr_v is not None else 0.5
        except Exception:
            base_thr_v = 0.5

        adj_thr_v  = res.get("threshold_after_volatility")
        try:
            adj_thr_v = float(adj_thr_v) if adj_thr_v is not None else None
        except Exception:
            adj_thr_v = None

        thr_show = adj_thr_v if isinstance(adj_thr_v, (int, float)) else base_thr_v
        decision = expl.get("decision")
        if not decision:
            decision = "triggered" if adj_score >= thr_show else "not_triggered"

        top_feats = expl.get("top_contributors") or []
        if not top_feats and isinstance(payload.get("features"), dict):
            # Fallback heuristic: top absolute-valued features from the input
            try:
                numeric = [(k, float(v)) for k, v in payload["features"].items()
                           if isinstance(v, (int, float))]
                numeric.sort(key=lambda kv: abs(kv[1]), reverse=True)
                top_feats = [k for k, _ in numeric[:3]]
            except Exception:
                top_feats = []

        md.append(f"- **{o}**: {decision}")
        md.append(
            f"  - adjusted_score={adj_score:.3f}  "
            f"threshold: base={base_thr_v:.3f}"
            + (f" → adjusted={thr_show:.3f}" if adj_thr_v is not None else "")
            + f" (regime={regime}, drift_penalty={drift_pen:.2f})"
        )
        if top_feats:
            md.append(f"  - top contributors: {', '.join(map(str, top_feats))}")

        shown += 1