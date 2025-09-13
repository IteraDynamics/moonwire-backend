# scripts/summary_sections/trigger_explainability.py
from __future__ import annotations
from scripts.summary_sections.common import SummaryContext

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🧠 Trigger Explainability")
    try:
        from src.ml.infer import infer_score_ensemble
    except Exception as e:
        md.append(f"_unavailable: {type(e).__name__}_")
        return

    origins_list = list(ctx.candidates or []) or ["reddit", "twitter", "rss_news"]

    feats_cache = ctx.caches.get("feats_cache", {}) or {}
    trends_map = ctx.caches.get("trends_map", {})
    regimes_map = ctx.caches.get("regimes_map", {})
    metrics_map = ctx.caches.get("metrics_map", {})
    bursts_map  = ctx.caches.get("bursts_map", {})
    dyn_map     = ctx.caches.get("dyn_thresholds", {}) or {}

    from .trigger_likelihood_v0 import _build_summary_features_for_origin

    shown = 0
    for o in origins_list:
        if shown >= 2:
            break
        feats = feats_cache.get(o)
        if feats is None:
            feats = _build_summary_features_for_origin(
                o, trends_by_origin=trends_map, regimes_map=regimes_map,
                metrics_map=metrics_map, bursts_by_origin=bursts_map,
            )

        payload = {"features": dict(feats or {})}
        used = (dyn_map.get(o) or {}).get("used")
        if used is not None:
            try:
                payload["base_threshold"] = float(used)
            except Exception:
                pass

        try:
            res = infer_score_ensemble(payload)
        except Exception:
            continue

        expl = res.get("explanation", {}) or {}
        regime     = (expl.get("volatility_regime") or res.get("volatility_regime") or "normal")
        drift_pen  = float(res.get("drift_penalty", 0.0) or 0.0)
        adj_score  = float(res.get("adjusted_score", res.get("prob_trigger_next_6h", 0.0)) or 0.0)

        base_thr_v = res.get("base_threshold")
        try: base_thr_v = float(base_thr_v) if base_thr_v is not None else 0.5
        except Exception: base_thr_v = 0.5

        adj_thr_v  = res.get("threshold_after_volatility")
        try: adj_thr_v = float(adj_thr_v) if adj_thr_v is not None else None
        except Exception: adj_thr_v = None

        thr_show = adj_thr_v if isinstance(adj_thr_v, (int, float)) else base_thr_v
        decision = expl.get("decision") or ("triggered" if adj_score >= thr_show else "not_triggered")

        top_feats = expl.get("top_contributors") or []
        if not top_feats and isinstance(payload.get("features"), dict):
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
            f"threshold: base={base_thr_v:.3f} → adjusted={thr_show:.3f} "
            f"(regime={regime}, drift_penalty={drift_pen:.2f})"
        )
        if top_feats:
            md.append(f"  - top contributors: {', '.join(map(str, top_feats))}")
        shown += 1