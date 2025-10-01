# scripts/summary_sections/trigger_explainability.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import SummaryContext, pick_candidate_origins, ensure_dir, parse_ts, _iso

# ---- Optional import: your TL v0 feature builder (kept from your version) ----
try:
    from .trigger_likelihood_v0 import _build_summary_features_for_origin
except Exception:
    def _build_summary_features_for_origin(
        origin: str,
        *,
        trends_by_origin: Dict[str, List[Dict[str, Any]]] | None = None,
        regimes_map: Dict[str, Any] | None = None,
        metrics_map: Dict[str, Dict[str, float]] | None = None,
        bursts_by_origin: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, float]:
        return {}

# ---- Optional import: demo explanation helper (fine if missing) --------------
try:
    from scripts.explain.explain_trigger import demo_explanation_for_origin as _demo_expl
except Exception:
    def _demo_expl(origin: str, k: int = 3) -> List[Dict[str, Any]]:
        tops = {
            "reddit": ["btc_return_1h", "reddit_burst_etf", "volatility_6h"],
            "twitter": ["solana_sentiment", "volatility_6h", "news_score"],
            "rss_news": ["sec_approval_term", "btc_price_jump", "liquidity_shock"],
        }.get(origin, ["feature_a", "feature_b", "feature_c"])[:k]
        return [{"feature": f, "contribution": round(0.25 - i * 0.07, 3)} for i, f in enumerate(tops)]

# ---- Optional: ensemble scorer (your existing path) --------------------------
def _try_import_infer():
    try:
        from src.ml.infer import infer_score_ensemble  # type: ignore
        return infer_score_ensemble
    except Exception:
        return None

# ---- Local utils -------------------------------------------------------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _pick_recent_fired(rows: List[Dict[str, Any]], now: datetime, lookback_h: int, limit: int) -> List[Dict[str, Any]]:
    cutoff = now - timedelta(hours=lookback_h)

    def _ts(r: Dict[str, Any]) -> datetime:
        t = r.get("ts") or r.get("timestamp")
        try:
            return parse_ts(t)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    def _is_fire(r: Dict[str, Any]) -> bool:
        d = (r.get("decision") or "").lower()
        if d == "fire":
            return True
        s = r.get("adjusted_score") or r.get("score")
        try:
            return float(s) >= 0.5
        except Exception:
            return False

    kept = [r for r in rows if _is_fire(r) and _ts(r) >= cutoff]
    kept.sort(key=_ts, reverse=True)
    return kept[:limit]

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def _plot_top_features_bar(art_dir: Path, top_counts: Dict[str, int]) -> None:
    ensure_dir(art_dir)
    fname = art_dir / "explainability_top_features.png"
    if not top_counts:
        # still produce an empty chart to satisfy artifact expectations
        plt.figure()
        plt.title("Top Features (none)")
        plt.savefig(str(fname))
        plt.close()
        return

    labels = list(top_counts.keys())
    values = [top_counts[k] for k in labels]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.title("Top Features across recent fired triggers")
    plt.tight_layout()
    plt.savefig(str(fname))
    plt.close()

@dataclass
class _Cfg:
    lookback_h: int = 72
    last_n: int = 20
    top_k: int = 3

# ---- Main entry --------------------------------------------------------------
def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Enhances your existing explainability section:
      1) Per-origin explanation via infer_score_ensemble (as you already had).
      2) Writes two artifacts:
         - models/explainability_sample.json  (last-N fired with explanations, or demo)
         - artifacts/explainability_top_features.png (bar of frequent top features)
      3) Adds a compact rollup line with most common top features per origin.
    """
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    logs_dir = Path(getattr(ctx, "logs_dir", "logs"))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    ensure_dir(models_dir)
    ensure_dir(artifacts_dir)

    cfg = _Cfg(
        lookback_h=72,
        last_n=20,
        top_k=3,
    )

    md.append("\n### 🧠 Trigger Explainability")

    infer_score_ensemble = _try_import_infer()
    # Prefer the same candidates used in TL v0; otherwise pick top 3
    origins_list = list(getattr(ctx, "candidates", []) or []) or pick_candidate_origins(
        getattr(ctx, "origins_rows", []), getattr(ctx, "yield_data", None), top=3
    )

    # Reuse cached analytics assembled by earlier sections when possible
    trends_map: Dict[str, List[Dict[str, Any]]] = {}
    regimes_map: Dict[str, Any] = {}
    metrics_map: Dict[str, Dict[str, float]] = {}
    bursts_map: Dict[str, List[Dict[str, Any]]] = {}
    dyn_thresholds: Dict[str, Dict[str, float]] = {}

    # origin trends
    try:
        t = (getattr(ctx, "caches", {}) or {}).get("origin_trends") or {}
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

    # regimes
    try:
        vr = (getattr(ctx, "caches", {}) or {}).get("volatility_regimes") or {}
        for row in (vr.get("origins") or []):
            o = row.get("origin")
            if o:
                regimes_map[o] = (row.get("regime") or "normal")
    except Exception:
        pass

    # metrics
    try:
        sm = (getattr(ctx, "caches", {}) or {}).get("source_metrics_7d") or {}
        for r in (sm.get("origins") or []):
            o = r.get("origin")
            if o:
                metrics_map[o] = {
                    "precision": float(r.get("precision", 0.0) or 0.0),
                    "recall": float(r.get("recall", 0.0) or 0.0),
                }
    except Exception:
    pass  # noqa

    # bursts
    try:
        bd = (getattr(ctx, "caches", {}) or {}).get("bursts_7d") or {}
        for item in (bd.get("origins") or []):
            o = item.get("origin")
            if o:
                bursts_map[o] = list(item.get("bursts", []) or [])
    except Exception:
        pass

    # dynamic thresholds
    try:
        dyn_thresholds = (getattr(ctx, "caches", {}) or {}).get("dyn_thresholds") or {}
    except Exception:
        dyn_thresholds = {}

    # ---- Per-origin explanation (kept from your version; with guards) ----
    shown = 0
    per_origin_top: Dict[str, List[str]] = {}
    for o in origins_list:
        if shown >= 2:
            break

        feats = {}
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

        base_thr = None
        try:
            drec = dyn_thresholds.get(o) or {}
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

        expl_top: List[str] = []
        if infer_score_ensemble is None:
            # No ensemble available -> demo explanation
            expl = _demo_expl(o, cfg.top_k)
            expl_top = [d["feature"] for d in expl]
            md.append(f"- **{o}**: _explain model unavailable → demo top_: {', '.join(expl_top)}")
        else:
            try:
                res = infer_score_ensemble(payload)  # type: ignore
                expl = res.get("explanation", {}) or {}
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
                decision = expl.get("decision") or ("triggered" if adj_score >= float(thr_show) else "not_triggered")

                top_feats = expl.get("top_contributors") or []
                if not top_feats and isinstance(payload.get("features"), dict):
                    # fallback heuristic if model didn't produce contributors
                    try:
                        numeric = [(k, float(v)) for k, v in payload["features"].items()
                                   if isinstance(v, (int, float))]
                        numeric.sort(key=lambda kv: abs(kv[1]), reverse=True)
                        top_feats = [k for k, _ in numeric[:3]]
                    except Exception:
                        top_feats = []

                expl_top = list(map(str, top_feats))[:cfg.top_k]

                md.append(f"- **{o}**: {decision}")
                md.append(
                    f"  - adjusted_score={adj_score:.3f}  "
                    f"threshold: base={base_thr_v:.3f}"
                    + (f" → adjusted={thr_show:.3f}" if adj_thr_v is not None else "")
                    + f" (regime={regime}, drift_penalty={drift_pen:.2f})"
                )
                if expl_top:
                    md.append(f"  - top contributors: {', '.join(expl_top)}")
            except Exception:
                # Ensemble call failed → demo
                expl = _demo_expl(o, cfg.top_k)
                expl_top = [d["feature"] for d in expl]
                md.append(f"- **{o}**: _explain failed → demo top_: {', '.join(expl_top)}")

        per_origin_top[o] = expl_top
        shown += 1

    # ---- Artifacts: sample JSON + top-features bar ---------------------------
    now = datetime.now(timezone.utc)
    sample_path = models_dir / "explainability_sample.json"

    # Prefer real logs if they already contain explanations
    trig = _load_jsonl(logs_dir / "trigger_history.jsonl")
    recent = _pick_recent_fired(trig, now, cfg.lookback_h, cfg.last_n * 3)
    rows_out: List[Dict[str, Any]] = []
    for r in recent:
        if r.get("explanation"):
            rows_out.append({
                "ts": r.get("ts") or r.get("timestamp"),
                "origin": r.get("origin", "unknown"),
                "model_version": r.get("model_version") or r.get("version") or "unknown",
                "adjusted_score": r.get("adjusted_score") or r.get("score"),
                "decision": r.get("decision", "fire"),
                "explanation": r.get("explanation"),
                "demo": False,
            })

    if rows_out:
        data = {
            "generated_at": _iso(now),
            "window_hours": cfg.lookback_h,
            "last_n": cfg.last_n,
            "top_k": cfg.top_k,
            "rows": rows_out[: cfg.last_n],
            "demo": False,
        }
    else:
        # Build deterministic demo sample based on per_origin_top (or internal demo map)
        demo_rows: List[Dict[str, Any]] = []
        subs = origins_list or ["reddit", "twitter", "rss_news"]
        for i, origin in enumerate(subs[:3]):
            # 6 demo rows/origin
            for j in range(1, 7):
                tops = per_origin_top.get(origin) or [d["feature"] for d in _demo_expl(origin, cfg.top_k)]
                demo_rows.append({
                    "ts": _iso(now - timedelta(hours=i * 3 + j)),
                    "origin": origin,
                    "model_version": f"v_demo_{i}",
                    "adjusted_score": 0.8,
                    "decision": "fire",
                    "explanation": [{"feature": f, "contribution": round(0.25 - k * 0.07, 3)} for k, f in enumerate(tops)],
                    "demo": True,
                })

        data = {
            "generated_at": _iso(now),
            "window_hours": cfg.lookback_h,
            "last_n": cfg.last_n,
            "top_k": cfg.top_k,
            "rows": demo_rows[: cfg.last_n],
            "demo": True,
        }

    _write_json(sample_path, data)

    # Aggregate for bar chart + concise rollup line
    top_counter: Counter[str] = Counter()
    by_origin: Dict[str, Counter[str]] = defaultdict(Counter)
    for r in data.get("rows", []):
        origin = r.get("origin", "unknown")
        for item in (r.get("explanation") or []):
            f = str(item.get("feature") or "").strip()
            if not f:
                continue
            top_counter[f] += 1
            by_origin[origin][f] += 1

    _plot_top_features_bar(artifacts_dir, dict(top_counter))

    # Compact rollup line
    demo_tag = " (demo)" if data.get("demo") else ""
    per_origin_lines = []
    for origin, ctr in sorted(by_origin.items()):
        common = [k for (k, _) in ctr.most_common(2)]
        if common:
            per_origin_lines.append(f"{origin} → top: {', '.join(common)}")
    if per_origin_lines:
        md.append(f"\n🔎 Top feature rollup{demo_tag}\n" + "\n".join(per_origin_lines))

    # Footer
    md.append("\n_Footer: Feature contributions estimated via model coefficients/importances (or ensemble-provided explainers)._")
