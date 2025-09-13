# scripts/summary_sections/drift_aware_inference.py
from __future__ import annotations
import os
from typing import Dict
from scripts.summary_sections.common import SummaryContext, pick_candidate_origins

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### ⚠️ Drift-Aware Inference")
    try:
        from src.ml.infer import infer_score_ensemble
    except Exception as e:
        md.append(f"_unavailable: {type(e).__name__}_")
        return

    cands = [o for o in (ctx.candidates or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=3)) if o != "unknown"][:3]
    feats_cache_local = ctx.caches.get("feats_cache", {}) or {}

    if not cands:
        md.append("_No candidate origins available._")
        return

    drift_counts = []
    drift_freq: Dict[str, int] = {}
    sample_line = None

    for o in cands:
        feats = feats_cache_local.get(o, {})
        res = infer_score_ensemble({"origin": o, "features": feats})
        drifted = list(res.get("drifted_features", []) or [])
        drift_counts.append(len(drifted))
        for k in drifted:
            drift_freq[k] = drift_freq.get(k, 0) + 1

        if sample_line is None and "adjusted_score" in res:
            try:
                s = float(res.get("ensemble_score", res.get("prob_trigger_next_6h")))
                a = float(res.get("adjusted_score"))
                pen = float(res.get("drift_penalty", 0.0))
                sample_line = f"- sample adjustment: score {s:.2f} → {a:.2f} (penalty={pen:.2f})"
            except Exception:
                pass

    if drift_counts:
        avg_drift = sum(drift_counts) / float(len(drift_counts))
        md.append(f"- avg drifted features per inference: {avg_drift:.2f}")
    else:
        md.append("- avg drifted features per inference: n/a")

    # derive a sample penalty based on env knobs (display-only)
    try:
        per_feat_pen = float(os.getenv("TL_DRIFT_PER_FEATURE_PENALTY", "0.05"))
        max_pen = float(os.getenv("TL_DRIFT_MAX_PENALTY", "0.5"))
    except Exception:
        per_feat_pen, max_pen = 0.05, 0.5

    avg_cnt = (sum(drift_counts) / len(drift_counts)) if drift_counts else 0.0
    sample_raw = 0.22
    pen = min(max_pen, per_feat_pen * avg_cnt)
    sample_adj = sample_raw * (1.0 - pen)
    if sample_line:
        md.append(sample_line)

    if drift_freq:
        top_feats = sorted(drift_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        md.append("- top drifted features: " + ", ".join(k for k, _ in top_feats))
    else:
        md.append("- top drifted features: _none_")