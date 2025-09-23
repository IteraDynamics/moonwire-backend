#!/usr/bin/env python3
"""

MoonWire CI Demo Summary (orchestrator-only)

Writes:
 - artifacts/demo_summary.md

This file only orchestrates; each section lives in scripts/summary_sections/*.py
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json

from typing import List

# Core paths
from src.paths import LOGS_DIR, MODELS_DIR
from src.analytics.origin_utils import compute_origin_breakdown

# ---- Shared context & helpers (from common.py) ----
from scripts.summary_sections.common import (
    SummaryContext,
    is_demo_mode,
    generate_demo_data_if_needed,
    parse_ts,
    band_weight_from_score,
    weight_to_label,
    red,
    pick_candidate_origins,
)

# Optional demo helper (present in common.py). If not present yet, no-op.
try:
    from scripts.summary_sections.common import generate_demo_origins_if_needed
except Exception:
    def generate_demo_origins_if_needed(x): return x

# ---- Sections (one module per section) ----
from scripts.summary_sections import (
    header_overview,
    source_yield_plan,
    origin_trends,
    cross_origin_correlations,
    lead_lag,
    volatility_regimes,
    nowcast_attention,
    trigger_likelihood_v0,
    ensemble_v0_4,
    dynamic_vs_static_thresholds,
    score_distribution,               # global 48h
    score_distribution_per_origin,    # per-origin 48h with drift overlay
    volatility_aware_thresholds,
    trigger_explainability,
    trigger_history,
    label_feedback,
    rolling_accuracy_snapshot,
    accuracy_by_model_version,
    training_data_snapshot,
    retrain_summary,
    latest_training_metadata,
    calibration,
    per_origin_thresholds,
    drift_aware_inference,
    drift_check,
    live_backtest,
    burst_detection,
    source_precision_recall,
    signal_quality,                   # (v0.5.5) batches over time
    signal_quality_per_origin,        # (v0.5.6)
    signal_quality_per_version,       # (v0.5.10) + trend
    calibration_reliability,
    calibration_per_origin,
    threshold_quality_per_origin,     # (v0.5.8)
    threshold_recommendations,
    threshold_backtest,
    threshold_auto_apply,  # v0.6.2
    trigger_coverage_summary,         # (v0.5.12)
    trigger_precision_by_origin,      # (v0.5.13)
    suppression_rate_by_origin,       # (v0.5.15)
    trigger_coverage_trend,           # (v0.5.14) trend chart
    trigger_suppression_trend,
    calibration_reliability_trend,    # ⬅️ NEW v0.6.5 (imported so we can call it)
)

# ---- Compatibility re-exports for any tests that import from mw_demo_summary ----
# (These names match what some tests might import directly.)
from scripts.summary_sections.common import (
    generate_demo_data_if_needed as generate_demo_data_if_needed,
    is_demo_mode as is_demo_mode,
    red as red,
    band_weight_from_score as band_weight_from_score,
    weight_to_label as weight_to_label,
    parse_ts as parse_ts,
    pick_candidate_origins as pick_candidate_origins,
)

# ---------- config ----------
ART = Path("artifacts")
ART.mkdir(exist_ok=True)
DEFAULT_THRESHOLD = 2.5


# ---------- tiny local utils ----------
def _load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def main():
    md: List[str] = []

    # ---------- maybe ensure demo has a couple rows ----------
    def _maybe_seed_real_logs_if_empty():
        if not is_demo_mode():
            return
        retrain_path = LOGS_DIR / "retraining_log.jsonl"
        try:
            if retrain_path.exists() and any(ln.strip() for ln in retrain_path.read_text().splitlines()):
                return
        except Exception:
            pass
        try:
            from scripts.demo_seed_reviewers import seed_once
            seed_once()
        except Exception:
            pass

    _maybe_seed_real_logs_if_empty()

    # ---------- load logs ----------
    retrain_log   = _load_jsonl(LOGS_DIR / "retraining_log.jsonl")
    triggered_log = _load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
    scores_log    = _load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
    score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

    # ---------- latest signal ----------
    if retrain_log:
        def _key(r):
            t = r.get("timestamp", 0)
            try:
                return float(t)
            except Exception:
                return 0.0
        latest = max(retrain_log, key=_key)
        sig_id = latest.get("signal_id", "unknown")
        sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
    else:
        sig_id = "none"
        sig_rows = []

    # ---------- reviewers & weights ----------
    seen = set()
    reviewers = []
    flag_times = []
    for r in sorted(sig_rows, key=lambda x: x.get("timestamp", 0)):
        t = parse_ts(r.get("timestamp"))
        if t:
            flag_times.append(t)
        rid = r.get("reviewer_id", "")
        if rid in seen:
            continue
        seen.add(rid)
        w = r.get("reviewer_weight")
        if w is None:
            sc = (score_by_id.get(rid) or {}).get("score")
            w = band_weight_from_score(sc)
        reviewers.append({"id": rid, "weight": round(float(w), 2)})

    reviewers, _ = generate_demo_data_if_needed(reviewers, flag_times)

    # ---------- compute origins ----------
    try:
        origins_rows, _ = compute_origin_breakdown(
            flags_path=LOGS_DIR / "retraining_log.jsonl",
            triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
            days=7,
            include_triggers=True,
        )
    except Exception:
        origins_rows = []

    origins_rows = generate_demo_origins_if_needed(origins_rows)

    # ---------- build context ----------
    ctx = SummaryContext(
        logs_dir=LOGS_DIR,
        models_dir=MODELS_DIR,
        is_demo=is_demo_mode(),
        origins_rows=origins_rows,
        yield_data=None,
        candidates=[],
        caches={},  # modules may place heavy intermediates here
    )

    # ---------- call sections in the same order as the last green CI ----------
    header_overview.append(md, ctx, reviewers=reviewers, threshold=DEFAULT_THRESHOLD,
                           sig_id=sig_id, triggered_log=triggered_log)
    # 1) Supply & stream activity (sets context like ctx.yield_data, trends)
    source_yield_plan.append(md, ctx)
    origin_trends.append(md, ctx)
    burst_detection.append(md, ctx)              # move earlier: belongs with “activity”
    cross_origin_correlations.append(md, ctx)
    lead_lag.append(md, ctx)
    volatility_regimes.append(md, ctx)
    nowcast_attention.append(md, ctx)

    # 2) Scoring & score shapes
    trigger_likelihood_v0.append(md, ctx)
    ensemble_v0_4.append(md, ctx)
    score_distribution.append(md, ctx)
    score_distribution_per_origin.append(md, ctx)
    calibration_reliability.append(md, ctx)
    calibration_per_origin.append(md, ctx)
    calibration_reliability_trend.append(md, ctx)   # v0.6.5 — new trend chart

    # 3) Thresholds & explainability (what would/wouldn’t fire, and why)
    dynamic_vs_static_thresholds.append(md, ctx)
    volatility_aware_thresholds.append(md, ctx)
    per_origin_thresholds.append(md, ctx)
    threshold_quality_per_origin.append(md, ctx)
    threshold_recommendations.append(md, ctx)
    threshold_backtest.append(md, ctx) # v 0.6.1
    threshold_auto_apply.append(md, ctx) # v0.6.2
    trigger_explainability.append(md, ctx)

    # 4) Decisions & labels (what actually happened)
    trigger_history.append(md, ctx)
    label_feedback.append(md, ctx)

    # 5) Quality & coverage (readability: coverage → suppression → precision → trend)
    signal_quality.append(md, ctx)
    signal_quality_per_origin.append(md, ctx)
    signal_quality_per_version.append(md, ctx)
    trigger_coverage_summary.append(md, ctx)
    suppression_rate_by_origin.append(md, ctx)   # sits right next to coverage
    trigger_precision_by_origin.append(md, ctx)
    trigger_coverage_trend.append(md, ctx)       # chart after the summaries it visualizes
    trigger_suppression_trend.append(md, ctx)

    # 6) Performance snapshots
    rolling_accuracy_snapshot.append(md, ctx)
    accuracy_by_model_version.append(md, ctx)
    source_precision_recall.append(md, ctx)
    calibration.append(md, ctx)

    # 7) Training & drift (pipeline artifacts & health checks)
    training_data_snapshot.append(md, ctx)
    retrain_summary.append(md, ctx)
    latest_training_metadata.append(md, ctx)
    drift_check.append(md, ctx)
    drift_aware_inference.append(md, ctx)
    live_backtest.append(md, ctx)

    # ---------- write once ----------
    out = ART / "demo_summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()