# scripts/summary_sections/score_distribution.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple
import os, json

import numpy as np  # stable bins

# Headless plotting (CI-safe)
import matplotlib.pyplot as plt  # CI sets MPLBACKEND=Agg

from .common import SummaryContext


def _parse_ts(s: str | float | int | None) -> datetime | None:
    if s is None:
        return None
    # epoch seconds
    try:
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    except Exception:
        pass
    # ISO8601 (with Z)
    try:
        txt = str(s)
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _load_recent_scores(trigger_history: Path, hours: int = 48) -> Tuple[list[float], list[float]]:
    """
    Returns (non_drifted_scores, drifted_scores) from the last `hours`.
    Uses `adjusted_score` if available; otherwise falls back to
    `prob_trigger_next_6h` or `score`.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    non_drifted: list[float] = []
    drifted: list[float] = []

    if not trigger_history.exists():
        return non_drifted, drifted

    try:
        for ln in trigger_history.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except Exception:
                continue

            ts = _parse_ts(row.get("timestamp"))
            if not ts or ts < cutoff:
                continue

            # score best-effort
            val = (
                row.get("adjusted_score", None)
                if row.get("adjusted_score", None) is not None
                else row.get("prob_trigger_next_6h", None)
            )
            if val is None:
                val = row.get("score", None)
            try:
                sc = float(val)
            except Exception:
                continue

            drifted_features = row.get("drifted_features") or []
            if isinstance(drifted_features, list) and len(drifted_features) > 0:
                drifted.append(sc)
            else:
                non_drifted.append(sc)
    except Exception:
        # fall back silently — caller may synthesize in demo mode
        pass

    return non_drifted, drifted


def _seed_demo_if_needed(non_drifted: list[float], drifted: list[float], want_total: int = 64) -> Tuple[list[float], list[float], bool]:
    """
    If there aren't enough real points, synthesize a plausible split
    (non-drifted > drifted, slightly lower mean for drifted).
    Returns (non_drifted, drifted, seeded_flag).
    """
    n = len(non_drifted) + len(drifted)
    if n >= max(8, want_total // 4):
        return non_drifted, drifted, False

    rng = np.random.default_rng(42)
    n_total = max(want_total, 2 * max(1, n))
    n_drift = int(n_total * 0.25)
    n_ok = n_total - n_drift

    # Non-drifted: mean ~0.28, some tail to ~0.6
    nd = np.clip(rng.normal(loc=0.28, scale=0.11, size=n_ok), 0.0, 1.0)
    # Drifted: mean lower ~0.18, similar spread
    dr = np.clip(rng.normal(loc=0.18, scale=0.10, size=n_drift), 0.0, 1.0)

    return list(nd), list(dr), True


def _summary_stats(values: list[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(np.median(arr)), float(np.quantile(arr, 0.90))


def append(md: List[str], ctx: SummaryContext, window_hours: int = 48) -> None:
    """
    📐 Score Distribution (48h)
    - Reads scores from models/trigger_history.jsonl (last 48h)
    - Splits into drifted vs non-drifted buckets
    - Renders a dual histogram and embeds it
    - Prints summary stats and counts
    """
    md.append("\n### 📐 Score Distribution (48h)")

    trig_hist = ctx.models_dir / "trigger_history.jsonl"

    non_drifted_scores, drifted_scores = _load_recent_scores(trig_hist, hours=window_hours)
    seeded = False
    if ctx.is_demo:
        non_drifted_scores, drifted_scores, seeded = _seed_demo_if_needed(non_drifted_scores, drifted_scores)

    all_scores = non_drifted_scores + drifted_scores
    n_total = len(all_scores)
    mean_all, median_all, p90_all = _summary_stats(all_scores)

    if seeded:
        md.append("(demo) synthesized scores for visibility")

    md.append(f"- n={n_total}, mean={mean_all:.3f}, median={median_all:.3f}, p90={p90_all:.3f}")

    # --- threshold line metadata (simple, section-local) ---
    # If another section stored a 'used' threshold into ctx.caches, use it;
    # otherwise fall back to 0.5 so the plot always has a sensible line.
    thr_used = 0.5
    try:
        # e.g., ctx.caches.get("thresholds", {}).get("used", 0.5)
        cache_thr = ctx.caches.get("score_distribution_threshold_used")
        if cache_thr is not None:
            thr_used = float(cache_thr)
    except Exception:
        pass

    # Print the same short thresholds line you already used
    md.append(f"- thresholds: dyn=n/a (0 pts) | static={thr_used:.3f} → used={thr_used:.3f}")

    # --- group split + means (NEW) ---
    def _mean(xs: list[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    mean_nd = _mean(non_drifted_scores)
    mean_d  = _mean(drifted_scores)
    delta   = mean_nd - mean_d  # >0 means non-drifted scores higher
    md.append(f"- split: drifted={len(drifted_scores)} | non-drifted={len(non_drifted_scores)}")
    md.append(f"- group means: drifted={mean_d:.3f}, non-drifted={mean_nd:.3f}, Δ={delta:+.3f}")

    # --- histogram overlay (NEW) ---
    img_out = Path("artifacts") / "score_hist_drift_overlay_48h.png"
    try:
        bins = np.linspace(0.0, 1.0, 21)  # stable bin edges

        plt.figure(figsize=(6.5, 2.6))
        if non_drifted_scores:
            plt.hist(non_drifted_scores, bins=bins, alpha=0.85, label="non-drifted")
        if drifted_scores:
            plt.hist(drifted_scores,     bins=bins, alpha=0.85, label="drifted")

        plt.title("Score distribution (48h)")
        plt.xlabel("score")
        plt.ylabel("count")

        # vertical line at the used threshold
        try:
            plt.axvline(float(thr_used), linestyle="--", linewidth=1)
        except Exception:
            pass

        plt.legend()
        plt.tight_layout()
        plt.savefig(img_out)
        plt.close()

        # Embed inline
        md.append(f"  \n![]({img_out.as_posix()})")
    except Exception as e:
        md.append(f"_⚠️ histogram render failed: {type(e).__name__}: {e}_")