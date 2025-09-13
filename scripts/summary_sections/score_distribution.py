from pathlib import Path
from datetime import datetime, timezone, timedelta
import statistics as S
import matplotlib.pyplot as plt

from .common import pick_candidate_origins

def append(md: list[str], ctx, hours: int = 48, min_points: int = 8):
    md.append("\n### 📐 Score Distribution (48h)")

    # Load recent scores + dynamic threshold helper
    try:
        from src.ml.recent_scores import load_recent_scores, dynamic_threshold_for_origin
    except Exception as e:
        md.append(f"_⚠️ recent score loader unavailable: {e}_")
        return

    # Optional static thresholds (probability scale if present)
    try:
        from src.ml.thresholds import load_per_origin_thresholds
        per_origin = load_per_origin_thresholds()
    except Exception:
        per_origin = {}

    # Pick one origin (prefer ctx.candidates → yield plan → origins_rows)
    candidates = ctx.candidates or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=1)
    origin = (candidates[0] if candidates else None)

    # Collect probabilities within window
    recent = load_recent_scores()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    def _take(filter_origin):
        return [
            r.proba for r in recent
            if r.proba is not None
            and r.ts >= cutoff
            and (filter_origin is None or r.origin == filter_origin)
        ]

    probas = _take(origin)
    label = origin or "overall"

    # Fallback to overall if too few points
    if len(probas) < min_points:
        probas = _take(None)
        label = "overall"

    # Demo seeding if still sparse
    if len(probas) < min_points and ctx.is_demo:
        import random
        probas = [max(0.0, min(1.0, random.betavariate(1.5, 8.0))) for _ in range(64)]
        md.append("_(demo) synthesized scores for visibility_")

    if not probas:
        md.append("_No recent scores to plot._")
        return

    # Thresholds: dynamic (if available) + best-effort static
    dyn, n_recent, static_default = dynamic_threshold_for_origin(origin or "", recent=recent, min_samples=5)
    static = static_default
    try:
        vals = per_origin.get(origin or "", {})
        for k in ("p80_proba", "p70_proba", "proba"):
            v = vals.get(k)
            if v is not None and 0.0 <= float(v) <= 1.0:
                static = float(v)
                break
    except Exception:
        pass
    used = dyn if dyn is not None else static

    # Plot (MPLBACKEND=Agg is already set in CI)
    try:
        ART = Path("artifacts"); ART.mkdir(exist_ok=True)
        fig = plt.figure(figsize=(5.0, 2.1))
        plt.hist(probas, bins=20, range=(0.0, 1.0))  # no colors/styles per guidelines
        if dyn is not None:
            plt.axvline(dyn, linestyle="--", linewidth=1)
        if static is not None:
            plt.axvline(static, linestyle=":", linewidth=1)
        plt.title(f"{label} — score distribution ({hours}h)")
        plt.tight_layout()
        out = ART / f"score_dist_{label.replace(' ', '_')}.png"
        fig.savefig(out)
        plt.close(fig)
        md.append(f"![]({out.as_posix()})")
    except Exception as e:
        md.append(f"_⚠️ plot failed: {e}_")

    # Quick stats
    try:
        sorted_p = sorted(probas)
        p50 = sorted_p[len(sorted_p)//2]
        p90 = sorted_p[int(0.9 * (len(sorted_p)-1))]
        _fmt = lambda x: f"{float(x):.3f}"
        md.append(f"- n={len(probas)}, mean={_fmt(S.fmean(probas))}, median={_fmt(p50)}, p90={_fmt(p90)}")
        md.append(f"- thresholds: dyn={_fmt(dyn) if dyn is not None else 'n/a'} ({n_recent} pts) | static={_fmt(static)} → used={_fmt(used)}")
    except Exception:
        pass