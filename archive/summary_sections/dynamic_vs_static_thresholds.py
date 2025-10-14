# scripts/summary_sections/dynamic_vs_static_thresholds.py

from scripts.summary_sections.common import pick_candidate_origins
from src.ml.recent_scores import load_recent_scores, dynamic_threshold_for_origin
from src.ml.thresholds import load_per_origin_thresholds

def append(md, ctx, **kwargs):
    md.append("\n### 🎚️ Dynamic vs Static Thresholds (48h)")
    try:
        cands = [o for o in (ctx.candidates or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=3)) if o != "unknown"][:2]
        if not cands:
            md.append("_No candidate origins available._")
            return

        recent = load_recent_scores()
        static_map = {}
        try:
            static_map = load_per_origin_thresholds() or {}
        except Exception:
            static_map = {}

        dyn_used = {}
        for o in cands:
            dyn, n_recent, static_prob_default = dynamic_threshold_for_origin(o, recent=recent, min_samples=2)
            # try read probability-scale static from file
            st = static_prob_default
            try:
                vals = static_map.get(o) or {}
                for k in ("p80_proba","p70_proba","proba"):
                    if k in vals and 0.0 <= float(vals[k]) <= 1.0:
                        st = float(vals[k]); break
            except Exception:
                pass
            used = dyn if dyn is not None else st
            def _fmt(v):
                try: return f"{float(v):.3f}"
                except Exception: return "n/a"
            md.append(f"- `{o}`: dyn={_fmt(dyn)} ({n_recent} pts) | static={_fmt(st)} → used={_fmt(used)}")
            dyn_used[o] = {"dynamic": dyn, "static": st, "used": used}

        ctx.caches["dyn_thresholds"] = dyn_used
    except Exception as e:
        md.append(f"_⚠️ Dynamic threshold section failed: {e}_")
        ctx.caches["dyn_thresholds"] = {}