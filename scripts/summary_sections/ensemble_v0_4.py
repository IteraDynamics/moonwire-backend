# scripts/summary_sections/ensemble_v0_4.py

from scripts.summary_sections.common import pick_candidate_origins
from src.ml.infer import infer_score_ensemble

def append(md, ctx, **kwargs):
    md.append("\n**Ensemble v0.4 (mean ± band)**")
    try:
        cands = ctx.candidates or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=3)
        if not cands:
            md.append("_No candidate origins available._")
            return
        for o in cands:
            res = infer_score_ensemble({"origin": o})
            p   = res.get("prob_trigger_next_6h")
            lo  = res.get("low"); hi = res.get("high")
            if isinstance(p, (int, float)):
                if lo is not None and hi is not None:
                    md.append(f"- {o}: **{p*100:.1f}%** (±{(hi-lo)*50:.1f}%)")
                else:
                    md.append(f"- {o}: **{p*100:.1f}%**")
                votes = res.get("votes") or {}
                if votes:
                    vote_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in sorted(votes.items()))
                    md.append(f"  - votes: {vote_str}")
                if res.get("demo"):
                    md.append("  - _(demo fallback)_")
            else:
                md.append(f"- {o}: _no score_")
    except Exception as e:
        md.append(f"_⚠️ Ensemble score failed: {e}_")