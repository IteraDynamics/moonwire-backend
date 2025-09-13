# scripts/summary_sections/label_feedback.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os, json
from scripts.summary_sections.common import SummaryContext

def _load_jsonl_safe(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    except Exception:
        return []
    return out

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🟨 Label Feedback")
    try:
        feedback_path = ctx.models_dir / "label_feedback.jsonl"
        rows = _load_jsonl_safe(feedback_path)

        if not rows and os.getenv("DEMO_MODE","false").lower() in ("1","true","yes"):
            now = datetime.now(timezone.utc)
            try:
                tv_path = ctx.models_dir / "training_version.txt"
                demo_mv = tv_path.read_text(encoding="utf-8").strip() if tv_path.exists() else "v0.0.0-demo"
                if not isinstance(demo_mv, str) or not demo_mv:
                    demo_mv = "v0.0.0-demo"
                if not demo_mv.startswith("v"):
                    demo_mv = f"v{demo_mv}"
            except Exception:
                demo_mv = "v0.0.0-demo"
            rows = [
                {"timestamp": (now - timedelta(minutes=40)).isoformat(), "origin": "reddit",   "adjusted_score": 0.72, "label": True,  "reviewer": "demo_reviewer", "model_version": demo_mv},
                {"timestamp": (now - timedelta(minutes=65)).isoformat(), "origin": "rss_news","adjusted_score": 0.44, "label": False, "reviewer": "demo_reviewer", "model_version": demo_mv},
                {"timestamp": (now - timedelta(minutes=90)).isoformat(), "origin": "twitter", "adjusted_score": 0.68, "label": True,  "reviewer": "demo_reviewer", "model_version": demo_mv},
            ]

        if not rows:
            md.append("_No feedback yet._")
            return

        def _ts_key(r):
            try:
                s = str(r.get("timestamp",""))
                s = s[:-1] + "+00:00" if s.endswith("Z") else s
                return datetime.fromisoformat(s).astimezone(timezone.utc)
            except Exception:
                return datetime.fromtimestamp(0, tz=timezone.utc)

        rows_sorted = sorted(rows, key=_ts_key, reverse=True)[:3]
        for r in rows_sorted:
            ts = _ts_key(r).strftime("%H:%M")
            o = r.get("origin","unknown")
            ok = bool(r.get("label", False))
            score = float(r.get("adjusted_score", 0.0) or 0.0)
            mv = r.get("model_version", "unknown")
            mv_str = mv if (isinstance(mv, str) and mv.startswith("v")) else f"v{mv}"
            mark = "✅ confirmed" if ok else "❌ rejected"
            md.append(f"- {o} @ {ts} → {mark} (score {score:.2f}, {mv_str})")

        pos = sum(1 for r in rows if bool(r.get("label", False)))
        neg = sum(1 for r in rows if not bool(r.get("label", False)))
        try:
            pos_scores = [float(r.get("adjusted_score", 0.0)) for r in rows if bool(r.get("label", False))]
            neg_scores = [float(r.get("adjusted_score", 0.0)) for r in rows if not bool(r.get("label", False))]
            avg_pos = (sum(pos_scores)/len(pos_scores)) if pos_scores else 0.0
            avg_neg = (sum(neg_scores)/len(neg_scores)) if neg_scores else 0.0
            md.append(f"- totals: true={pos} | false={neg}")
            md.append(f"- avg score: positives={avg_pos:.2f} | negatives={avg_neg:.2f}")
        except Exception:
            pass
    except Exception as e:
        md.append(f"\n⚠️ Label feedback section failed: {e}")