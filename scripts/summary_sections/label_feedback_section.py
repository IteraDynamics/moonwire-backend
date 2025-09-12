# scripts/summary_sections/label_feedback_section.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List
from datetime import datetime, timezone, timedelta


def _load_jsonl_safe(p: Path) -> list:
    if not p.exists():
        return []
    out: list = []
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


def _ts_key(r) -> datetime:
    """Parse timestamp to aware UTC; return epoch start on failure for stable sort."""
    try:
        s = str(r.get("timestamp", ""))
        s = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def render(md: List[str], models_dir: Path | str) -> None:
    """
    Append the '🟨 Label Feedback' section to the markdown list.

    Args:
        md: markdown buffer (list of lines) to mutate
        models_dir: base models directory (Path or str)
    """
    try:
        mdir = Path(models_dir)
        md.append("\n### 🟨 Label Feedback")

        feedback_path = mdir / "label_feedback.jsonl"
        rows = _load_jsonl_safe(feedback_path)

        # Demo seeding if empty
        demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
        if not rows and demo_mode:
            now = datetime.now(timezone.utc)
            # pull a demo model version from training_version.txt if present
            try:
                tv_path = mdir / "training_version.txt"
                demo_mv = tv_path.read_text(encoding="utf-8").strip() if tv_path.exists() else "v0.0.0-demo"
                if not isinstance(demo_mv, str) or not demo_mv:
                    demo_mv = "v0.0.0-demo"
                if not demo_mv.startswith("v"):
                    demo_mv = f"v{demo_mv}"
            except Exception:
                demo_mv = "v0.0.0-demo"

            rows = [
                {
                    "timestamp": (now - timedelta(minutes=40)).isoformat(),
                    "origin": "reddit",
                    "adjusted_score": 0.72,
                    "label": True,
                    "reviewer": "demo_reviewer",
                    "model_version": demo_mv,
                },
                {
                    "timestamp": (now - timedelta(minutes=65)).isoformat(),
                    "origin": "rss_news",
                    "adjusted_score": 0.44,
                    "label": False,
                    "reviewer": "demo_reviewer",
                    "model_version": demo_mv,
                },
                {
                    "timestamp": (now - timedelta(minutes=90)).isoformat(),
                    "origin": "twitter",
                    "adjusted_score": 0.68,
                    "label": True,
                    "reviewer": "demo_reviewer",
                    "model_version": demo_mv,
                },
            ]

        if not rows:
            md.append("_No feedback yet._")
            return

        # Show last 3 by timestamp (best-effort)
        rows_sorted = sorted(rows, key=_ts_key, reverse=True)
        last3 = rows_sorted[:3]

        for r in last3:
            ts = _ts_key(r)
            hhmm = ts.strftime("%H:%M")
            o = r.get("origin", "unknown")
            ok = bool(r.get("label", False))
            score = float(r.get("adjusted_score", 0.0) or 0.0)
            mv = r.get("model_version", "unknown")
            mv_str = mv if (isinstance(mv, str) and str(mv).startswith("v")) else f"v{mv}"
            mark = "✅ confirmed" if ok else "❌ rejected"
            md.append(f"- {o} @ {hhmm} → {mark} (score {score:.2f}, {mv_str})")

        # Small stats
        pos = sum(1 for r in rows if bool(r.get("label", False)))
        neg = sum(1 for r in rows if not bool(r.get("label", False)))
        try:
            pos_scores = [float(r.get("adjusted_score", 0.0)) for r in rows if bool(r.get("label", False))]
            neg_scores = [float(r.get("adjusted_score", 0.0)) for r in rows if not bool(r.get("label", False))]
            avg_pos = (sum(pos_scores) / len(pos_scores)) if pos_scores else 0.0
            avg_neg = (sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0
            md.append(f"- totals: true={pos} | false={neg}")
            md.append(f"- avg score: positives={avg_pos:.2f} | negatives={avg_neg:.2f}")
        except Exception:
            pass

    except Exception as e:
        md.append(f"\n⚠️ Label feedback section failed: {e}")