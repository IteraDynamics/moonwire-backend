# scripts/summary_sections/trigger_history.py
from __future__ import annotations
from datetime import datetime, timezone
import json
from scripts.summary_sections.common import SummaryContext

def append(md: list[str], ctx: SummaryContext):
    md.append("\n🗂️ Trigger History (Last 3)")
    try:
        hist_path = ctx.models_dir / "trigger_history.jsonl"
        last = []
        if hist_path.exists():
            for ln in hist_path.read_text(encoding="utf-8").splitlines()[-64:]:
                s = ln.strip()
                if not s:
                    continue
                try:
                    last.append(json.loads(s))
                except Exception:
                    pass
        last = last[-3:]

        if not last:
            md.append("(waiting for events…)")
            return

        def _hhmm(s):
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
                return dt.strftime("%H:%M")
            except Exception:
                return "??:??"

        for row in last:
            hhmm = _hhmm(row.get("timestamp", ""))
            origin = row.get("origin", "unknown")
            decision = row.get("decision", "unknown")
            check = "✅ triggered" if decision == "triggered" else "❌ not_triggered"
            score = float(row.get("adjusted_score", 0.0) or 0.0)
            thr = row.get("threshold", None)
            regime = row.get("volatility_regime", None)
            drift = row.get("drifted_features") or []
            drift_txt = "none" if not drift else ", ".join(drift[:2]) + ("" if len(drift) <= 2 else "…")
            ver = row.get("model_version", "unknown")

            if thr is None:
                md.append(f"[{hhmm}] {origin} → {check} @ 0.00{score:.2f}"[0:])  # safe formatting
                md.append(f"[{hhmm}] {origin} → {check} @ {score:.2f} — {regime or 'n/a'} — v{ver}")
            else:
                md.append(f"[{hhmm}] {origin} → {check} @ {score:.2f} (thr={thr:.2f}) — {regime or 'n/a'} — v{ver} (drift: {drift_txt})")
    except Exception as e:
        md.append(f"⚠️ trigger history failed: {type(e).__name__}: {e}")