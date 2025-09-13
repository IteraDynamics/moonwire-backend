# scripts/summary_sections/trigger_history.py
from datetime import datetime, timezone
import json

def _parse_ts_iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)

def _ensure_v_prefix(v):
    v = str(v or "unknown")
    return v if v.startswith("v") else f"v.{v}" if v and v[0].isalpha() else f"v{v}"

def append(md, ctx):
    try:
        hist_path = ctx.models_dir / "trigger_history.jsonl"
        rows = []
        if hist_path.exists():
            for ln in hist_path.read_text(encoding="utf-8").splitlines()[-64:]:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass

        # last 3 unique by (origin, timestamp, decision, score)
        seen = set()
        tail = []
        for r in reversed(rows):
            key = (
                r.get("origin","unknown"),
                r.get("timestamp",""),
                r.get("decision",""),
                round(float(r.get("adjusted_score", 0.0) or 0.0), 4),
            )
            if key in seen:
                continue
            seen.add(key)
            tail.append(r)
            if len(tail) >= 3:
                break
        tail = list(reversed(tail))

        md.append("\n🗂️ Trigger History (Last 3)")
        if not tail:
            md.append("(waiting for events…)")
            return

        for row in tail:
            ts = _parse_ts_iso(row.get("timestamp"))
            hhmm = ts.strftime("%H:%M")
            origin = row.get("origin", "unknown")
            decision = row.get("decision", "unknown")
            check = "✅ triggered" if decision == "triggered" else "❌ not_triggered"

            try:
                score = float(row.get("adjusted_score", 0.0) or 0.0)
            except Exception:
                score = 0.0

            thr = row.get("threshold", None)
            regime = row.get("volatility_regime") or "n/a"
            ver = _ensure_v_prefix(row.get("model_version", "unknown"))
            drift = row.get("drifted_features") or []
            drift_txt = "none" if not drift else ", ".join(map(str, drift[:2])) + ("…" if len(drift) > 2 else "")

            if isinstance(thr, (int, float)):
                md.append(f"- [{hhmm}] {origin} → {check} @ {score:.2f} (thr={thr:.2f}) — {regime} — {ver} (drift: {drift_txt})")
            else:
                md.append(f"- [{hhmm}] {origin} → {check} @ {score:.2f} — {regime} — {ver}")

    except Exception as e:
        md.append("\n🗂️ Trigger History (Last 3)")
        md.append(f"⚠️ trigger history failed: {type(e).__name__}: {e}")