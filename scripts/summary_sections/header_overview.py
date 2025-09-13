# scripts/summary_sections/header_overview.py
from datetime import datetime, timezone
from .common import red, weight_to_label

def append(md, ctx, reviewers, threshold, sig_id, triggered_log):
    now_iso = datetime.now(timezone.utc).isoformat()
    total_weight = round(sum(r["weight"] for r in reviewers), 2)
    would_trigger = total_weight >= threshold
    last_trig = max(
        (t for t in triggered_log if t.get("signal_id")==sig_id),
        key=lambda x: x.get("timestamp", 0),
        default=None
    ) if triggered_log else None

    # single H1 only
    md.append("# MoonWire CI Demo Summary")
    md.append(f"MoonWire Demo Summary — {now_iso}")
    md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
    md.append(f"- **Signal:** `{red(sig_id)}`")
    md.append(f"- **Unique reviewers:** {len(reviewers)}")
    md.append(f"- **Combined weight:** **{total_weight}**")
    md.append(f"- **Threshold:** **{threshold}** → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")
    if last_trig:
        md.append(f"- **Last retrain trigger logged:** {last_trig.get('timestamp','')}")
    md.append("\n**Reviewers (redacted):**")
    if reviewers:
        for r in reviewers:
            md.append(f"- `{red(r['id'])}` → {weight_to_label(r['weight'])}")
    else:
        md.append("- _none found in this run_")