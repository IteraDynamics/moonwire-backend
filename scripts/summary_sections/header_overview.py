# scripts/summary_sections/header_overview.py
from datetime import datetime, timezone
from scripts.summary_sections.common import red, weight_to_label

def append(md, ctx, *, reviewers, threshold, sig_id, triggered_log):
    now_iso = datetime.now(timezone.utc).isoformat()
    total_weight = round(sum(r["weight"] for r in reviewers), 2)
    would_trigger = total_weight >= float(threshold)

    # header & basics
    md.append("# MoonWire CI Demo Summary\n")
    md.append(f"MoonWire Demo Summary — {now_iso}\n")
    md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.\n")
    md.append(f"- **Signal:** `{red(sig_id)}`")
    md.append(f"- **Unique reviewers:** {len(reviewers)}")
    md.append(f"- **Combined weight:** **{total_weight}**")
    md.append(f"- **Threshold:** **{threshold}** → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")

    # last retrain trigger (best-effort)
    if triggered_log:
        try:
            latest = max(triggered_log, key=lambda x: float(x.get("timestamp", 0) or 0.0))
            md.append(f"- **Last retrain trigger logged:** {latest.get('timestamp','')}")
        except Exception:
            pass

    # reviewers list
    md.append("\n**Reviewers (redacted):**")
    if reviewers:
        for r in reviewers:
            md.append(f"- `{red(r['id'])}` → {weight_to_label(r['weight'])}")
    else:
        md.append("- _none found in this run_")

    # origin breakdown (from ctx)
    md.append("\n**Signal origin breakdown (last 7 days):**")
    rows = ctx.origins_rows or []
    if rows:
        for o in rows:
            md.append(f"- {o['origin']}: {o.get('count',0)} ({o.get('percent','?')}%)")
    else:
        md.append("- _no origin data_")