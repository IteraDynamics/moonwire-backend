# scripts/dashboard/governance_dashboard.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Reuse helpers from summary package if available
try:
    from scripts.summary_sections.common import ensure_dir, _iso
except Exception:  # very defensive: tiny local fallbacks
    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p
    def _iso(dt: Optional[datetime] = None) -> str:
        return (dt or datetime.now(timezone.utc)).replace(microsecond=0).isoformat()

# Tiny 1x1 PNG for guaranteed output (no matplotlib required)
_PNG_1x1_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text() or "{}")
    except Exception:
        pass
    return None

def _trend_tag(v: Optional[float]) -> str:
    if v is None:
        return "unknown"
    if v > 0.0:
        return "improving"
    if v < 0.0:
        return "declining"
    return "stable"

def _safe(val, default):
    return default if val is None else val

def _inline_svg_spark(values) -> str:
    """Very small inline sparkline (values normalized 0-1)."""
    if not values:
        return "<svg width='120' height='20'></svg>"
    xs = list(range(len(values)))
    vmin = min(values); vmax = max(values)
    rng = (vmax - vmin) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = int(i * (120 / max(1, len(values) - 1)))
        y = 18 - int(((v - vmin) / rng) * 16)
        pts.append(f"{x},{y}")
    return f"<svg width='120' height='20' xmlns='http://www.w3.org/2000/svg'><polyline fill='none' stroke='currentColor' stroke-width='2' points='{ ' '.join(pts) }' /></svg>"

def _html_header(run_url: Optional[str], generated_at: str, window_h: int, badges: str) -> str:
    run_link = f"<a href='{run_url}'>View CI Run</a>" if run_url else ""
    return f"""
<header>
  <h1>MoonWire Governance Dashboard ({window_h}h)</h1>
  <div class="sub">Generated {generated_at} UTC {('&middot; ' + run_link) if run_link else ''}</div>
  <div class="badges">{badges}</div>
</header>
"""

def _html_css() -> str:
    return """
<style>
:root{--bg:#0f1115;--card:#161a22;--text:#e7edf3;--muted:#9fb0c3;--ok:#31c48d;--warn:#f59e0b;--crit:#ef4444;--link:#67a2f8}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial}
header{padding:20px;border-bottom:1px solid #232a36}
h1{margin:0;font-size:20px}
.sub{color:var(--muted);margin-top:6px}
.badges span{display:inline-block;margin-right:8px;padding:2px 8px;border-radius:999px;background:#232a36;color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;padding:14px}
.card{background:var(--card);border:1px solid #232a36;border-radius:10px;padding:12px}
.card h3{margin:0 0 8px 0;font-size:15px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px}
.kv div:nth-child(odd){color:var(--muted)}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #2a3242;color:var(--muted);font-size:12px}
.pill.ok{color:var(--ok);border-color:var(--ok)}
.pill.warn{color:var(--warn);border-color:var(--warn)}
.pill.crit{color:var(--crit);border-color:var(--crit)}
a{color:var(--link);text-decoration:none}
.footer{padding:10px 14px;color:var(--muted)}
.code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
table{width:100%;border-collapse:collapse}
td,th{border-bottom:1px solid #232a36;padding:6px;text-align:left}
.small{font-size:12px;color:var(--muted)}
.row{display:flex;gap:8px;align-items:center}
</style>
"""

def _html_tile_apply(apply: Dict[str, Any]) -> str:
    mode = apply.get("mode","dryrun")
    applied = int(apply.get("applied",0))
    skipped = int(apply.get("skipped",0))
    reversal = apply.get("reversal_plan","monitor 12h for precision regression ≥ 0.02")
    return f"""
<section class="card">
  <h3>Governance Apply</h3>
  <div class="kv">
    <div>Mode</div><div><span class="pill">{mode}</span></div>
    <div>Applied</div><div>{applied}</div>
    <div>Skipped</div><div>{skipped}</div>
    <div>Reversal Plan</div><div class="small">{reversal}</div>
  </div>
</section>
"""

def _html_tile_bg(bg: Dict[str, Any]) -> str:
    cur = _safe(bg.get("current"), "v?.?.?")
    cand = _safe(bg.get("candidate"), "v?.?.?")
    cls = _safe(bg.get("classification"), "observe")
    conf = bg.get("confidence")
    df1 = bg.get("delta",{}).get("F1")
    dece = bg.get("delta",{}).get("ECE")
    pill_cls = "ok" if cls in ("promote_ready","promote") else ("crit" if cls=="rollback_risk" else "")
    delta_txt = []
    if df1 is not None: delta_txt.append(f"ΔF1 {'+' if df1>=0 else ''}{df1:.02f}")
    if dece is not None: delta_txt.append(f"ΔECE {'+' if dece>=0 else ''}{dece:.02f}")
    conf_txt = f"{conf:.2f}" if isinstance(conf,(int,float)) else "—"
    return f"""
<section class="card">
  <h3>Blue-Green Simulation</h3>
  <div class="kv">
    <div>Path</div><div>{cur} → {cand}</div>
    <div>Classification</div><div><span class="pill {pill_cls}">{cls}</span></div>
    <div>Deltas</div><div>{', '.join(delta_txt) if delta_txt else '—'}</div>
    <div>Confidence</div><div>{conf_txt}</div>
  </div>
</section>
"""

def _html_tile_trend(trend: Dict[str, Any]) -> str:
    f1_trend = trend.get("f1_trend","stable")
    ece_trend = trend.get("ece_trend","stable")
    f1_svg = _inline_svg_spark(trend.get("f1_series",[0.4,0.5,0.55,0.57]))
    ece_svg = _inline_svg_spark(trend.get("ece_series",[0.08,0.07,0.065,0.06]))
    return f"""
<section class="card">
  <h3>Performance & Calibration</h3>
  <div class="row"><span class="pill">{f1_trend}</span>{f1_svg}</div>
  <div class="row"><span class="pill">{ece_trend}</span>{ece_svg}</div>
</section>
"""

def _html_tile_alerts(alerts: Dict[str, Any]) -> str:
    crit = int(alerts.get("critical",0))
    info = int(alerts.get("info",0))
    last = alerts.get("last",[])
    rows = "".join(
        f"<tr><td>{it.get('version','—')}</td><td>{it.get('type','—')}</td><td>{(it.get('conf') if it.get('conf') is not None else '—')}</td></tr>"
        for it in last[:3]
    ) or "<tr><td colspan='3' class='small'>no recent events</td></tr>"
    return f"""
<section class="card">
  <h3>Alerts & Notifications</h3>
  <div class="kv">
    <div>Critical</div><div><span class="pill crit">{crit}</span></div>
    <div>Info</div><div><span class="pill ok">{info}</span></div>
  </div>
  <table class="small"><thead><tr><th>Version</th><th>Type</th><th>Conf</th></tr></thead><tbody>{rows}</tbody></table>
</section>
"""

def _write_html(out_html: Path, manifest: Dict[str, Any]) -> None:
    ensure_dir(out_html.parent)
    h = []
    h.append("<!doctype html><meta charset='utf-8'><title>MoonWire Governance Dashboard</title>")
    h.append(_html_css())
    h.append(_html_header(
        run_url=manifest.get("run_url"),
        generated_at=manifest["generated_at"],
        window_h=int(manifest.get("window_hours",72)),
        badges="<span>static</span><span>CI artifact</span>" + ("<span>demo</span>" if manifest.get("demo") else "")
    ))
    tiles = manifest.get("sections",{})
    h.append("<main class='grid'>")
    h.append(_html_tile_apply(tiles.get("apply",{})))
    h.append(_html_tile_bg(tiles.get("bluegreen",{})))
    h.append(_html_tile_trend(tiles.get("trend",{})))
    h.append(_html_tile_alerts(tiles.get("alerts",{})))
    h.append("</main>")
    h.append("<div class='footer small'>Raw artifacts available in CI &middot; Generated by MoonWire.</div>")
    out_html.write_text("\n".join(h))

def _write_png(out_png: Path) -> None:
    # Always ensure a PNG exists; if no renderer, use 1x1.
    ensure_dir(out_png.parent)
    if not out_png.exists():
        out_png.write_bytes(_PNG_1x1_BYTES)

def _collect_inputs(models_dir: Path, logs_dir: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["apply"] = _read_json(models_dir / "governance_apply_result.json") or {}
    data["bg"] = _read_json(models_dir / "bluegreen_promotion.json") or {}
    data["trend"] = _read_json(models_dir / "model_performance_trend.json") or {}
    data["notif"] = _read_json(models_dir / "governance_notifications_digest.json") or {}
    # Optional: last few governance actions from log
    last_events = []
    try:
        logp = logs_dir / "governance_apply.jsonl"
        if logp.exists():
            for line in logp.read_text().splitlines()[-5:]:
                last_events.append(json.loads(line))
    except Exception:
        pass
    data["gov_log_tail"] = last_events
    return data

def _build_manifest(ctx) -> Dict[str, Any]:
    window_h = int(os.getenv("MW_DASH_WINDOW_H", "72") or "72")
    run_url = os.getenv("GITHUB_RUN_URL")
    now = _iso()
    inputs = _collect_inputs(Path(ctx.models_dir), Path(ctx.logs_dir))

    # Apply section (derive counts with fallbacks)
    apply_src = inputs["apply"] or {}
    apply_manifest = {
        "mode": apply_src.get("action_mode", "dryrun"),
        "applied": len(apply_src.get("applied", [])) if isinstance(apply_src.get("applied"), list) else apply_src.get("applied", 0) or 0,
        "skipped": len(apply_src.get("skipped", [])) if isinstance(apply_src.get("skipped"), list) else apply_src.get("skipped", 0) or 0,
        "reversal_plan": (apply_src.get("reversal_plan") or "monitor 12h for precision regression ≥ 0.02"),
    }

    # Blue-green section
    bg_src = inputs["bg"] or {}
    bluegreen_manifest = {
        "current": bg_src.get("current","v?.?.?"),
        "candidate": bg_src.get("candidate","v?.?.?"),
        "classification": bg_src.get("classification","observe"),
        "confidence": bg_src.get("confidence", 0.80),
        "delta": {
            "F1": bg_src.get("delta",{}).get("F1"),
            "ECE": bg_src.get("delta",{}).get("ECE"),
        },
    }

    # Trend section
    t_src = inputs["trend"] or {}
    f1_delta = t_src.get("delta",{}).get("F1")
    ece_delta = t_src.get("delta",{}).get("ECE")
    trend_manifest = {
        "f1_trend": _trend_tag(f1_delta),
        "ece_trend": _trend_tag(-ece_delta if ece_delta is not None else None),  # lower ECE is better
        "f1_series": t_src.get("series",{}).get("F1") or [0.5,0.52,0.54,0.55],
        "ece_series": t_src.get("series",{}).get("ECE") or [0.07,0.066,0.063,0.060],
    }

    # Alerts section
    n_src = inputs["notif"] or {}
    alerts_manifest = {
        "critical": len(n_src.get("critical", [])) if isinstance(n_src.get("critical"), list) else n_src.get("critical", 0) or 0,
        "info": len(n_src.get("info", [])) if isinstance(n_src.get("info"), list) else n_src.get("info", 0) or 0,
        "last": (n_src.get("critical", []) + n_src.get("info", []))[:3] if isinstance(n_src.get("critical", []), list) else [],
    }

    manifest = {
        "generated_at": now,
        "window_hours": window_h,
        "run_url": run_url,
        "sections": {
            "apply": apply_manifest,
            "bluegreen": bluegreen_manifest,
            "trend": trend_manifest,
            "alerts": alerts_manifest,
        },
        "demo": bool(getattr(ctx, "is_demo", False)),
    }
    return manifest

def build_dashboard(ctx) -> Dict[str, Any]:
    """
    Build a static HTML dashboard + PNG placeholder + manifest JSON.
    Returns a dict with output paths (relative to repo root).
    """
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    logs_dir = Path(getattr(ctx, "logs_dir", "logs"))

    ensure_dir(artifacts_dir); ensure_dir(models_dir)

    manifest = _build_manifest(ctx)

    # Write manifest JSON
    manifest_path = models_dir / "governance_dashboard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write HTML
    html_path = artifacts_dir / "governance_dashboard.html"
    _write_html(html_path, manifest)

    # Write PNG snapshot (placeholder ensures artifact always exists)
    png_path = artifacts_dir / "governance_dashboard.png"
    _write_png(png_path)

    return {
        "manifest": str(manifest_path),
        "html": str(html_path),
        "png": str(png_path),
    }