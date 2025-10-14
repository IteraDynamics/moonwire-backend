# scripts/governance/governance_alerts.py
from __future__ import annotations

import json
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.summary_sections.common import ensure_dir, _iso


@dataclass
class _Env:
    mode: str = os.getenv("MW_ALERT_MODE", "print").strip().lower() or "print"
    slack_webhook: Optional[str] = os.getenv("MW_ALERT_SLACK_WEBHOOK") or None
    email_to: Optional[str] = os.getenv("MW_ALERT_EMAIL_TO") or None
    min_conf: float = float(os.getenv("MW_ALERT_MIN_CONFIDENCE", "0.85"))
    trigger_delta: float = float(os.getenv("MW_ALERT_TRIGGER_DELTA", "-0.02"))  # negative = regression
    include_sim: bool = (os.getenv("MW_ALERT_INCLUDE_SIM", "true").strip().lower() == "true")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text() or "{}")
    except Exception:
        pass
    return {}


def _read_jsonl_last(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        last = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except Exception:
                    continue
        return last
    except Exception:
        return None


def _performance_regression(trend: Dict[str, Any], trigger_delta: float) -> Optional[Tuple[str, float]]:
    """
    Return (version, delta_f1) if a regression ≥ |trigger_delta| (negative delta) is detected.
    Assumes trend["versions"] list with fields: version, F1 or f1 or f1_delta, precision, ece, etc.
    """
    try:
        versions = trend.get("versions", [])
        for v in versions:
            # Try to find a delta or compute approximate delta from optional fields.
            df1 = (
                v.get("F1_delta")
                or v.get("f1_delta")
                or v.get("delta_F1")
                or v.get("delta_f1")
            )
            if df1 is None:
                # fallback: if both present, create a tiny synthetic "delta" of +0.0
                # (only alerts on explicit negatives)
                continue
            df1 = float(df1)
            # trigger when F1 change is ≤ trigger_delta (e.g., -0.02 or worse)
            if df1 <= float(trigger_delta):
                return str(v.get("version", "?")), df1
    except Exception:
        pass
    return None


def _compose_alert_lines(alerts: List[Dict[str, Any]], mode: str) -> List[str]:
    lines = ["", "📣 Governance Alerts (72 h)"]
    if not alerts:
        lines.append("• (no alerts)")
    else:
        for a in alerts:
            v = a.get("version", "?")
            cls = a.get("classification", "?")
            conf = a.get("confidence")
            d_f1 = a.get("delta", {}).get("F1")
            d_ece = a.get("delta", {}).get("ECE")
            # Normalize signs for display
            def fmt(x: Optional[float], plus: bool = True) -> str:
                if x is None:
                    return "0.00"
                return f"{x:+.02f}" if plus else f"{x:.02f}"

            conf_s = f"{float(conf):.2f}" if conf is not None else "—"
            d_f1_s = fmt(d_f1, plus=True).replace("+", "＋").replace("-", "−")
            d_ece_s = fmt(d_ece, plus=True).replace("+", "＋").replace("-", "−")
            lines.append(f"• {v} → {cls} (ΔF1 {d_f1_s}, ΔECE {d_ece_s}, conf {conf_s})")
    lines.append(f"• Mode: {mode}")
    return lines


def _log_line(logs_dir: Path, text: str) -> None:
    ensure_dir(logs_dir)
    path = logs_dir / "governance_alerts.log"
    ts = _iso(_now_utc())
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")
    except Exception:
        # best effort, but don't fail build
        pass


def _send_slack(webhook: str, payload: Dict[str, Any]) -> bool:
    """
    Best-effort Slack webhook POST. If networking is disabled, this simply no-ops.
    We avoid adding requests dependency; use urllib.
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            _ = resp.read()
        return True
    except Exception:
        return False


def _send_email(to_addr: str, subject: str, body: str) -> bool:
    """
    Light SMTP send if environment is configured. If not, no-op.
    Env (optional): SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_STARTTLS=true/false
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "0") or 0)
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    starttls = os.getenv("SMTP_STARTTLS", "true").strip().lower() == "true"
    if not (host and port and user and pwd and to_addr):
        return False
    try:
        msg = EmailMessage()
        msg["From"] = user
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)

        if starttls:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=5) as s:
                s.starttls(context=context)
                s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=5) as s:
                s.login(user, pwd)
                s.send_message(msg)
        return True
    except Exception:
        return False


def _collect_inputs(models_dir: Path, logs_dir: Path) -> Dict[str, Any]:
    plan_apply = _read_json(models_dir / "governance_apply_result.json")
    bluegreen = _read_json(models_dir / "bluegreen_promotion.json")
    trend = _read_json(models_dir / "model_performance_trend.json")
    ledger_last = _read_jsonl_last(logs_dir / "governance_apply.jsonl") or {}
    return {
        "apply": plan_apply,
        "bluegreen": bluegreen,
        "trend": trend,
        "ledger_last": ledger_last,
    }


def _demo_seed(alerts: List[Dict[str, Any]]) -> None:
    """
    Ensure at least one promote_ready and one rollback_risk appear in demo mode.
    """
    have_promote = any(a.get("classification") == "promote_ready" for a in alerts)
    have_rollback = any(a.get("classification") == "rollback_risk" for a in alerts)
    if not have_promote:
        alerts.append({
            "version": "v0.7.7",
            "classification": "promote_ready",
            "confidence": 0.88,
            "delta": {"F1": +0.02, "ECE": -0.01},
            "notes": ["demo_seed"]
        })
    if not have_rollback:
        alerts.append({
            "version": "v0.7.9",
            "classification": "rollback_risk",
            "confidence": 0.74,
            "delta": {"F1": -0.03, "ECE": +0.01},
            "notes": ["demo_seed"]
        })


def run_alerts(ctx) -> List[str]:
    """
    Public API: invoked from mw_demo_summary after Blue-Green simulation.
    Returns CI markdown lines to append to the summary.
    Always best-effort; never raises.
    """
    try:
        models = Path(ctx.models_dir or "models")
        logs = Path(ctx.logs_dir or "logs")
        ensure_dir(models); ensure_dir(logs)

        env = _Env()
        inputs = _collect_inputs(models, logs)

        alerts: List[Dict[str, Any]] = []
        ts = _iso(_now_utc())

        # 1) From Governance Apply: low-confidence applies
        apply = inputs.get("apply", {})
        for a in apply.get("applied", []):
            conf = float(a.get("confidence", 0.0) or 0.0)
            if conf < env.min_conf:
                alerts.append({
                    "version": a.get("version", "?"),
                    "classification": a.get("action", "observe"),
                    "confidence": conf,
                    "delta": {},
                    "notes": ["low_confidence_apply"]
                })

        # 2) From Blue-Green Simulation
        if env.include_sim:
            bg = inputs.get("bluegreen", {})
            if bg:
                cls = str(bg.get("classification", "observe"))
                conf = float(bg.get("confidence", 0.0) or 0.0)
                delta = bg.get("delta", {})
                v = str(bg.get("candidate") or bg.get("current_model") or "?")
                # Alert on rollback_risk OR low-confidence promote_ready
                if cls == "rollback_risk" or (cls == "promote_ready" and conf < env.min_conf):
                    alerts.append({
                        "version": v,
                        "classification": cls,
                        "confidence": conf,
                        "delta": {"F1": delta.get("F1"), "ECE": delta.get("ECE")},
                        "notes": ["bluegreen_classification"]
                    })

        # 3) Performance trend regression
        trend = inputs.get("trend", {})
        reg = _performance_regression(trend, env.trigger_delta)
        if reg:
            v, df1 = reg
            alerts.append({
                "version": v,
                "classification": "observe",
                "confidence": None,
                "delta": {"F1": df1, "ECE": None},
                "notes": ["f1_regression"]
            })

        # Demo guarantee
        demo = bool(getattr(ctx, "is_demo", False)) or (os.getenv("DEMO_MODE", "false").lower() == "true")
        if demo:
            _demo_seed(alerts)

        # Compose and dispatch
        ci_lines = _compose_alert_lines(alerts, env.mode)

        # Emit log line(s) and send
        for a in alerts:
            v = a.get("version", "?")
            cls = a.get("classification", "?")
            conf = a.get("confidence")
            df1 = a.get("delta", {}).get("F1")
            conf_s = f"{float(conf):.2f}" if conf is not None else "—"
            df1_s = f"{float(df1):+.02f}" if df1 is not None else "0.00"
            _log_line(logs, f"Governance Alert: {v} {cls} ΔF1 {df1_s} (conf={conf_s})")

        # Dispatch external notification (best-effort, never fail)
        if alerts:
            # Minimal text summary
            top = alerts[0]
            txt = f":rotating_light: Governance Alert: {top.get('version','?')} classified as {top.get('classification','?')}"
            d = top.get("delta", {})
            if "F1" in d:
                txt += f" (ΔF1 {float(d['F1']):+.02f})"
            if top.get("confidence") is not None:
                txt += f", conf {float(top['confidence']):.2f}"
            payload = {
                "text": txt,
                "attachments": [
                    {"text": "Reversal plan active if applicable. View CI summary for full trace."}
                ]
            }

            if env.mode == "slack" and env.slack_webhook:
                _send_slack(env.slack_webhook, payload)
            elif env.mode == "email" and env.email_to:
                _send_email(env.email_to, "MoonWire Governance Alert", txt)
            else:
                # print mode → write a console-looking line into the log as well (already done),
                # and no-op for actual console in CI context.
                pass

        # Write a small JSON status for audit (optional, not required by prompt)
        status = {
            "generated_at": ts,
            "mode": env.mode,
            "alerts": alerts,
        }
        (logs / "governance_alerts.status.json").write_text(json.dumps(status, indent=2))

        return ci_lines
    except Exception as e:
        # Never block CI summary; return a failure line
        return [f"", f"❌ Governance Alerts failed: {e}"]
