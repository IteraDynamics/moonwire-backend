# tests/test_signal_quality_summary.py
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, importlib, os

from scripts.summary_sections.common import SummaryContext

def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def test_signal_quality_classification_and_persist(tmp_path, monkeypatch):
    # Isolate models dir for the section
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("MW_SIGNAL_BATCH_H", "3")
    monkeypatch.setenv("MW_SIGNAL_WINDOW_H", "24")

    # Build small trigger & label logs across 2 buckets
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Triggers (decision=triggered)
    trig_rows = [
        {"timestamp": _iso(now - timedelta(hours=1, minutes=0)), "origin": "twitter", "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=5)), "origin": "twitter", "decision": "triggered"},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=0)), "origin": "reddit",  "decision": "triggered"},
    ]
    (models_dir / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig_rows), encoding="utf-8")

    # Labels: 2 labels near the recent 2 triggers (1 True, 1 False) → Mixed; 1 True for older bucket → Strong
    lab_rows = [
        {"timestamp": _iso(now - timedelta(hours=1, minutes=2)), "origin": "twitter", "label": True},
        {"timestamp": _iso(now - timedelta(hours=1, minutes=7)), "origin": "twitter", "label": False},
        {"timestamp": _iso(now - timedelta(hours=4, minutes=3)), "origin": "reddit",  "label": True},
    ]
    (models_dir / "label_feedback.jsonl").write_text("\n".join(json.dumps(r) for r in lab_rows), encoding="utf-8")

    # Context + call
    from scripts.summary_sections import signal_quality
    importlib.reload(signal_quality)

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models_dir,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)

    md = []
    signal_quality.append(md, ctx)

    out_md = "\n".join(md)
    # Expect header
    assert "Signal Quality Summary" in out_md
    # One mixed bucket (twitter recent 2 triggers → 1/2 = 0.5) and one strong bucket (reddit 1/1 = 1.0)
    assert "Mixed" in out_md
    assert "Strong" in out_md

    # Persisted JSON exists
    jq = models_dir / "signal_quality_summary.json"
    assert jq.exists(), "signal_quality_summary.json should be written"
    payload = json.loads(jq.read_text(encoding="utf-8"))
    assert "batches" in payload and isinstance(payload["batches"], list)