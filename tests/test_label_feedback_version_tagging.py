# tests/test_feedback_endpoint_version.py
import json
from pathlib import Path
from fastapi.testclient import TestClient

from main import app  # main mounts the router
import src.trigger_likelihood_router as tlr  # <-- import the SAME module main uses

client = TestClient(app)

def test_feedback_endpoint_writes_model_version(tmp_path):
    trig: Path = tmp_path / "trigger_history.jsonl"
    fb: Path   = tmp_path / "label_feedback.jsonl"

    # Patch the router globals to temp files used by the mounted app
    tlr._TRIGGER_HISTORY_PATH = trig  # type: ignore[attr-defined]
    tlr._LABEL_FEEDBACK_PATH  = fb    # type: ignore[attr-defined]

    # Seed a trigger row within the ±5m window
    trig.write_text(json.dumps({
        "timestamp": "2025-09-11T12:38:00Z",
        "origin": "reddit",
        "adjusted_score": 0.73,
        "decision": True,
        "model_version": "v0.5.1"
    }) + "\n", encoding="utf-8")

    # Send feedback that should match the above trigger
    payload = {
        "timestamp": "2025-09-11T12:40:00Z",
        "origin": "reddit",
        "adjusted_score": 0.72,
        "label": True,
    }
    resp = client.post("/internal/trigger-likelihood/feedback", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v0.5.1"

    # Verify file write
    lines = fb.read_text(encoding="utf-8").splitlines()
    assert lines
    row = json.loads(lines[0])
    assert row["model_version"] == "v0.5.1"