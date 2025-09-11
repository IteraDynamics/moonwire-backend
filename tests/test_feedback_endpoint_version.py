# tests/test_feedback_endpoint_version.py
import json, sys, tempfile
from pathlib import Path
from fastapi.testclient import TestClient

# import router + app
sys.path.append("src")
import trigger_likelihood_router as tlr  # type: ignore
from main import app  # assumes main.py creates FastAPI app and mounts router

client = TestClient(app)

def test_feedback_endpoint_writes_model_version(monkeypatch):
    with tempfile.TemporaryDirectory() as dtmp:
        tmpdir = Path(dtmp)
        trig = tmpdir / "trigger_history.jsonl"
        fb   = tmpdir / "label_feedback.jsonl"

        # patch router globals to use temp files
        tlr._TRIGGER_HISTORY_PATH = trig  # type: ignore
        tlr._LABEL_FEEDBACK_PATH  = fb    # type: ignore

        # seed a trigger row
        with trig.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": "2025-09-11T12:38:00Z",
                "origin": "reddit",
                "adjusted_score": 0.73,
                "decision": True,
                "model_version": "v0.5.1"
            }) + "\n")

        # send feedback within ±5m window
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

        # check file contents
        lines = fb.read_text(encoding="utf-8").splitlines()
        assert lines, "feedback file should have a row"
        row = json.loads(lines[0])
        assert row["model_version"] == "v0.5.1"