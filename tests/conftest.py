# tests/conftest.py

import os
import json
import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import importlib

from main import app

@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """
    Overrides LOGS_DIR to a temp dir for clean test state.
    Reloads src.paths to apply the override.
    """
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    import src.paths
    importlib.reload(src.paths)

    # Ensure empty JSONL files exist
    (logs_dir / "retraining_log.jsonl").write_text("")
    (logs_dir / "reviewer_scores.jsonl").write_text("")
    (logs_dir / "retraining_triggered.jsonl").write_text("")
    # NEW: history file for trends
    (logs_dir / "reviewer_scores_history.jsonl").write_text("")

    yield  # test runs here

@pytest.fixture
def client():
    return TestClient(app)

# ---------- HELPERS ----------

from src.paths import (
    RETRAINING_LOG_PATH,
    REVIEWER_SCORES_PATH,
    REVIEWER_SCORES_HISTORY_PATH,
)

def append_jsonl(path: Path, obj: dict):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")

@pytest.fixture
def write_flag():
    def _write(signal_id: str, reviewer_id: str, weight: float = None):
        entry = {
            "signal_id": signal_id,
            "reviewer_id": reviewer_id,
            "timestamp": time.time(),
        }
        if weight is not None:
            entry["reviewer_weight"] = weight
        append_jsonl(RETRAINING_LOG_PATH, entry)
    return _write

@pytest.fixture
def write_score():
    def _write(reviewer_id: str, score: float):
        entry = {"reviewer_id": reviewer_id, "score": score, "timestamp": time.time()}
        append_jsonl(REVIEWER_SCORES_PATH, entry)
    return _write

@pytest.fixture
def write_score_history():
    def _write(reviewer_id: str, score: float, ts: float = None):
        entry = {
            "reviewer_id": reviewer_id,
            "score": score,
            "timestamp": ts if ts is not None else time.time(),
        }
        append_jsonl(REVIEWER_SCORES_HISTORY_PATH, entry)
    return _write