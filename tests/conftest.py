# tests/conftest.py

import os
import json
import time
import pytest
import importlib
from pathlib import Path
from fastapi.testclient import TestClient

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

    # Reload src.paths to pick up the new LOGS_DIR env var
    import src.paths
    importlib.reload(src.paths)

    # Ensure empty JSONL files exist
    (logs_dir / "retraining_log.jsonl").write_text("")
    (logs_dir / "reviewer_scores.jsonl").write_text("")
    (logs_dir / "retraining_triggered.jsonl").write_text("")

    yield  # test runs here


@pytest.fixture
def client():
    return TestClient(app)


# ---------- HELPER WRITERS ----------

def append_jsonl(path: Path, obj: dict):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


@pytest.fixture
def write_flag():
    def _write(signal_id: str, reviewer_id: str, weight: float = None):
        # ✅ Import here so monkeypatching takes effect
        from src.paths import RETRAINING_LOG_PATH
        entry = {
            "signal_id": signal_id,
            "reviewer_id": reviewer_id,
            "timestamp": time.time(),
        }
        if weight is not None:
            entry["reviewer_weight"] = weight
        append_jsonl(Path(RETRAINING_LOG_PATH), entry)
    return _write


@pytest.fixture
def write_score():
    def _write(reviewer_id: str, score: float):
        # ✅ Import here so monkeypatching takes effect
        from src.paths import REVIEWER_SCORES_PATH
        entry = {"reviewer_id": reviewer_id, "score": score}
        append_jsonl(Path(REVIEWER_SCORES_PATH), entry)
    return _write