# tests/conftest.py

import shutil
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from main import app
from src.paths import LOGS_DIR, REVIEWER_IMPACT_LOG_PATH, REVIEWER_SCORES_PATH

# Ensure a clean logs directory for each test run and re-create the two JSONL files
@pytest.fixture(autouse=True)
def cleanup_logs():
    # Remove and recreate logs dir
    shutil.rmtree(LOGS_DIR, ignore_errors=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Touch the two target files so they 'exist'
    REVIEWER_IMPACT_LOG_PATH.touch(exist_ok=True)
    REVIEWER_SCORES_PATH.touch(exist_ok=True)

    yield

    # Cleanup afterwards too
    shutil.rmtree(LOGS_DIR, ignore_errors=True)

# HTTP client for FastAPI
@pytest.fixture
def client():
    return TestClient(app)

def append_jsonl(path: Path, entry: dict):
    """
    Append a dict as a JSONL line to the given path, 
    creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")