# tests/conftest.py

import shutil
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from main import app
from src.paths import LOGS_DIR  # reuse the same logs dir

# Ensure a clean logs directory for each test run
@pytest.fixture(autouse=True)
def cleanup_logs():
    shutil.rmtree(LOGS_DIR, ignore_errors=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    yield
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