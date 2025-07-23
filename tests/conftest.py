# tests/conftest.py

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.paths as paths
from src.main import app

# Determine whether to run in local (in-memory) mode or live (Render) mode
RUN_LOCAL = os.getenv("RUN_LOCAL", "true").lower() == "true"

@pytest.fixture(scope="session")
def run_local():
    """True for local TestClient tests, False for live HTTP tests."""
    return RUN_LOCAL

@pytest.fixture(scope="session", autouse=True)
def setup_logs_dir(run_local, monkeypatch):
    """
    When running locally, redirect all log paths to a temporary directory
    so we don’t pollute the real logs.
    """
    if run_local:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mw_test_logs_"))
        # Monkey-patch the paths in src.paths
        monkeypatch.setattr(paths, "LOGS_DIR", tmp_dir)
        monkeypatch.setattr(paths, "REVIEWER_IMPACT_LOG_PATH", tmp_dir / "reviewer_impact_log.jsonl")
        monkeypatch.setattr(paths, "REVIEWER_SCORES_PATH",     tmp_dir / "reviewer_scores.jsonl")
        monkeypatch.setattr(paths, "OVERRIDE_LOG_PATH",        tmp_dir / "override_log.jsonl")

@pytest.fixture(scope="session")
def client(run_local):
    """
    Provides FastAPI TestClient when running locally.
    Skipped in live mode.
    """
    if run_local:
        return TestClient(app)
    pytest.skip("Skipping TestClient in live mode")

@pytest.fixture(scope="session")
def http(run_local):
    """
    Provides the requests module when running live against BASE_URL.
    Skipped in local mode.
    """
    if not run_local:
        import requests
        return requests
    pytest.skip("Skipping live HTTP fixture in local mode")

@pytest.fixture(scope="session")
def base_url(run_local):
    """
    Provides BASE_URL for live tests (e.g. https://moonwire-signal-engine-1.onrender.com).
    Skipped in local mode.
    """
    if not run_local:
        return os.getenv("BASE_URL")
    pytest.skip("Skipping BASE_URL in local mode")
