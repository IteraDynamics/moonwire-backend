import os
import pytest
import requests
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="session")
def client():
    """
    If RUN_LOCAL=true (default), use FastAPI TestClient.
    Otherwise, use requests against BASE_URL.
    """
    run_local = os.getenv("RUN_LOCAL", "true").lower() in ("1", "true", "yes")
    if run_local:
        return TestClient(app)
    base = os.getenv("BASE_URL")
    if not base:
        pytest.skip("BASE_URL not set for live testing")
    class LiveClient:
        def get(self, path, **kwargs):
            return requests.get(base + path, **kwargs)
        def post(self, path, **kwargs):
            return requests.post(base + path, **kwargs)
        def head(self, path, **kwargs):
            return requests.head(base + path, **kwargs)
    return LiveClient()
