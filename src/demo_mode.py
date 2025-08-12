# src/demo_mode.py
import os

def is_demo_mode() -> bool:
    """Return True when DEMO_MODE is enabled via env var."""
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
