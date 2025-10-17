# src/signal_log.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

# --- Constants ---
_CANONICAL = Path("logs/signal_history.jsonl")
_LEGACY = Path("logs/signals.jsonl")
_ENV_PATH = "SIGNALS_FILE"

_REQUIRED_KEYS = [
    "ts",            # ISO8601 UTC string
    "symbol",        # e.g., "BTC"
    "direction",     # "long" | "short"
    "confidence",    # float
    "price",         # float
    "source",        # str
    "model_version", # str
]

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def make_signal_id(ts_iso: str, symbol: str, direction: str) -> str:
    """Deterministic ID format required by downstream readers."""
    sym = (symbol or "").strip().upper()
    d = (direction or "").strip().lower()
    ts = (ts_iso or "").strip()
    return f"sig_{ts}_{sym}_{d}"

def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a signal row in-place, returning the same dict."""
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")

    # Strip trivial whitespace where appropriate
    for k in ("ts", "symbol", "direction", "source", "model_version", "id"):
        if k in row and isinstance(row[k], str):
            row[k] = row[k].strip()

    # Required keys/type checks
    missing = [k for k in _REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")

    # Normalize symbol/direction
    row["symbol"] = str(row["symbol"]).upper()
    row["direction"] = str(row["direction"]).lower()

    # Basic type coercion/validation
    try:
        row["confidence"] = float(row["confidence"])
    except Exception as e:
        raise ValueError(f"confidence must be float-like: {e}")

    try:
        row["price"] = float(row["price"])
    except Exception as e:
        raise ValueError(f"price must be float-like: {e}")

    if not isinstance(row["ts"], str) or not row["ts"]:
        raise ValueError("ts must be a non-empty ISO8601 string")

    if not isinstance(row["source"], str) or not row["source"]:
        raise ValueError("source must be a non-empty string")

    if not isinstance(row["model_version"], str) or not row["model_version"]:
        raise ValueError("model_version must be a non-empty string")

    # Optional keys: id, outcome
    if not row.get("id"):
        row["id"] = make_signal_id(row["ts"], row["symbol"], row["direction"])

    # Ensure keys exactly match schema + allow 'id' and 'outcome'
    # (Do not drop extra keys silently; keep them to preserve forward-compat.)
    return row

def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file with durability (fsync)."""
    _ensure_parent(path)
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    # Use text mode with explicit encoding; fsync for durability
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())

def write_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public API (v0.9.0): validate/normalize and write the signal.
    Dual-write policy:
      - If SIGNALS_FILE is set -> write ONLY to that path.
      - Else -> write to both logs/signal_history.jsonl (canonical)
               and logs/signals.jsonl (legacy).
    Returns the normalized row.
    """
    row = _normalize_row(dict(row))  # copy defensively

    override = os.getenv(_ENV_PATH)
    if override:
        _append_jsonl(Path(override), row)
    else:
        _append_jsonl(_CANONICAL, row)
        _append_jsonl(_LEGACY, row)

    return row

# --- Backwards compatibility shim ---
def log_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Back-compat alias used by older routers/generators.
    Delegates to write_signal().
    """
    return write_signal(row)

__all__ = [
    "write_signal",
    "make_signal_id",
    "log_signal",  # back-compat export
]