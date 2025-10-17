from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


CANONICAL_PATH = Path("logs/signal_history.jsonl")
LEGACY_PATH = Path("logs/signals.jsonl")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_signal_id(ts_iso: str, symbol: str, direction: str) -> str:
    """
    Build a deterministic ID using the ISO timestamp, uppercased symbol, and lowercased direction.
    """
    ts_iso = (ts_iso or "").strip()
    symbol = (symbol or "").strip().upper()
    direction = (direction or "").strip().lower()
    return f"sig_{ts_iso}_{symbol}_{direction}"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """
    Append a single JSON object as a line to the given path.
    Ensures UTF-8 encoding, newline-terminated, and fsync for durability.
    """
    _ensure_parent(path)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    # newline translation is fine in text mode; force utf-8
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some environments (e.g., certain CI file systems) may not support fsync.
            pass


@dataclass
class _SchemaSpec:
    required: tuple = (
        "ts",
        "symbol",
        "direction",
        "confidence",
        "price",
        "source",
        "model_version",
        "outcome",
    )
    optional: tuple = ("id",)


def _validate_and_normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate required keys/types and normalize fields:
      - symbol -> UPPER
      - direction -> lower
      - id -> generated if missing
      - keys must be lowercase (by contract)
    """
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")

    # Trim string-like inputs
    def _s(v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    row = {str(k): row[k] for k in row}  # ensure string keys
    # Ensure lower-case keys only (do not rename; require contract)
    for k in list(row.keys()):
        if k != k.lower():
            # Preserve value but enforce lower-case key copy, then delete old key
            row[k.lower()] = row.pop(k)

    spec = _SchemaSpec()
    for key in spec.required:
        if key not in row:
            raise KeyError(f"missing required key: {key}")

    # Normalize fields
    row["symbol"] = _s(row["symbol"]).upper()
    row["direction"] = _s(row["direction"]).lower()
    row["ts"] = _s(row["ts"]) if row["ts"] is not None else _now_utc_iso()
    row["source"] = _s(row["source"])
    row["model_version"] = _s(row["model_version"])

    # Basic type checks (lightweight / permissive casting where sensible)
    if not isinstance(row["ts"], str):
        raise TypeError("ts must be ISO8601 string")
    if not isinstance(row["symbol"], str):
        raise TypeError("symbol must be string")
    if row["direction"] not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")

    try:
        row["confidence"] = float(row["confidence"])
    except Exception as e:
        raise TypeError("confidence must be float-like") from e

    try:
        row["price"] = float(row["price"])
    except Exception as e:
        raise TypeError("price must be float-like") from e

    # outcome can be null or any JSON value; do not coerce

    # ID generation if missing/empty
    if "id" not in row or not row["id"]:
        row["id"] = make_signal_id(row["ts"], row["symbol"], row["direction"])
    else:
        row["id"] = _s(row["id"])

    return row


def write_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write the signal to the appropriate JSONL file(s) and return the normalized row.
      - If SIGNALS_FILE is set -> write only to that file.
      - Else -> dual-write to:
            logs/signal_history.jsonl (canonical)
            logs/signals.jsonl       (legacy)
    """
    normalized = _validate_and_normalize(dict(row))  # copy defensively

    custom = os.getenv("SIGNALS_FILE")
    if custom:
        _append_jsonl(Path(custom), normalized)
        return normalized

    # Dual-write (canonical + legacy)
    _append_jsonl(CANONICAL_PATH, normalized)
    _append_jsonl(LEGACY_PATH, normalized)
    return normalized