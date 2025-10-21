# scripts/ml/utils.py
from __future__ import annotations
import os, json, time, pathlib
from typing import Any, Dict

ROOT = pathlib.Path(".").resolve()

def ensure_dirs():
    for p in ("data", "models", "logs", "artifacts"):
        (ROOT / p).mkdir(parents=True, exist_ok=True)

def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

def save_json(path: str | os.PathLike, obj: Dict[str, Any]):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def append_jsonl(path: str | os.PathLike, obj: Dict[str, Any]):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def utc_now_ts() -> int:
    return int(time.time())

def to_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [y.strip() for y in x.split(",") if y.strip()]
    return list(x)