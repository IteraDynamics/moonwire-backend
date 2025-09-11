*** a/src/trigger_likelihood_router.py
--- b/src/trigger_likelihood_router.py
@@
 from fastapi import APIRouter, Body, HTTPException, Query
 from src import paths
 import json
 import os
-from datetime import datetime, timezone
+from datetime import datetime, timezone, timedelta
 from pathlib import Path
 from src.paths import MODELS_DIR
 
 from src.ml.infer import (
     infer_score,
@@
 _LABEL_FEEDBACK_PATH: Path = MODELS_DIR / "label_feedback.jsonl"
+_TRIGGER_HISTORY_PATH: Path = MODELS_DIR / "trigger_history.jsonl"
 
 
 def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
     path.parent.mkdir(parents=True, exist_ok=True)
     with path.open("a", encoding="utf-8") as f:
         f.write(json.dumps(obj, ensure_ascii=False) + "\n")
 
+def _iter_jsonl(path: Path):
+    if not path.exists():
+        return
+    with path.open("r", encoding="utf-8") as f:
+        for line in f:
+            line = line.strip()
+            if not line:
+                continue
+            try:
+                yield json.loads(line)
+            except Exception:
+                # skip malformed/mid-write lines
+                continue
+
+def _parse_any_ts(ts: str) -> datetime:
+    """
+    Accepts:
+      - 'YYYY-MM-DDTHH:MM:SSZ'
+      - 'YYYY-MM-DDTHH:MM:SS.sssZ'
+      - ISO with offset: '...+00:00'
+    Returns tz-aware UTC datetime.
+    """
+    s = str(ts).strip()
+    if s.endswith("Z"):
+        # normalize fractional seconds if present
+        if "." in s:
+            s = s.split(".")[0] + "Z"
+        # convert to offset form for fromisoformat
+        s = s[:-1] + "+00:00"
+    dt = datetime.fromisoformat(s)
+    # ensure UTC
+    if dt.tzinfo is None:
+        dt = dt.replace(tzinfo=timezone.utc)
+    else:
+        dt = dt.astimezone(timezone.utc)
+    return dt
+
+def _find_model_version_for_label(*, label_timestamp: str, origin: str, window_minutes: int = 5) -> str:
+    """
+    Scan MODELS_DIR/trigger_history.jsonl for the closest row (by |Δt|) that
+    shares the same origin and is within ±window. Return its model_version or 'unknown'.
+    """
+    label_dt = _parse_any_ts(label_timestamp)
+    window = timedelta(minutes=window_minutes)
+    best_row = None
+    best_abs = None
+
+    for row in _iter_jsonl(_TRIGGER_HISTORY_PATH):
+        if row.get("origin") != origin:
+            continue
+        ts = row.get("timestamp")
+        if not ts:
+            continue
+        try:
+            trig_dt = _parse_any_ts(ts)
+        except Exception:
+            continue
+        delta = trig_dt - label_dt
+        if abs(delta) <= window:
+            ad = abs(delta)
+            if best_row is None or ad < best_abs:
+                best_row, best_abs = row, ad
+
+    mv = (best_row or {}).get("model_version")
+    return str(mv) if mv else "unknown"
 
 def _validate_feedback_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
     """
     Required:
       - timestamp (ISO-8601 or epoch seconds)
@@
     return out
 
 
 @router.post("/internal/trigger-likelihood/feedback")
 async def post_label_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
     """
     Accepts label feedback and appends it to models/label_feedback.jsonl.
     Schema enforced by _validate_feedback_payload.
     """
     try:
         record = _validate_feedback_payload(payload)
+        # --- v0.5.2: attach model_version from trigger history (fallback 'unknown')
+        record["model_version"] = _find_model_version_for_label(
+            label_timestamp=record["timestamp"],
+            origin=record["origin"],
+            window_minutes=5,
+        )
         _append_jsonl(_LABEL_FEEDBACK_PATH, record)
-        return {"status": "ok", "written": True}
+        return {"status": "ok", "written": True, "model_version": record["model_version"]}
     except ValueError as ve:
         # 400-like error (FastAPI will still wrap as 200 unless you raise HTTPException;
         # keeping minimal to avoid changing imports)
         return {"status": "error", "written": False, "error": str(ve)}
     except Exception as e:
         return {"status": "error", "written": False, "error": f"internal: {type(e).__name__}: {e}"}