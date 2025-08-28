import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.paths import LOGS_DIR, MODELS_DIR
from src.analytics.origin_utils import normalize_origin as _norm
from src.ml.infer import score as infer_score

RETRAINING_LOG_PATH = LOGS_DIR / "retraining_log.jsonl"
RETRAINING_TRIGGERED_LOG_PATH = LOGS_DIR / "retraining_triggered.jsonl"
THRESHOLDS_PATH = MODELS_DIR / "per_origin_thresholds.json"
DEFAULT_THRESHOLD = 2.5

def _parse_ts(v):
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        try:
            s = str(v); s = s[:-1] + "+00:00" if s.endswith("Z") else s
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None

def _load_jsonl(path: Path) -> List[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        return []

def compute_thresholds(min_total=30, min_positive=5, targets=[0.7, 0.8]) -> Dict[str, Dict[str, float]]:
    rows = _load_jsonl(RETRAINING_LOG_PATH)
    labels = _load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)
    label_lookup = defaultdict(list)

    for r in labels:
        o = _norm(r.get("origin", ""))
        ts = _parse_ts(r.get("timestamp"))
        if ts:
            label_lookup[o].append(ts)

    grouped: Dict[str, List[tuple]] = defaultdict(list)
    for row in rows:
        origin = _norm(row.get("origin", "unknown"))
        ts = _parse_ts(row.get("timestamp"))
        if not ts:
            continue
        p = infer_score(row).get("prob_trigger_next_6h", 0.0)
        label = int(any(t0 > ts and t0 <= ts + timedelta(hours=6) for t0 in label_lookup[origin]))
        grouped[origin].append((p, label))

    out: Dict[str, Dict[str, float]] = {}
    for origin, pairs in grouped.items():
        if len(pairs) < min_total or sum(y for _, y in pairs) < min_positive:
            out[origin] = {"p70": DEFAULT_THRESHOLD, "p80": DEFAULT_THRESHOLD}
            continue

        probs, ys = zip(*pairs)
        precisions, recalls, thresholds = precision_recall_curve(ys, probs)

        def find_thresh(target):
            for p, t in zip(precisions, thresholds):
                if p >= target:
                    return float(t)
            return float(DEFAULT_THRESHOLD)

        out[origin] = {
            "p70": find_thresh(0.70),
            "p80": find_thresh(0.80),
        }

    with THRESHOLDS_PATH.open("w") as f:
        json.dump(out, f, indent=2)

    return out