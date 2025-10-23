# scripts/ci/dump_env.py
from __future__ import annotations
import os, json, sys, pathlib

"""
Dump the current MW_* environment (and a few extras) plus manifest fields (if present)
to a JSON file you specify as argv[1].

Optionally control which keys to include and order via:
  MW_ENV_SUMMARY_KEYS="KEY1,KEY2,..."
If not set, we include every env that starts with MW_ (sorted).
"""

EXTRA_MANIFEST_PATH = "models/ml_model_manifest.json"

def main(out_path: str):
    # 1) collect env
    env = {k: v for k, v in os.environ.items() if k.startswith("MW_")}
    # allow optional explicit ordering/whitelist
    keys_raw = os.getenv("MW_ENV_SUMMARY_KEYS", "").strip()
    if keys_raw:
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        env = {k: env.get(k, "") for k in keys}
    else:
        env = dict(sorted(env.items(), key=lambda kv: kv[0]))

    # 2) add a few non-MW flags that are useful
    extras = {
        "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED", ""),
        "PYTHONUNBUFFERED": os.getenv("PYTHONUNBUFFERED", ""),
    }

    # 3) read manifest if present
    manifest = {}
    p = pathlib.Path(EXTRA_MANIFEST_PATH)
    if p.exists():
        try:
            manifest = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    payload = {
        "env": env,
        "extras": extras,
        "manifest": {
            "model_type": manifest.get("model_type"),
            "social_enabled": manifest.get("social_enabled"),
            "social_include": manifest.get("social_include"),
            "features": manifest.get("features", []),
            "feature_list": manifest.get("feature_list", []),
            "symbols": manifest.get("symbols"),
            "lookback_days": manifest.get("lookback_days"),
            "horizon_h": manifest.get("horizon_h"),
            "train_days": manifest.get("train_days"),
            "test_days": manifest.get("test_days"),
        },
    }

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "models/run_env_dump.json"
    main(out)
