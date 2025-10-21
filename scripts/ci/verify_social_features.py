# scripts/ci/verify_social_features.py
from __future__ import annotations
import os, json
from pathlib import Path

from scripts.ml.social_features import compute_social_series

def main():
    # 1) Gate is on?
    gate = str(os.getenv("MW_SOCIAL_ENABLED","0")).strip().lower() in {"1","true","yes"}
    if not gate:
        raise SystemExit("MW_SOCIAL_ENABLED is off")

    # 2) Reddit jsonl exists and is non-empty
    reddit_path = Path("logs/social_reddit.jsonl")
    assert reddit_path.exists() and reddit_path.stat().st_size > 0, "social_reddit.jsonl missing or empty"

    # 3) Build series
    df = compute_social_series(Path("."))
    assert (not df.empty) and ("social_score" in df.columns), "social_score series is missing/empty"
    s = df["social_score"].dropna()
    non_neutral = int((s != 0.5).sum())
    ratio = float(non_neutral) / max(1, len(s))
    start = str(s.index.min()) if len(s) else "n/a"
    end = str(s.index.max()) if len(s) else "n/a"

    # 4) Model manifest check: accept "features" OR "feature_list"
    manifest = Path("models/ml_model_manifest.json")
    has_social = False
    if manifest.exists():
        try:
            jm = json.loads(manifest.read_text())
            feats = jm.get("features") or jm.get("feature_list") or []
            has_social = any("social" in str(x).lower() for x in feats)
        except Exception:
            pass

    # 5) Emit CI summary
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY","summary.md")
    with open(summary_path, "a", encoding="utf-8") as out:
        out.write("### Social feature verification\n")
        out.write(f"- MW_SOCIAL_ENABLED env: {int(gate)}\n")
        out.write(f"- social_reddit.jsonl size: {reddit_path.stat().st_size} bytes\n")
        out.write(f"- social_score rows: {len(s)}\n")
        out.write(f"- non-neutral share: {ratio:.2%}\n")
        out.write(f"- time span: {start} → {end}\n")
        out.write(f"- model_has_social_features: {has_social}\n")

if __name__ == "__main__":
    main()