# scripts/ci/verify_social_features.py
from __future__ import annotations

import os, json
from pathlib import Path

def main() -> None:
    # 1) Social must be enabled
    if str(os.getenv("MW_SOCIAL_ENABLED", "0")).lower() not in {"1", "true", "yes"}:
        raise SystemExit("MW_SOCIAL_ENABLED is off")

    # 2) Reddit JSONL must exist & be non-empty
    reddit_path = Path("logs/social_reddit.jsonl")
    if not reddit_path.exists() or reddit_path.stat().st_size == 0:
        raise SystemExit("social_reddit.jsonl missing or empty")

    # 3) Build social series using project code
    try:
        from scripts.ml.social_features import compute_social_series
    except Exception as e:
        raise SystemExit(f"import social_features failed: {e}")

    df = compute_social_series(Path("."))
    if df.empty or "social_score" not in df.columns:
        raise SystemExit("social_score series is missing/empty")

    s = df["social_score"].dropna()
    non_neutral = int((s != 0.5).sum())
    ratio = float(non_neutral) / max(1, len(s))
    start = str(s.index.min()) if len(s) else "n/a"
    end = str(s.index.max()) if len(s) else "n/a"

    # 4) Optional: check model manifest for feature names
    features_note = ""
    manifest = Path("models/ml_model_manifest.json")
    if manifest.exists():
        try:
            jm = json.loads(manifest.read_text())
            feats = jm.get("features") or []
            has_social = any(("social" in str(x).lower()) for x in feats)
            features_note = f" | model_has_social_features={has_social}"
        except Exception:
            pass

    # 5) Write to CI summary
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "summary.md")
    with open(summary_path, "a", encoding="utf-8") as out:
        out.write("### Social feature verification\n")
        out.write(f"- social_reddit.jsonl size: {reddit_path.stat().st_size} bytes\n")
        out.write(f"- social_score rows: {len(s)}\n")
        out.write(f"- non-neutral share: {ratio:.2%}\n")
        out.write(f"- time span: {start} → {end}{features_note}\n")

if __name__ == "__main__":
    main()