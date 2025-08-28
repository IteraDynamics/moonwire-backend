# scripts/debug_ensemble_artifacts.py
from src.ml.infer import _paths_for, _load_one
import numpy as np

def main():
    print("\n== artifact + predict check ==")
    for kind in ("logistic","rf","gb"):
        mpath, jpath = _paths_for(kind)
        print(f"\n[{kind}]")
        print("  files:", mpath.exists(), jpath.exists(), str(mpath))
        model, meta = _load_one(kind)
        print("  loaded:", model is not None, "meta:", bool(meta))
        if not (model and meta):
            continue
        order = meta.get("feature_order") or []
        print("  n_features_in(meta):", len(order))
        x = np.zeros((1, len(order)), dtype=float)
        try:
            p = float(model.predict_proba(x)[0,1])
            print("  predict_proba OK:", p)
        except Exception as e:
            print("  predict_proba ERROR:", type(e).__name__, e)

if __name__ == "__main__":
    main()
