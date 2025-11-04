import joblib
import numpy as np

# Test Standard tier
print("Testing Standard Tier Models:")
btc_std = joblib.load("models/standard/BTC_model.joblib")
eth_std = joblib.load("models/standard/ETH_model.joblib")

# Dummy features (8 features as shown in training)
dummy_X = np.array([[0.01, 0.02, -0.01, 0.015, 1500, 0.05, 1, 0.5]])

btc_pred = btc_std.predict_proba(dummy_X)[0, 1]
eth_pred = eth_std.predict_proba(dummy_X)[0, 1]

print(f"  BTC prediction: {btc_pred:.4f}")
print(f"  ETH prediction: {eth_pred:.4f}")

# Test Elite tier
print("\nTesting Elite Tier Models:")
btc_elite = joblib.load("models/elite/BTC_model.joblib")
eth_elite = joblib.load("models/elite/ETH_model.joblib")

btc_pred_elite = btc_elite.predict_proba(dummy_X)[0, 1]
eth_pred_elite = eth_elite.predict_proba(dummy_X)[0, 1]

print(f"  BTC prediction: {btc_pred_elite:.4f}")
print(f"  ETH prediction: {eth_pred_elite:.4f}")

print("\n✓ All models loaded and working!")