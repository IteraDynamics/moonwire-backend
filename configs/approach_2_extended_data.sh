#!/usr/bin/env bash
# Approach 2: Extended Training Data (365 days)
# Train on full year of data to capture more market cycles

set -e

echo "=================================================="
echo "ML Improvement Approach 2: Extended Training Data"
echo "=================================================="
echo ""
echo "Strategy: Train on 365 days instead of 180"
echo "- More market cycles (bull, bear, ranging)"
echo "- Better generalization across regimes"
echo "- Reduced overfitting to recent conditions"
echo ""
echo "Expected impact: More robust model across market conditions"
echo "Target: 52%+ WR, 1.15+ PF"
echo ""

# Set environment variables for this approach
export MW_ML_LOOKBACK_DAYS=365
export MW_ML_MODEL=hybrid
export MW_ML_SYMBOLS=BTC,ETH

# Disable regime features for baseline comparison
export MW_REGIME_ENABLED=0
export MW_REGIME_FILTER_ENABLED=0

echo "Configuration:"
echo "- Training data: 365 days (2x baseline)"
echo "- Model type: hybrid (LR + GB)"
echo "- Regime filtering: DISABLED"
echo "- Feature selection: ALL features"
echo ""

# Run training
python -m scripts.ml.train_predict

echo ""
echo "✓ Training complete!"
echo "Check artifacts/ml_* and models/ml_model_manifest.json for results"
