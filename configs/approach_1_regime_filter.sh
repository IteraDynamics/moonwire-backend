#!/usr/bin/env bash
# Approach 1: Regime Detection + Filtering
# Only train on trending markets to improve model edge

set -e

echo "=================================================="
echo "ML Improvement Approach 1: Regime Detection + Filtering"
echo "=================================================="
echo ""
echo "Strategy: Train model only on trending markets"
echo "- Regime detection identifies trending vs choppy markets"
echo "- Model trained only on trending periods (where it has edge)"
echo "- Predicts on all periods but learned from clean data"
echo ""
echo "Expected impact: Reduce overfitting to choppy markets"
echo "Target: 52%+ WR, 1.15+ PF"
echo ""

# Set environment variables for this approach
export MW_ML_LOOKBACK_DAYS=180
export MW_ML_MODEL=hybrid
export MW_REGIME_ENABLED=1
export MW_REGIME_FILTER_ENABLED=1
export MW_REGIME_VOLATILITY_THRESHOLD=0.06
export MW_REGIME_TREND_THRESHOLD=0.08
export MW_ML_SYMBOLS=BTC,ETH

echo "Configuration:"
echo "- Training data: 180 days"
echo "- Model type: hybrid (LR + GB)"
echo "- Regime filtering: ENABLED"
echo "- Volatility threshold: < 6%"
echo "- Trend strength threshold: > 8%"
echo ""

# Run training
python -m scripts.ml.train_predict

echo ""
echo "✓ Training complete!"
echo "Check artifacts/ml_* and models/ml_model_manifest.json for results"
