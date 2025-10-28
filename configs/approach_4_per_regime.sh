#!/usr/bin/env bash
# Approach 4: Per-Regime Models
# Train separate models for trending vs choppy markets

set -e

echo "=================================================="
echo "ML Improvement Approach 4: Per-Regime Models"
echo "=================================================="
echo ""
echo "Strategy: Train specialized models per market regime"
echo "- Model A: trained on trending markets only"
echo "- Model B: trained on choppy markets only"
echo "- Fallback: trained on all data (safety net)"
echo "- Inference: detect current regime → use appropriate model"
echo ""
echo "Expected impact: Best of both worlds - specialized performance"
echo "Target: 52%+ WR, 1.15+ PF consistently"
echo ""

# Set environment variables for this approach
export MW_ML_LOOKBACK_DAYS=365
export MW_ML_MODEL=hybrid
export MW_REGIME_ENABLED=1
export MW_PER_REGIME_MODELS=1
export MW_REGIME_VOLATILITY_THRESHOLD=0.06
export MW_REGIME_TREND_THRESHOLD=0.08
export MW_ML_SYMBOLS=BTC,ETH

echo "Configuration:"
echo "- Training data: 365 days"
echo "- Model type: hybrid (LR + GB) per regime"
echo "- Regime detection: ENABLED"
echo "- Per-regime training: ENABLED"
echo "- Models trained: trending, choppy, fallback"
echo ""

# Run training
python -m scripts.ml.train_predict

echo ""
echo "✓ Training complete!"
echo "Check artifacts/ml_* and models/ml_model_manifest.json for results"
echo ""
echo "Regime models saved to models/current/*_trending_model.joblib"
echo "                         models/current/*_choppy_model.joblib"
echo "                         models/current/*_fallback_model.joblib"
