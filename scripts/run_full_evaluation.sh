#!/bin/bash
# scripts/run_full_evaluation.sh
# Full 500-sample evaluation

set -e

echo "🚀 Running full evaluation (500 samples)..."
echo "============================================"

# Check for API key
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ Error: GROQ_API_KEY not set"
    echo "   Please export GROQ_API_KEY=your_key"
    exit 1
fi

# Run main evaluation
echo "1. Main evaluation..."
python experiments/run_main_evaluation.py

# Run ablations
echo ""
echo "2. Ablation study..."
python experiments/run_ablations.py

echo ""
echo "✓ Full evaluation complete!"
echo "  Results: results/"
echo "  - main_evaluation/results_500.csv"
echo "  - main_evaluation/summary.json"
echo "  - ablations/ablation_results.json"