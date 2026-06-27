#!/bin/bash
# scripts/run_smoke_test.sh
# Quick 5-sample smoke test

set -e

echo "🔥 Running smoke test (5 samples)..."
echo "===================================="

export NUM_SAMPLES=5

python experiments/run_main_evaluation.py

echo ""
echo "✓ Smoke test complete!"
echo "  Results: results/main_evaluation/"