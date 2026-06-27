"""
Evaluation module for comprehensive metrics and analysis
Standard metrics (EM, F1), advanced metrics (ROUGE, BERTScore),
hallucination detection, and statistical testing
"""

from src.evaluation.metrics import (
    normalize_answer,
    exact_match,
    f1_score,
    compute_rouge_scores,
    compute_hallucination_metrics,
    compute_semantic_entropy,
)
from src.evaluation.statistical_tests import (
    paired_t_test,
    wilcoxon_test,
    cohen_d,
    bootstrap_confidence_interval,
    effect_size_hedges_g,
    power_analysis,
)
from src.evaluation.error_analysis import ErrorAnalyzer

__all__ = [
    # Metrics
    "normalize_answer",
    "exact_match",
    "f1_score",
    "compute_rouge_scores",
    "compute_hallucination_metrics",
    "compute_semantic_entropy",
    # Statistical Tests
    "paired_t_test",
    "wilcoxon_test",
    "cohen_d",
    "bootstrap_confidence_interval",
    "effect_size_hedges_g",
    "power_analysis",
    # Error Analysis
    "ErrorAnalyzer",
]
