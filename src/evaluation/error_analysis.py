# src/evaluation/error_analysis.py
"""
Comprehensive error analysis and failure categorization
Analyzes failure modes, correlations, and provides actionable insights
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import numpy as np
from src.evaluation.metrics import exact_match, f1_score
from src.utils.logging_utils import setup_logger

logger = setup_logger("error_analysis")

class ErrorCategories:
    """Error category definitions"""
    INCOMPLETE_ANSWER = "incomplete_answer"
    HALLUCINATION = "hallucination"
    RETRIEVAL_FAILURE = "retrieval_failure"
    REASONING_ERROR = "reasoning_error"
    ENTITY_CONFUSION = "entity_confusion"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    OVER_REFUSAL = "over_refusal"
    FACTUAL_ERROR = "factual_error"


class ErrorAnalyzer:
    """
    Analyzes errors and failure modes in RAG systems
    Categorizes and provides insights into error types
    """
    
    def __init__(self):
        self.errors = []
        self.categories = ErrorCategories()
    
    def analyze(self, predictions: List[Dict], gold_answers: List[str]) -> Dict:
        """
        Analyze errors across predictions
        
        Args:
            predictions: List of prediction dictionaries
            gold_answers: List of gold answer strings
            
        Returns:
            Dictionary with error analysis results
        """
        error_counts = Counter()
        
        for pred, gold in zip(predictions, gold_answers):
            answer = pred.get("answer", "")
            is_error = not exact_match(answer, gold)
            
            if is_error:
                error_type = self._categorize_error(answer, gold)
                error_counts[error_type] += 1
        
        return {
            "total_errors": sum(error_counts.values()),
            "error_breakdown": dict(error_counts),
            "error_rate": sum(error_counts.values()) / len(predictions) if predictions else 0,
        }
    
    def _categorize_error(self, prediction: str, gold: str) -> str:
        """Categorize error type"""
        pred_len = len(prediction.split())
        gold_len = len(gold.split())
        
        # Too short = incomplete answer
        if pred_len < gold_len / 2:
            return self.categories.INCOMPLETE_ANSWER
        
        # Too long with extra text = hallucination
        if pred_len > gold_len * 1.5:
            return self.categories.HALLUCINATION
        
        # Default to semantic mismatch
        return self.categories.SEMANTIC_MISMATCH


def categorize_error(
    prediction: str,
    gold: str,
    metadata: Dict,
    question_type: str
) -> str:
    """
    Categorize error type with detailed heuristics
    
    Args:
        prediction: Model prediction
        gold: Ground truth answer
        metadata: Additional metadata (retrieval quality, verification, etc.)
        question_type: Question type (bridge, comparison, etc.)
        
    Returns:
        Error category string
    """
    pred_len = len(prediction.split())
    gold_len = len(gold.split())
    f1 = f1_score(prediction, gold)
    
    # Over-refusal detection
    refusal_phrases = ["don't know", "cannot", "unable", "i don't", "not sure"]
    if any(phrase in prediction.lower() for phrase in refusal_phrases):
        return ErrorCategories.OVER_REFUSAL
    
    # Incomplete answer (too short or vague)
    if pred_len < 5 or pred_len < gold_len * 0.3:
        return ErrorCategories.INCOMPLETE_ANSWER
    
    # Retrieval failure
    retrieval_quality = metadata.get('retrieval_critique', {}).get('sufficiency', 'unknown')
    if retrieval_quality == 'insufficient':
        return ErrorCategories.RETRIEVAL_FAILURE
    
    # Hallucination (very low F1, prediction has content but wrong)
    if f1 < 0.2 and pred_len > 10:
        return ErrorCategories.HALLUCINATION
    
    # Reasoning error (multi-hop query with moderate F1)
    is_multi_hop = metadata.get('query_analysis', {}).get('is_multi_hop', False)
    if is_multi_hop and 0.2 <= f1 < 0.5:
        return ErrorCategories.REASONING_ERROR
    
    # Entity confusion (question asks for entity, wrong entity provided)
    question = metadata.get('question', '')
    entity_questions = ['who', 'what', 'which', 'where', 'when']
    if any(q in question.lower()[:20] for q in entity_questions) and 0.1 < f1 < 0.4:
        return ErrorCategories.ENTITY_CONFUSION
    
    # Factual error (moderate F1, some overlap but key facts wrong)
    if 0.3 <= f1 < 0.6:
        return ErrorCategories.FACTUAL_ERROR
    
    # Default: semantic mismatch
    return ErrorCategories.SEMANTIC_MISMATCH


def analyze_errors(
    predictions: List[str],
    golds: List[str],
    metadata_list: List[Dict],
    question_types: List[str],
    questions: List[str]
) -> Dict:
    """
    Comprehensive error analysis
    
    Args:
        predictions: Model predictions
        golds: Ground truth answers
        metadata_list: Metadata for each prediction
        question_types: Question types
        questions: Original questions
        
    Returns:
        Dictionary with error analysis results
    """
    errors = []
    
    for i, (pred, gold, metadata, qtype, question) in enumerate(
        zip(predictions, golds, metadata_list, question_types, questions)
    ):
        # Only analyze errors
        if not exact_match(pred, gold):
            metadata['question'] = question
            error_category = categorize_error(pred, gold, metadata, qtype)
            
            error = {
                'index': i,
                'question': question[:100],
                'question_type': qtype,
                'prediction': pred[:200],
                'gold': gold,
                'f1': f1_score(pred, gold),
                'error_category': error_category,
                'retrieval_quality': metadata.get('retrieval_critique', {}).get('sufficiency', 'unknown'),
                'is_multi_hop': metadata.get('query_analysis', {}).get('is_multi_hop', False),
                'iterations': metadata.get('iterations', 0),
                'confidence': metadata.get('confidence', 0.0),
                'verdict': metadata.get('verdict', 'UNKNOWN')
            }
            errors.append(error)
    
    return errors


def compute_error_statistics(errors: List[Dict]) -> Dict:
    """
    Compute comprehensive error statistics
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Statistics dictionary
    """
    if not errors:
        return {'total_errors': 0}
    
    stats = {
        'total_errors': len(errors),
        'by_category': Counter([e['error_category'] for e in errors]),
        'by_question_type': defaultdict(list),
        'by_retrieval_quality': Counter([e['retrieval_quality'] for e in errors]),
        'multi_hop_errors': sum(1 for e in errors if e['is_multi_hop']),
        'avg_f1': np.mean([e['f1'] for e in errors]),
        'avg_confidence': np.mean([e['confidence'] for e in errors]),
        'low_confidence_errors': sum(1 for e in errors if e['confidence'] < 0.5)
    }
    
    # Group by question type
    for error in errors:
        qtype = error['question_type']
        stats['by_question_type'][qtype].append(error['error_category'])
    
    # Category breakdown by question type
    stats['category_by_type'] = {}
    for qtype, categories in stats['by_question_type'].items():
        stats['category_by_type'][qtype] = Counter(categories)
    
    return stats


def print_error_analysis(errors: List[Dict], stats: Dict):
    """
    Print formatted error analysis report
    
    Args:
        errors: List of errors
        stats: Error statistics
    """
    print(f"\n{'='*80}")
    print("ERROR ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    print(f"Total Errors: {stats['total_errors']}")
    
    # Distribution by category
    print(f"\n{'Error Distribution by Category':}")
    print("-" * 80)
    for category, count in stats['by_category'].most_common():
        pct = count / stats['total_errors'] * 100
        category_errors = [e for e in errors if e['error_category'] == category]
        avg_f1 = np.mean([e['f1'] for e in category_errors])
        print(f"  {category:<30}: {count:>3} ({pct:>5.1f}%) - Avg F1: {avg_f1:.3f}")
    
    # Distribution by question type
    print(f"\n{'Errors by Question Type':}")
    print("-" * 80)
    for qtype, categories in stats['category_by_type'].items():
        total_type = sum(categories.values())
        print(f"  {qtype:<15}: {total_type} errors")
        
        # Top 3 categories for this type
        for cat, count in categories.most_common(3):
            print(f"    - {cat}: {count}")
    
    # Correlation analysis
    print(f"\n{'Correlation Analysis':}")
    print("-" * 80)
    
    print(f"  Multi-hop errors: {stats['multi_hop_errors']} "
          f"({stats['multi_hop_errors']/stats['total_errors']:.1%})")
    
    insufficient = sum(1 for e in errors if e['retrieval_quality'] == 'insufficient')
    print(f"  Errors with insufficient retrieval: {insufficient} "
          f"({insufficient/stats['total_errors']:.1%})")
    
    print(f"  Low confidence errors (<0.5): {stats['low_confidence_errors']} "
          f"({stats['low_confidence_errors']/stats['total_errors']:.1%})")
    
    print(f"  Average F1 on errors: {stats['avg_f1']:.3f}")
    print(f"  Average confidence on errors: {stats['avg_confidence']:.3f}")
    
    # Example errors
    print(f"\n{'Example Errors by Category':}")
    print("=" * 80)
    
    for category in list(stats['by_category'].keys())[:3]:
        category_errors = [e for e in errors if e['error_category'] == category]
        if category_errors:
            ex = category_errors[0]
            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  Question: {ex['question']}")
            print(f"  Gold: {ex['gold']}")
            print(f"  Pred: {ex['prediction'][:120]}...")
            print(f"  F1: {ex['f1']:.3f}, Confidence: {ex['confidence']:.2f}")
            print(f"  Type: {ex['question_type']}, Multi-hop: {ex['is_multi_hop']}")


def generate_recommendations(stats: Dict) -> List[str]:
    """
    Generate actionable recommendations based on error analysis
    
    Args:
        stats: Error statistics
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Check hallucination rate
    hallucination_rate = stats['by_category'].get('hallucination', 0) / stats['total_errors']
    if hallucination_rate > 0.15:
        recommendations.append(
            f"⚠️  High hallucination rate ({hallucination_rate:.1%}). "
            "Consider: (1) Stricter retrieval filtering, (2) Improved answer verification, "
            "(3) Add faithfulness constraints to generation"
        )
    
    # Check retrieval quality
    retrieval_failures = stats['by_category'].get('retrieval_failure', 0) / stats['total_errors']
    if retrieval_failures > 0.20:
        recommendations.append(
            f"⚠️  High retrieval failure rate ({retrieval_failures:.1%}). "
            "Consider: (1) Increase retrieval k, (2) Add query expansion, "
            "(3) Use hybrid retrieval (dense + sparse)"
        )
    
    # Check multi-hop performance
    multi_hop_pct = stats['multi_hop_errors'] / stats['total_errors']
    if multi_hop_pct > 0.40:
        recommendations.append(
            f"⚠️  Multi-hop questions challenging ({multi_hop_pct:.1%} of errors). "
            "Consider: (1) Improve query decomposition, (2) Add explicit sub-question answering, "
            "(3) Enhance reasoning chain verification"
        )
    
    # Check reasoning errors
    reasoning_rate = stats['by_category'].get('reasoning_error', 0) / stats['total_errors']
    if reasoning_rate > 0.15:
        recommendations.append(
            f"⚠️  High reasoning error rate ({reasoning_rate:.1%}). "
            "Consider: (1) Add chain-of-thought prompting, (2) Implement step-by-step verification, "
            "(3) Use reasoning-specialized models"
        )
    
    # Check over-refusal
    over_refusal_rate = stats['by_category'].get('over_refusal', 0) / stats['total_errors']
    if over_refusal_rate > 0.10:
        recommendations.append(
            f"⚠️  Over-refusal detected ({over_refusal_rate:.1%}). "
            "Consider: (1) Lower confidence thresholds, (2) Add retrieval fallbacks, "
            "(3) Implement answer attempt before refusing"
        )
    
    if not recommendations:
        recommendations.append("✓ Error distribution appears balanced. Continue monitoring performance.")
    
    return recommendations