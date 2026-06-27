# experiments/run_error_analysis.py
"""
Error analysis: Categorize and analyze failure modes
Critical for understanding system behavior and limitations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agentic_self_rag import AgenticSelfRAG
from src.evaluation.metrics import exact_match, f1_score
from datasets import load_dataset
import json
import logging
from collections import Counter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def categorize_error(prediction: str, gold: str, metadata: dict) -> str:
    """
    Categorize error type
    
    Categories:
    - incomplete_answer: Answer too short/vague
    - hallucination: Answer contains unsupported information
    - retrieval_failure: Insufficient retrieval quality
    - reasoning_error: Logic error in multi-hop reasoning
    - entity_confusion: Wrong entity mentioned
    - semantic_mismatch: Correct meaning, wrong wording
    """
    pred_len = len(prediction.split())
    gold_len = len(gold.split())
    
    # Incomplete answer
    if pred_len < 5 or pred_len < gold_len * 0.3:
        return 'incomplete_answer'
    
    # Retrieval failure
    retrieval_quality = metadata.get('retrieval_critique', {}).get('sufficiency', 'unknown')
    if retrieval_quality == 'insufficient':
        return 'retrieval_failure'
    
    # Check F1 score for semantic similarity
    f1 = f1_score(prediction, gold)
    
    # Hallucination (very low F1)
    if f1 < 0.2:
        return 'hallucination'
    
    # Reasoning error (multi-hop query, moderate F1)
    if metadata.get('query_analysis', {}).get('is_multi_hop') and f1 < 0.5:
        return 'reasoning_error'
    
    # Entity confusion
    if any(word in prediction.lower() for word in ['who', 'what', 'which', 'when', 'where']):
        return 'entity_confusion'
    
    # Default: semantic mismatch
    return 'semantic_mismatch'

def analyze_errors(predictions: list, golds: list, metadata_list: list, question_types: list):
    """Comprehensive error analysis"""
    
    errors = []
    
    for i, (pred, gold, metadata, qtype) in enumerate(zip(predictions, golds, metadata_list, question_types)):
        if not exact_match(pred, gold):
            error = {
                'index': i,
                'prediction': pred[:200],
                'gold': gold,
                'f1': f1_score(pred, gold),
                'question_type': qtype,
                'retrieval_quality': metadata.get('retrieval_critique', {}).get('sufficiency', 'unknown'),
                'is_multi_hop': metadata.get('query_analysis', {}).get('is_multi_hop', False),
                'iterations': metadata.get('iterations', 0),
                'confidence': metadata.get('confidence', 0.0),
                'error_category': categorize_error(pred, gold, metadata)
            }
            errors.append(error)
    
    return errors

def print_error_analysis(errors: list):
    """Print comprehensive error analysis report"""
    
    print(f"\n{'='*80}")
    print("ERROR ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    print(f"Total Errors: {len(errors)}")
    
    # Distribution by category
    by_category = {}
    for error in errors:
        cat = error['error_category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(error)
    
    print(f"\nError Distribution by Category:")
    print("-" * 80)
    for category, errors_list in sorted(by_category.items(), key=lambda x: -len(x[1])):
        pct = len(errors_list) / len(errors) * 100
        avg_f1 = sum(e['f1'] for e in errors_list) / len(errors_list)
        print(f"  {category:<25}: {len(errors_list):>3} ({pct:>5.1f}%) - Avg F1: {avg_f1:.3f}")
    
    # Distribution by question type
    by_type = {}
    for error in errors:
        qtype = error['question_type']
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(error)
    
    print(f"\nErrors by Question Type:")
    print("-" * 80)
    for qtype, errors_list in by_type.items():
        print(f"  {qtype:<15}: {len(errors_list)} errors")
        
        # Breakdown by category within type
        type_categories = Counter([e['error_category'] for e in errors_list])
        for cat, count in type_categories.most_common(3):
            print(f"    - {cat}: {count}")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    print("-" * 80)
    
    # Retrieval quality vs errors
    insufficient_retrieval = [e for e in errors if e['retrieval_quality'] == 'insufficient']
    print(f"  Errors with insufficient retrieval: {len(insufficient_retrieval)} ({len(insufficient_retrieval)/len(errors):.1%})")
    
    # Multi-hop vs errors
    multi_hop_errors = [e for e in errors if e['is_multi_hop']]
    print(f"  Errors on multi-hop questions: {len(multi_hop_errors)} ({len(multi_hop_errors)/len(errors):.1%})")
    
    # Low confidence errors
    low_conf_errors = [e for e in errors if e['confidence'] < 0.5]
    print(f"  Errors with low confidence (<0.5): {len(low_conf_errors)} ({len(low_conf_errors)/len(errors):.1%})")
    
    # Example errors
    print(f"\nExample Errors by Category:")
    print("=" * 80)
    
    for category, errors_list in sorted(by_category.items(), key=lambda x: -len(x[1]))[:3]:
        if errors_list:
            ex = errors_list[0]
            print(f"\n{category.upper()}:")
            print(f"  Gold: {ex['gold']}")
            print(f"  Pred: {ex['prediction'][:150]}...")
            print(f"  F1: {ex['f1']:.3f}, Confidence: {ex['confidence']:.2f}")
            print(f"  Type: {ex['question_type']}, Multi-hop: {ex['is_multi_hop']}")

def main():
    """Run error analysis"""
    
    NUM_SAMPLES = 500
    
    logger.info("Running error analysis...")
    
    # Load test data
    dataset = load_dataset('hotpot_qa', name='fullwiki', split='validation')
    test_samples = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    
    # Initialize system
    logger.info("Initializing Agentic Self-RAG...")
    system = AgenticSelfRAG()
    all_contexts = [item['context']['sentences'] for item in test_samples]
    system.setup_vectorstore(all_contexts)
    
    # Run predictions
    logger.info("Generating predictions...")
    predictions = []
    golds = []
    metadata_list = []
    question_types = []
    
    for item in tqdm(test_samples):
        try:
            result = system.answer(item['question'])
            predictions.append(result['answer'])
            golds.append(item['answer'])
            metadata_list.append({
                'query_analysis': result['query_analysis'],
                'retrieval_critique': result['retrieval_critique'],
                'iterations': result['iterations'],
                'confidence': result['confidence']
            })
            question_types.append(item['type'])
        except Exception as e:
            logger.error(f"Error: {e}")
            predictions.append("")
            golds.append(item['answer'])
            metadata_list.append({})
            question_types.append(item.get('type', 'unknown'))
    
    # Analyze errors
    logger.info("Analyzing errors...")
    errors = analyze_errors(predictions, golds, metadata_list, question_types)
    
    # Print report
    print_error_analysis(errors)
    
    # Save results
    os.makedirs('results/error_analysis', exist_ok=True)
    
    with open('results/error_analysis/errors.json', 'w') as f:
        json.dump(errors, f, indent=2)
    
    logger.info(f"\n✓ Error analysis complete! ({len(errors)} errors analyzed)")

if __name__ == "__main__":
    main()