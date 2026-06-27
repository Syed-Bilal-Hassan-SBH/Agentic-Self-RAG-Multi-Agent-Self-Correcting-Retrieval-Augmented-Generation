# experiments/compare_baselines.py
"""
Comprehensive baseline comparison with advanced metrics
Includes hallucination detection, faithfulness scoring, and latency benchmarking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vanilla_rag import VanillaRAG
from src.self_rag import SimplifiedSelfRAG
from src.agentic_self_rag import AgenticSelfRAG
from src.baselines.published_self_rag import PublishedSelfRAG
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical_tests import compare_systems, print_comparison
from src.evaluation.error_analysis import analyze_errors, compute_error_statistics, print_error_analysis
from src.utils.repro import set_seed, get_environment_info
from src.utils.data_utils import load_hotpotqa
from src.utils.logging_utils import ExperimentLogger
import json
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm
import pandas as pd

# Advanced metrics imports
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


def compute_advanced_metrics(predictions: List[str], golds: List[str]) -> Dict:
    """
    Compute advanced metrics beyond EM/F1
    
    Metrics:
    - ROUGE-L (longest common subsequence)
    - BERTScore (semantic similarity)
    - Semantic entropy (uncertainty estimation)
    - Length statistics
    
    Args:
        predictions: Model predictions
        golds: Ground truth answers
        
    Returns:
        Dictionary with advanced metrics
    """
    metrics = {}
    
    # ROUGE-L
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(g, p)['rougeL'].fmeasure 
                       for p, g in zip(predictions, golds)]
        metrics['ROUGE-L'] = float(np.mean(rouge_scores))
        metrics['ROUGE-L_std'] = float(np.std(rouge_scores))
    
    # BERTScore
    if BERTSCORE_AVAILABLE:
        try:
            P, R, F1 = bert_score(predictions, golds, lang='en', verbose=False)
            metrics['BERTScore_F1'] = float(F1.mean())
            metrics['BERTScore_F1_std'] = float(F1.std())
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
    
    # Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    gold_lengths = [len(g.split()) for g in golds]
    
    metrics['avg_pred_length'] = float(np.mean(pred_lengths))
    metrics['avg_gold_length'] = float(np.mean(gold_lengths))
    metrics['length_ratio'] = float(np.mean(pred_lengths) / np.mean(gold_lengths))
    
    # Answer rate (non-empty, non-refusal)
    refusal_phrases = ["don't know", "cannot answer", "i don't", "not sure"]
    non_refusals = [
        p for p in predictions 
        if len(p.strip()) > 0 and not any(phrase in p.lower() for phrase in refusal_phrases)
    ]
    metrics['answer_rate'] = float(len(non_refusals) / len(predictions))
    
    return metrics


def compute_hallucination_metrics(
    predictions: List[str],
    contexts: List[List[str]],
    golds: List[str]
) -> Dict:
    """
    Compute hallucination detection metrics
    
    Metrics:
    - Context overlap ratio
    - Entity hallucination rate
    - Faithfulness score (heuristic)
    
    Args:
        predictions: Model predictions
        contexts: Retrieved contexts
        golds: Ground truth answers
        
    Returns:
        Dictionary with hallucination metrics
    """
    from sentence_transformers import SentenceTransformer, util
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    context_overlaps = []
    faithfulness_scores = []
    
    for pred, context_list, gold in zip(predictions, contexts, golds):
        # Skip empty predictions
        if not pred.strip():
            continue
        
        # Context overlap (token-level)
        pred_tokens = set(pred.lower().split())
        context_tokens = set(" ".join(context_list).lower().split())
        
        if pred_tokens:
            overlap = len(pred_tokens & context_tokens) / len(pred_tokens)
            context_overlaps.append(overlap)
        
        # Faithfulness (semantic similarity to context)
        context_text = " ".join(context_list)[:1000]  # Limit length
        pred_emb = model.encode(pred, convert_to_tensor=True)
        context_emb = model.encode(context_text, convert_to_tensor=True)
        
        faithfulness = float(util.cos_sim(pred_emb, context_emb)[0][0])
        faithfulness_scores.append(faithfulness)
    
    metrics = {
        'avg_context_overlap': float(np.mean(context_overlaps)) if context_overlaps else 0.0,
        'avg_faithfulness': float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
        'low_faithfulness_rate': float(sum(1 for s in faithfulness_scores if s < 0.5) / len(faithfulness_scores)) if faithfulness_scores else 0.0
    }
    
    return metrics


def evaluate_system_comprehensive(
    system,
    system_name: str,
    test_data: List[Dict],
    logger: ExperimentLogger
) -> Dict:
    """
    Comprehensive system evaluation with all metrics
    
    Args:
        system: System instance
        system_name: System name
        test_data: Test samples
        logger: Experiment logger
        
    Returns:
        Complete evaluation results
    """
    logger.logger.info(f"\n{'='*80}")
    logger.logger.info(f"Evaluating: {system_name}")
    logger.logger.info(f"{'='*80}")
    
    predictions = []
    gold_answers = []
    question_types = []
    questions = []
    contexts = []
    times = []
    metadata_list = []
    errors = []
    
    for i, item in enumerate(tqdm(test_data, desc=f"{system_name}")):
        try:
            start_time = time.time()
            
            # Call appropriate method
            if system_name == 'Vanilla RAG':
                result = system.answer_question(item['question'])
                pred = result['answer']
                metadata = {'method': 'vanilla'}
            elif system_name == 'Published Self-RAG':
                result = system.answer_with_self_rag(item['question'])
                pred = result['answer']
                metadata = result
            elif system_name == 'Simplified Self-RAG':
                result = system.answer_with_reflection(item['question'])
                pred = result['answer']
                metadata = result
            elif 'Agentic' in system_name:
                result = system.answer(item['question'])
                pred = result['answer']
                metadata = result
            else:
                raise ValueError(f"Unknown system: {system_name}")
            
            elapsed = time.time() - start_time
            
            predictions.append(pred)
            gold_answers.append(item['answer'])
            question_types.append(item['type'])
            questions.append(item['question'])
            contexts.append(item['context']['sentences'])
            times.append(elapsed)
            metadata_list.append(metadata)
            
        except Exception as e:
            logger.logger.error(f"Error on sample {i}: {str(e)}")
            errors.append({'index': i, 'error': str(e)})
            predictions.append("")
            gold_answers.append(item['answer'])
            question_types.append(item.get('type', 'unknown'))
            questions.append(item['question'])
            contexts.append(item['context']['sentences'])
            times.append(0)
            metadata_list.append({})
    
    # Compute standard metrics
    logger.logger.info("Computing standard metrics...")
    standard_metrics = compute_metrics(predictions, gold_answers)
    
    # Compute advanced metrics
    logger.logger.info("Computing advanced metrics...")
    advanced_metrics = compute_advanced_metrics(predictions, gold_answers)
    
    # Compute hallucination metrics
    logger.logger.info("Computing hallucination metrics...")
    hallucination_metrics = compute_hallucination_metrics(predictions, contexts, gold_answers)
    
    # Error analysis
    logger.logger.info("Analyzing errors...")
    error_list = analyze_errors(predictions, gold_answers, metadata_list, question_types, questions)
    error_stats = compute_error_statistics(error_list)
    
    # Latency statistics
    latency_stats = {
        'mean_latency': float(np.mean(times)),
        'median_latency': float(np.median(times)),
        'p95_latency': float(np.percentile(times, 95)),
        'p99_latency': float(np.percentile(times, 99))
    }
    
    # Log summary
    logger.logger.info(f"\n{system_name} Results:")
    logger.logger.info(f"  EM: {standard_metrics['EM']:.1%} ± {standard_metrics['EM_std']:.3f}")
    logger.logger.info(f"  F1: {standard_metrics['F1']:.3f} ± {standard_metrics['F1_std']:.3f}")
    if ROUGE_AVAILABLE:
        logger.logger.info(f"  ROUGE-L: {advanced_metrics.get('ROUGE-L', 0):.3f}")
    if BERTSCORE_AVAILABLE:
        logger.logger.info(f"  BERTScore: {advanced_metrics.get('BERTScore_F1', 0):.3f}")
    logger.logger.info(f"  Faithfulness: {hallucination_metrics['avg_faithfulness']:.3f}")
    logger.logger.info(f"  Mean Latency: {latency_stats['mean_latency']:.2f}s")
    logger.logger.info(f"  Errors: {len(errors)}")
    
    return {
        'system_name': system_name,
        'predictions': predictions,
        'gold_answers': gold_answers,
        'question_types': question_types,
        'times': times,
        'metadata_list': metadata_list,
        'errors': errors,
        'standard_metrics': standard_metrics,
        'advanced_metrics': advanced_metrics,
        'hallucination_metrics': hallucination_metrics,
        'latency_stats': latency_stats,
        'error_list': error_list,
        'error_stats': error_stats
    }


def save_baseline_results(results: Dict, output_dir: str = "results/baselines"):
    """Save baseline comparison results to JSON and CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare summary data
    summary_data = {}
    for system_name, result in results.items():
        summary_data[system_name] = {
            'standard_metrics': result['standard_metrics'],
            'advanced_metrics': result['advanced_metrics'],
            'hallucination_metrics': result['hallucination_metrics'],
            'latency_stats': result['latency_stats'],
            'error_stats': result['error_stats'],
            'num_errors': len(result['errors'])
        }
    
    # Save to JSON
    summary_path = os.path.join(output_dir, 'baseline_comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    # Create comparison dataframe
    rows = []
    for system_name, result in results.items():
        rows.append({
            'System': system_name,
            'EM (%)': result['standard_metrics']['EM'] * 100,
            'F1': result['standard_metrics']['F1'],
            'ROUGE-L': result['advanced_metrics'].get('ROUGE-L', 0),
            'BERTScore': result['advanced_metrics'].get('BERTScore_F1', 0),
            'Faithfulness': result['hallucination_metrics']['avg_faithfulness'],
            'Context Overlap': result['hallucination_metrics']['avg_context_overlap'],
            'Mean Latency (s)': result['latency_stats']['mean_latency'],
            'P95 Latency (s)': result['latency_stats']['p95_latency'],
            'Error Rate (%)': (len(result['errors']) / len(result['predictions']) * 100)
        })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'baseline_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table: {csv_path}")
    
    # Print formatted table
    print(f"\n{'='*120}")
    print("BASELINE COMPARISON RESULTS")
    print(f"{'='*120}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*120}\n")


def perform_pairwise_comparisons(results: Dict, output_dir: str = "results/baselines"):
    """Perform statistical comparisons between all pairs of systems"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("PAIRWISE STATISTICAL COMPARISONS")
    print(f"{'='*80}\n")
    
    system_names = list(results.keys())
    comparison_results = {}
    
    for i, sys_a in enumerate(system_names):
        for sys_b in system_names[i+1:]:
            scores_a = results[sys_a]['standard_metrics']['F1_scores']
            scores_b = results[sys_b]['standard_metrics']['F1_scores']
            
            comparison = compare_systems(scores_a, scores_b, sys_a, sys_b, alpha=0.05)
            print_comparison(comparison)
            
            comparison_key = f"{sys_a} vs {sys_b}"
            comparison_results[comparison_key] = comparison
    
    # Save comparisons
    comp_path = os.path.join(output_dir, 'pairwise_comparisons.json')
    with open(comp_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"✓ Saved pairwise comparisons: {comp_path}")
    
    return comparison_results


def main():
    """Run comprehensive baseline comparison"""
    
    # Setup logging
    from src.utils.logging_utils import setup_logger
    logger = setup_logger("baseline_comparison", log_file="logs/baseline_comparison.log")
    
    # Configuration
    num_samples = int(os.getenv('NUM_SAMPLES', 500))
    random_seed = 42
    
    logger.info(f"\n{'='*80}")
    logger.info(f"BASELINE COMPARISON EVALUATION")
    logger.info(f"Samples: {num_samples}")
    logger.info(f"Seed: {random_seed}")
    logger.info(f"{'='*80}\n")
    
    # Set seed
    set_seed(random_seed)
    
    # Load data
    logger.info("Loading test data...")
    test_data = load_hotpotqa(
        split='validation',
        num_samples=num_samples,
        random_seed=random_seed
    )
    logger.info(f"✓ Loaded {len(test_data)} samples from HotpotQA")
    
    # Initialize systems
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING SYSTEMS")
    logger.info(f"{'='*80}\n")
    
    all_contexts = [item['context']['sentences'] for item in test_data]
    
    logger.info("1. Vanilla RAG...")
    vanilla_rag = VanillaRAG()
    vanilla_rag.create_vectorstore(all_contexts)
    logger.info("✓ Vanilla RAG initialized")
    
    logger.info("2. Simplified Self-RAG...")
    simplified_self_rag = SimplifiedSelfRAG(vanilla_rag)
    logger.info("✓ Simplified Self-RAG initialized")
    
    logger.info("3. Published Self-RAG...")
    published_self_rag = PublishedSelfRAG(vanilla_rag)
    logger.info("✓ Published Self-RAG initialized")
    
    logger.info("4. Agentic Self-RAG...")
    agentic_rag = AgenticSelfRAG()
    agentic_rag.setup_vectorstore(all_contexts)
    logger.info("✓ Agentic Self-RAG initialized")
    
    # Define systems
    systems = {
        'Vanilla RAG': vanilla_rag,
        'Simplified Self-RAG': simplified_self_rag,
        'Published Self-RAG': published_self_rag,
        'Agentic Self-RAG': agentic_rag
    }
    
    # Evaluate each system
    logger.info(f"\n{'='*80}")
    logger.info("RUNNING EVALUATIONS")
    logger.info(f"{'='*80}\n")
    
    results = {}
    for system_name, system in systems.items():
        result = evaluate_system_comprehensive(system, system_name, test_data, logger)
        results[system_name] = result
    
    # Save results
    output_dir = "results/baselines"
    logger.info(f"\nSaving results to {output_dir}...")
    save_baseline_results(results, output_dir)
    
    # Perform pairwise comparisons
    logger.info(f"\n{'='*80}")
    logger.info("STATISTICAL TESTING")
    logger.info(f"{'='*80}\n")
    comparisons = perform_pairwise_comparisons(results, output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info("✓ BASELINE COMPARISON COMPLETE")
    logger.info(f"{'='*80}\n")
    
    return results, comparisons


if __name__ == "__main__":
    main()