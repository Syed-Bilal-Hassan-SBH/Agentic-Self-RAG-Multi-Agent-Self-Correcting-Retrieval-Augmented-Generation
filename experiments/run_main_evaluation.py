# experiments/run_main_evaluation.py
"""
Main evaluation script for 500-sample comparison
COMPLETE REWRITE: Proper metrics, baselines, statistical tests
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vanilla_rag import VanillaRAG
from src.self_rag import SimplifiedSelfRAG
from src.agentic_self_rag import AgenticSelfRAG
from src.ultra_agentic_rag import UltraAgenticRAG
from src.baselines.published_self_rag import PublishedSelfRAG
from src.evaluation.metrics import (
    compute_metrics, 
    compute_metrics_by_type,
    compute_hallucination_metrics,
    compute_refusal_metrics,
    compute_length_metrics
)
from src.evaluation.statistical_tests import compare_systems, print_comparison
from src.utils.repro import set_seed, get_environment_info
from src.utils.data_utils import load_hotpotqa
import json
import time
import logging
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import yaml
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'configs/experiment_config.yaml') -> Dict:
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_system(system, system_name: str, test_data: List[Dict], live_csv_path: str = None) -> Dict:
    """
    Evaluate a single system on test data
    
    Args:
        system: System instance with .answer() or .answer_question() method
        system_name: Name for logging
        test_data: List of test samples
        live_csv_path: Optional path to live CSV logging file
        
    Returns:
        Dictionary with predictions, metrics, and metadata
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {system_name}")
    logger.info(f"{'='*80}")
    
    predictions = []
    gold_answers = []
    question_types = []
    times = []
    errors = []
    contexts = []  # Track retrieved contexts for hallucination detection
    
    for i, item in enumerate(tqdm(test_data, desc=f"{system_name}")):
        try:
            start_time = time.time()
            
            # Call appropriate method based on system
            if system_name == 'Vanilla RAG':
                result = system.answer_question(item['question'])
                pred = result['answer']
                context = item.get('context', [])
            elif system_name == 'Published Self-RAG':
                result = system.answer_with_self_rag(item['question'])
                pred = result['answer']
                context = item.get('context', [])
            elif system_name == 'Simplified Self-RAG':
                result = system.answer_with_reflection(item['question'])
                pred = result['answer']
                context = item.get('context', [])
            elif system_name == 'Agentic Self-RAG':
                result = system.answer(item['question'])
                pred = result['answer']
                context = item.get('context', [])
            elif system_name == 'Ultra Agentic RAG':
                result = system.answer(item['question'])
                pred = result['answer']
                context = item.get('context', [])
            else:
                raise ValueError(f"Unknown system: {system_name}")
            
            elapsed = time.time() - start_time
            
            predictions.append(pred)
            gold_answers.append(item['answer'])
            question_types.append(item['type'])
            times.append(elapsed)
            contexts.append(context if isinstance(context, list) else [str(context)])
            
            # Live per-sample logging
            if live_csv_path is not None:
                em_val = compute_metrics([pred], [item['answer']])['EM']  # EM as 0/1
                f1_val = compute_metrics([pred], [item['answer']])['F1']
                with open(live_csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        i,
                        system_name,
                        item.get('question', ''),
                        item.get('answer', ''),
                        pred,
                        float(em_val),
                        float(f1_val),
                        round(elapsed, 4)
                    ])
            
        except Exception as e:
            logger.error(f"Error on sample {i}: {str(e)}")
            errors.append({'index': i, 'error': str(e)})
            predictions.append("")
            gold_answers.append(item['answer'])
            question_types.append(item.get('type', 'unknown'))
            times.append(0)
            contexts.append(item.get('context', []))
    
    # Compute metrics
    logger.info("Computing metrics...")
    overall_metrics = compute_metrics(predictions, gold_answers)
    by_type_metrics = compute_metrics_by_type(predictions, gold_answers, question_types)
    
    # Compute hallucination metrics
    logger.info("Computing hallucination metrics...")
    hallucination_metrics = compute_hallucination_metrics(predictions, contexts, gold_answers)
    
    # Compute refusal metrics
    refusal_metrics = compute_refusal_metrics(predictions)
    
    # Compute length metrics
    length_metrics = compute_length_metrics(predictions, gold_answers)
    
    # Log summary
    logger.info(f"\n{system_name} Results:")
    logger.info(f"  EM: {overall_metrics['EM']:.1%} +/- {overall_metrics['EM_std']:.3f}")
    logger.info(f"  F1: {overall_metrics['F1']:.3f} +/- {overall_metrics['F1_std']:.3f}")
    logger.info(f"  Avg time: {sum(times)/len(times):.2f}s")
    logger.info(f"  Errors: {len(errors)}")
    logger.info(f"\n  [HALLUCINATION METRICS]")
    logger.info(f"    Context overlap: {hallucination_metrics['avg_context_overlap']:.1%}")
    logger.info(f"    Faithfulness: {hallucination_metrics['avg_faithfulness']:.3f}")
    logger.info(f"    Entity hallucination rate: {hallucination_metrics['avg_entity_hallucination_rate']:.1%}")
    logger.info(f"    Low overlap (<30%): {hallucination_metrics['low_overlap_rate']:.1%}")
    logger.info(f"\n  [REFUSAL METRICS]")
    logger.info(f"    Refusal rate: {refusal_metrics['refusal_rate']:.1%}")
    logger.info(f"    Answer rate: {refusal_metrics['answer_rate']:.1%}")
    logger.info(f"\n  [LENGTH METRICS]")
    logger.info(f"    Avg pred length: {length_metrics['avg_pred_length']:.1f} tokens")
    logger.info(f"    Avg gold length: {length_metrics['avg_ref_length']:.1f} tokens")
    logger.info(f"    Overly long: {length_metrics['overly_long_rate']:.1%}")
    
    for qtype, metrics in by_type_metrics.items():
        logger.info(f"  {qtype}: EM={metrics['EM']:.1%}, F1={metrics['F1']:.3f}")
    
    return {
        'system_name': system_name,
        'predictions': predictions,
        'gold_answers': gold_answers,
        'question_types': question_types,
        'times': times,
        'errors': errors,
        'contexts': contexts,
        'overall_metrics': overall_metrics,
        'by_type_metrics': by_type_metrics,
        'hallucination_metrics': hallucination_metrics,
        'refusal_metrics': refusal_metrics,
        'length_metrics': length_metrics
    }


def save_results(results: Dict[str, Dict], output_dir: str, config: Dict):
    """Save results to CSV and JSON formats"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed predictions to CSV
    rows = []
    for system_name, result in results.items():
        for i, (pred, gold, qtype, time_taken) in enumerate(zip(
            result['predictions'],
            result['gold_answers'],
            result['question_types'],
            result['times']
        )):
            from src.evaluation.metrics import exact_match, f1_score
            rows.append({
                'system': system_name,
                'sample_id': i,
                'question_type': qtype,
                'prediction': pred,
                'gold_answer': gold,
                'exact_match': exact_match(pred, gold),
                'f1_score': f1_score(pred, gold),
                'time_seconds': time_taken
            })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'results_500.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved detailed results: {csv_path}")
    
    # Save summary to JSON
    summary = {}
    for system_name, result in results.items():
        summary[system_name] = {
            'system_name': result['system_name'],
            'overall_metrics': {
                'EM': float(result['overall_metrics']['EM']),
                'F1': float(result['overall_metrics']['F1']),
                'EM_std': float(result['overall_metrics']['EM_std']),
                'F1_std': float(result['overall_metrics']['F1_std']),
                'num_samples': int(result['overall_metrics']['num_samples'])
            },
            'by_type_metrics': {
                qtype: {
                    'EM': float(metrics['EM']),
                    'F1': float(metrics['F1']),
                    'num_samples': int(metrics['num_samples'])
                }
                for qtype, metrics in result['by_type_metrics'].items()
            },
            'hallucination_metrics': {
                'avg_context_overlap': float(result['hallucination_metrics']['avg_context_overlap']),
                'avg_faithfulness': float(result['hallucination_metrics']['avg_faithfulness']),
                'avg_entity_hallucination_rate': float(result['hallucination_metrics']['avg_entity_hallucination_rate']),
                'low_overlap_rate': float(result['hallucination_metrics']['low_overlap_rate']),
                'low_faithfulness_rate': float(result['hallucination_metrics']['low_faithfulness_rate'])
            },
            'refusal_metrics': {
                'refusal_rate': float(result['refusal_metrics']['refusal_rate']),
                'answer_rate': float(result['refusal_metrics']['answer_rate']),
                'num_refusals': int(result['refusal_metrics']['num_refusals']),
                'num_answered': int(result['refusal_metrics']['num_answered'])
            },
            'length_metrics': {
                'avg_pred_length': float(result['length_metrics']['avg_pred_length']),
                'avg_ref_length': float(result['length_metrics']['avg_ref_length']),
                'length_ratio': float(result['length_metrics']['length_ratio']),
                'overly_long_rate': float(result['length_metrics']['overly_long_rate'])
            },
            'avg_time': float(sum(result['times']) / len(result['times'])),
            'num_errors': len(result['errors'])
        }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary: {summary_path}")
    
    # Save metadata
    metadata = {
        'config': config,
        'environment': get_environment_info(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata: {metadata_path}")


def compare_all_systems(results: Dict[str, Dict], output_dir: str):
    """Perform statistical comparisons between all systems"""
    
    logger.info(f"\n{'='*80}")
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info(f"{'='*80}")
    
    system_names = list(results.keys())
    comparisons = []
    
    # All pairwise comparisons
    for i, sys_a in enumerate(system_names):
        for sys_b in system_names[i+1:]:
            scores_a = results[sys_a]['overall_metrics']['F1_scores']
            scores_b = results[sys_b]['overall_metrics']['F1_scores']
            
            comparison = compare_systems(scores_a, scores_b, sys_a, sys_b, alpha=0.05)
            print_comparison(comparison)
            comparisons.append(comparison)
    
    # Save comparisons
    comp_path = os.path.join(output_dir, 'statistical_comparisons.json')
    with open(comp_path, 'w') as f:
        json.dump(comparisons, f, indent=2)
    logger.info(f"✓ Saved statistical tests: {comp_path}")
    
    return comparisons


def print_results_table(results: Dict[str, Dict]):
    """Print formatted results table"""
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY TABLE")
    print(f"{'='*80}\n")
    
    print(f"{'System':<25} {'EM':<10} {'F1':<10} {'Time (s)':<12} {'Errors':<8}")
    print("-" * 65)
    
    for system_name, result in results.items():
        metrics = result['overall_metrics']
        avg_time = sum(result['times']) / len(result['times'])
        num_errors = len(result['errors'])
        
        print(f"{system_name:<25} "
              f"{metrics['EM']:.1%}      "
              f"{metrics['F1']:.3f}    "
              f"{avg_time:>6.2f}       "
              f"{num_errors}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run main evaluation"""
    
    # Load configuration
    config = load_config()
    
    # Set random seed
    set_seed(config['experiment']['random_seed'])
    
    # Determine sample size
    num_samples = int(os.getenv('NUM_SAMPLES', config['experiment']['num_samples']))
    
    logger.info(f"\n{'='*80}")
    logger.info(f"AGENTIC SELF-RAG EVALUATION")
    logger.info(f"Samples: {num_samples}")
    logger.info(f"Seed: {config['experiment']['random_seed']}")
    logger.info(f"{'='*80}\n")
    
    # Load test data
    logger.info("Loading test data...")
    test_data = load_hotpotqa(
        split=config['dataset']['split'],
        num_samples=num_samples,
        cache_dir=config['dataset']['cache_dir'],
        random_seed=config['experiment']['random_seed']
    )
    
    # Initialize systems
    logger.info("\n" + "="*80)
    logger.info("INITIALIZING SYSTEMS")
    logger.info("="*80)
    
    all_contexts = [item['context']['sentences'] for item in test_data]
    
    logger.info("\n1. Vanilla RAG...")
    vanilla_rag = VanillaRAG()
    vanilla_rag.create_vectorstore(all_contexts)
    
    logger.info("2. Simplified Self-RAG...")
    simplified_self_rag = SimplifiedSelfRAG(vanilla_rag)
    
    logger.info("3. Published Self-RAG...")
    published_self_rag = PublishedSelfRAG(vanilla_rag)
    
    logger.info("4. Agentic Self-RAG...")
    agentic_rag = AgenticSelfRAG(
        rag=vanilla_rag,
        use_multi_hop=config['agentic_rag']['use_multi_hop'],
        use_adaptive_iteration=config['agentic_rag']['use_adaptive_iteration'],
        max_iterations=config['agentic_rag']['max_iterations'],
        confidence_threshold_high=config['agentic_rag']['confidence_threshold_high'],
        confidence_threshold_low=config['agentic_rag']['confidence_threshold_low']
    )
    agentic_rag.setup_vectorstore(all_contexts)
    
    logger.info("5. Ultra Agentic RAG...")
    ultra_agentic_rag = UltraAgenticRAG(
        max_iterations=1,
        high_confidence_threshold=0.70,
        refinement_enabled=False,  # Disabled to avoid rate limits
        use_query_analysis=False   # Disabled to avoid rate limits
    )
    ultra_agentic_rag.setup_vectorstore(all_contexts)
    
    systems = {
        'Vanilla RAG': vanilla_rag,
        'Simplified Self-RAG': simplified_self_rag,
        'Published Self-RAG': published_self_rag,
        'Agentic Self-RAG': agentic_rag,
        'Ultra Agentic RAG': ultra_agentic_rag
    }
    
    output_dir = 'results/main_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    live_csv_path = os.path.join(output_dir, 'live_results_main.csv')
    # Initialize/overwrite live CSV with header
    with open(live_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id',
            'system',
            'question',
            'gold_answer',
            'prediction',
            'exact_match',
            'f1_score',
            'time_seconds'
        ])
    
    logger.info(f"Live per-sample CSV: {live_csv_path}")
    
    # Evaluate all systems
    results = {}
    for system_name, system in systems.items():
        result = evaluate_system(system, system_name, test_data, live_csv_path=live_csv_path)
        results[system_name] = result
    
    # Statistical comparisons
    comparisons = compare_all_systems(results, output_dir)
    
    # Print results table
    print_results_table(results)
    
    # Save results
    save_results(results, output_dir, config)
    
    logger.info("\n✓ Main evaluation complete!")
    logger.info(f"  Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()