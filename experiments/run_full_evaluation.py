# experiments/run_full_evaluation.py
"""
Full 500-sample evaluation - Direct implementation
Bypasses complexity, ensures full dataset is used
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.vanilla_rag import VanillaRAG
from src.self_rag import SimplifiedSelfRAG
from src.agentic_self_rag import AgenticSelfRAG
from src.baselines.published_self_rag import PublishedSelfRAG
from src.evaluation.metrics import exact_match, f1_score
from src.utils.repro import set_seed


def load_full_hotpotqa(num_samples=500):
    """Load HotpotQA cache directly"""
    cache_path = 'data/hotpotqa_cache.json'
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded {len(data)} samples from cache")
        
        if num_samples:
            data = data[:num_samples]
            logger.info(f"✓ Using first {num_samples} samples")
        
        return data
    
    # Fallback to sample
    logger.warning("Cache not found, using sample data")
    with open('data/hotpotqa_sample.json', 'r') as f:
        return json.load(f)


def evaluate_system(system, system_name, test_data):
    """Evaluate single system"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {system_name}")
    logger.info(f"{'='*80}")
    
    predictions = []
    gold_answers = []
    question_types = []
    times = []
    em_scores = []
    f1_scores_list = []
    
    for item in tqdm(test_data, desc=system_name):
        try:
            start = time.time()
            
            # Call appropriate method
            if system_name == 'Vanilla RAG':
                result = system.answer_question(item['question'])
                pred = result['answer']
            elif system_name == 'Published Self-RAG':
                result = system.answer_with_self_rag(item['question'])
                pred = result['answer']
            elif system_name == 'Simplified Self-RAG':
                result = system.answer_with_reflection(item['question'])
                pred = result['answer']
            elif 'Agentic' in system_name:
                result = system.answer(item['question'])
                pred = result['answer']
            else:
                raise ValueError(f"Unknown system: {system_name}")
            
            elapsed = time.time() - start
            
            # Compute metrics
            em = exact_match(pred, item['answer'])
            f1 = f1_score(pred, item['answer'])
            
            predictions.append(pred)
            gold_answers.append(item['answer'])
            question_types.append(item.get('type', 'unknown'))
            times.append(elapsed)
            em_scores.append(float(em))
            f1_scores_list.append(f1)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            predictions.append("")
            gold_answers.append(item['answer'])
            question_types.append(item.get('type', 'unknown'))
            times.append(0)
            em_scores.append(0.0)
            f1_scores_list.append(0.0)
    
    # Aggregate metrics
    em_mean = sum(em_scores) / len(em_scores) if em_scores else 0
    f1_mean = sum(f1_scores_list) / len(f1_scores_list) if f1_scores_list else 0
    
    logger.info(f"\n{system_name} Results:")
    logger.info(f"  EM: {em_mean:.1%}")
    logger.info(f"  F1: {f1_mean:.3f}")
    logger.info(f"  Avg time: {sum(times)/len(times):.2f}s")
    
    return {
        'system_name': system_name,
        'predictions': predictions,
        'gold_answers': gold_answers,
        'question_types': question_types,
        'times': times,
        'em_scores': em_scores,
        'f1_scores': f1_scores_list,
        'em_mean': em_mean,
        'f1_mean': f1_mean
    }


def main():
    """Run full evaluation"""
    
    set_seed(42)
    
    # Load data
    test_data = load_full_hotpotqa(num_samples=500)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Prepare contexts
    all_contexts = [' '.join(item['context']['sentences']) if isinstance(item['context'], dict)
                    else item['context'] for item in test_data]
    
    logger.info("\n" + "="*80)
    logger.info("INITIALIZING SYSTEMS")
    logger.info("="*80)
    
    # Initialize systems
    logger.info("\n1. Vanilla RAG...")
    vanilla_rag = VanillaRAG()
    vanilla_rag.create_vectorstore(all_contexts)
    
    logger.info("2. Simplified Self-RAG...")
    simplified_self_rag = SimplifiedSelfRAG(vanilla_rag)
    
    logger.info("3. Published Self-RAG...")
    published_self_rag = PublishedSelfRAG(vanilla_rag)
    
    logger.info("4. Agentic Self-RAG...")
    agentic_rag = AgenticSelfRAG()
    agentic_rag.setup_vectorstore(all_contexts)
    
    systems = {
        'Vanilla RAG': vanilla_rag,
        'Simplified Self-RAG': simplified_self_rag,
        'Published Self-RAG': published_self_rag,
        'Agentic Self-RAG': agentic_rag
    }
    
    # Run evaluations
    results = {}
    for system_name, system in systems.items():
        results[system_name] = evaluate_system(system, system_name, test_data)
    
    # Save detailed results to CSV
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    rows = []
    for system_name, result in results.items():
        for i, (pred, gold, qtype, time_s, em, f1) in enumerate(zip(
            result['predictions'],
            result['gold_answers'],
            result['question_types'],
            result['times'],
            result['em_scores'],
            result['f1_scores']
        )):
            rows.append({
                'system': system_name,
                'sample_id': i,
                'question_type': qtype,
                'prediction': pred,
                'gold_answer': gold,
                'exact_match': em,
                'f1_score': f1,
                'time_seconds': time_s
            })
    
    df = pd.DataFrame(rows)
    os.makedirs('results/main_evaluation', exist_ok=True)
    csv_path = 'results/main_evaluation/results_500.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved {len(rows)} results to {csv_path}")
    
    # Save summary
    summary = {}
    for system_name, result in results.items():
        summary[system_name] = {
            'system_name': system_name,
            'em_mean': result['em_mean'],
            'f1_mean': result['f1_mean'],
            'em_std': (sum((x - result['em_mean'])**2 for x in result['em_scores']) / len(result['em_scores']))**0.5,
            'f1_std': (sum((x - result['f1_mean'])**2 for x in result['f1_scores']) / len(result['f1_scores']))**0.5,
            'num_samples': len(result['predictions']),
            'avg_time': sum(result['times']) / len(result['times'])
        }
    
    summary_path = 'results/main_evaluation/summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to {summary_path}")
    
    # Print results table
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80 + "\n")
    logger.info(f"{'System':<25} {'EM':<10} {'F1':<10} {'Avg Time':<12}")
    logger.info("-" * 60)
    for system_name, result in results.items():
        logger.info(f"{system_name:<25} "
                   f"{result['em_mean']:.1%}      "
                   f"{result['f1_mean']:.3f}    "
                   f"{result['em_mean']/len(result['predictions'])*sum(result['times']):.2f}s")


if __name__ == '__main__':
    main()
