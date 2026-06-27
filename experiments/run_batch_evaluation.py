#!/usr/bin/env python3
# experiments/run_batch_evaluation.py
"""
Batch evaluation with real-time output and side-by-side comparison
Processes 500 samples in configurable batch sizes
Displays comprehensive metrics and saves side-by-side results
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
    compute_length_metrics,
    exact_match,
    f1_score
)
from src.utils.repro import set_seed, get_environment_info
from src.utils.data_utils import load_hotpotqa
import json
import time
import logging
import csv
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
import yaml
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchEvaluator:
    """Handles batch evaluation with progress tracking"""
    
    def __init__(self, batch_size: int = 50, live_csv_path: str = None):
        self.batch_size = batch_size
        self.live_csv_path = live_csv_path
        self.batch_results = []
        self.current_batch = 0
        
    def process_batch(self, 
                     systems: Dict,
                     test_data: List[Dict],
                     batch_start: int,
                     batch_end: int) -> Dict:
        """Process a single batch of samples"""
        
        self.current_batch += 1
        batch_data = test_data[batch_start:batch_end]
        batch_num = f"Batch {self.current_batch} ({batch_start+1}-{batch_end})"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {batch_num}")
        logger.info(f"{'='*80}")
        
        batch_results = {}
        
        for system_name, system in systems.items():
            logger.info(f"\n  Evaluating {system_name} ({len(batch_data)} samples)...")
            
            predictions = []
            times = []
            errors = []
            
            for i, item in enumerate(tqdm(batch_data, desc=f"  {system_name}", leave=False)):
                try:
                    start_time = time.time()
                    
                    # Call appropriate method based on system
                    if system_name == 'Vanilla RAG':
                        result = system.answer_question(item['question'])
                        pred = result['answer']
                    elif system_name == 'Published Self-RAG':
                        result = system.answer_with_self_rag(item['question'])
                        pred = result['answer']
                    elif system_name == 'Simplified Self-RAG':
                        result = system.answer_with_reflection(item['question'])
                        pred = result['answer']
                    elif system_name == 'Agentic Self-RAG':
                        result = system.answer(item['question'])
                        pred = result['answer']
                    elif system_name == 'Ultra Agentic RAG':
                        result = system.answer(item['question'])
                        pred = result['answer']
                    else:
                        raise ValueError(f"Unknown system: {system_name}")
                    
                    elapsed = time.time() - start_time
                    predictions.append(pred)
                    times.append(elapsed)
                    
                    # Live per-sample logging (tall format: one row per system per sample)
                    if self.live_csv_path is not None:
                        global_index = batch_start + i
                        gold_answer = item.get('answer', '')
                        question = item.get('question', '')
                        em_val = exact_match(pred, gold_answer)
                        f1_val = f1_score(pred, gold_answer)
                        with open(self.live_csv_path, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                self.current_batch,
                                global_index,
                                system_name,
                                question,
                                gold_answer,
                                pred,
                                int(em_val),
                                round(f1_val, 4),
                                round(elapsed, 4)
                            ])
                    
                except Exception as e:
                    logger.error(f"    Error on sample {batch_start + i}: {str(e)}")
                    errors.append({'index': batch_start + i, 'error': str(e)})
                    predictions.append("")
                    times.append(0)
            
            # Store batch results
            batch_results[system_name] = {
                'predictions': predictions,
                'times': times,
                'errors': errors
            }
        
        # Display batch summary
        self._print_batch_summary(batch_results, batch_data, batch_num)
        
        return batch_results
    
    def _print_batch_summary(self, batch_results: Dict, batch_data: List[Dict], batch_label: str):
        """Print summary for current batch"""
        
        logger.info(f"\n{batch_label} RESULTS:\n")
        logger.info(f"{'System':<25} {'EM':<8} {'F1':<8} {'Avg Time':<12} {'Errors':<8}")
        logger.info("-" * 65)
        
        for system_name, results in batch_results.items():
            predictions = results['predictions']
            gold_answers = [item['answer'] for item in batch_data]
            times = results['times']
            errors = results['errors']
            
            # Calculate metrics
            em_scores = [float(exact_match(p, g)) for p, g in zip(predictions, gold_answers)]
            f1_scores = [f1_score(p, g) for p, g in zip(predictions, gold_answers)]
            
            avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            avg_time = sum(times) / len(times) if times else 0
            
            logger.info(f"{system_name:<25} {avg_em:>6.1%} {avg_f1:>8.3f} {avg_time:>10.2f}s {len(errors):>8}")

def save_comprehensive_results(all_results: Dict, 
                              test_data: List[Dict],
                              output_dir: str):
    """Save comprehensive results in multiple formats"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SIDE-BY-SIDE CSV - One row per sample with all systems' results
    logger.info(f"\nSaving side-by-side results...")
    systems = list(all_results['systems'].keys())
    sidebyside_rows = []
    
    for i, item in enumerate(test_data):
        row = {
            'sample_id': i,
            'question': item['question'][:200],
            'gold_answer': item['answer'],
            'question_type': item['type']
        }
        
        # Add each system's results
        for system_name in systems:
            predictions = all_results['systems'][system_name]['predictions']
            times = all_results['systems'][system_name]['times']
            pred = predictions[i] if i < len(predictions) else ''
            time_taken = times[i] if i < len(times) else 0
            em = exact_match(pred, item['answer'])
            f1 = f1_score(pred, item['answer'])
            
            row[f'{system_name}_prediction'] = pred
            row[f'{system_name}_em'] = int(em)
            row[f'{system_name}_f1'] = round(f1, 4)
            row[f'{system_name}_time'] = round(time_taken, 4)
        
        sidebyside_rows.append(row)
    
    df_sidebyside = pd.DataFrame(sidebyside_rows)
    sidebyside_csv = os.path.join(output_dir, 'sidebyside_results_500.csv')
    df_sidebyside.to_csv(sidebyside_csv, index=False)
    logger.info(f"✓ Saved side-by-side: {sidebyside_csv}")
    
    # 2. Detailed CSV with all predictions (one row per system)
    logger.info(f"Saving detailed results...")
    rows = []
    for system_name in systems:
        predictions = all_results['systems'][system_name]['predictions']
        times = all_results['systems'][system_name]['times']
        
        for i, (pred, item, time_taken) in enumerate(zip(predictions, test_data, times)):
            em = exact_match(pred, item['answer'])
            f1 = f1_score(pred, item['answer'])
            
            rows.append({
                'sample_id': i,
                'system': system_name,
                'question': item['question'][:100],
                'question_type': item['type'],
                'prediction': pred,
                'gold_answer': item['answer'],
                'exact_match': int(em),
                'f1_score': round(f1, 4),
                'time_seconds': round(time_taken, 4)
            })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'detailed_results_500.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved detailed: {csv_path}")
    
    # 2. Side-by-side comparison
    logger.info(f"Generating side-by-side comparison...")
    sidebyside_path = os.path.join(output_dir, 'sidebyside_comparison_500.txt')
    _save_sidebyside_comparison(all_results, test_data, sidebyside_path)
    logger.info(f"✓ Saved: {sidebyside_path}")
    
    # 3. Performance summary
    summary_path = os.path.join(output_dir, 'performance_summary_500.json')
    _save_performance_summary(all_results, test_data, summary_path)
    logger.info(f"✓ Saved: {summary_path}")
    
    # 4. System comparison table
    table_path = os.path.join(output_dir, 'system_comparison_500.txt')
    _save_comparison_table(all_results, test_data, table_path)
    logger.info(f"✓ Saved: {table_path}")
    
    # 5. Per-sample detailed log
    log_path = os.path.join(output_dir, 'detailed_log_500.txt')
    _save_detailed_log(all_results, test_data, log_path)
    logger.info(f"✓ Saved: {log_path}")

def _save_sidebyside_comparison(all_results: Dict, test_data: List[Dict], filepath: str):
    """Save side-by-side comparison of all systems"""
    
    systems = list(all_results['systems'].keys())
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 150 + "\n")
        f.write("SIDE-BY-SIDE SYSTEM COMPARISON - 500 SAMPLES\n")
        f.write("=" * 150 + "\n\n")
        
        # Summary table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 150 + "\n")
        f.write(f"{'System':<25} {'EM %':<10} {'F1 Score':<12} {'Avg Time (s)':<15} {'Errors':<10} {'Answer Rate':<12}\n")
        f.write("-" * 150 + "\n")
        
        for system_name in systems:
            predictions = all_results['systems'][system_name]['predictions']
            times = all_results['systems'][system_name]['times']
            errors = all_results['systems'][system_name].get('errors', [])
            
            em_scores = [float(exact_match(p, t['answer'])) for p, t in zip(predictions, test_data)]
            f1_scores = [f1_score(p, t['answer']) for p, t in zip(predictions, test_data)]
            
            em_pct = (sum(em_scores) / len(em_scores) * 100) if em_scores else 0
            f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            time_avg = sum(times) / len(times) if times else 0
            answer_rate = (len(predictions) - len([p for p in predictions if not p.strip()])) / len(predictions) * 100
            
            f.write(f"{system_name:<25} {em_pct:>8.1f}% {f1_avg:>10.3f} {time_avg:>13.2f}s {len(errors):>8} {answer_rate:>10.1f}%\n")
        
        f.write("\n" + "=" * 150 + "\n\n")
        
        # Per-sample comparison
        f.write("PER-SAMPLE PREDICTIONS\n")
        f.write("-" * 150 + "\n")
        
        for idx, item in enumerate(test_data[:20]):  # Show first 20 samples
            f.write(f"\n[Sample {idx+1}] {item['question'][:80]}\n")
            f.write(f"Question Type: {item['type']}\n")
            f.write(f"Gold Answer: {item['answer']}\n")
            f.write("Predictions:\n")
            
            for system_name in systems:
                pred = all_results['systems'][system_name]['predictions'][idx]
                em = exact_match(pred, item['answer'])
                f1 = f1_score(pred, item['answer'])
                time = all_results['systems'][system_name]['times'][idx]
                
                status = "✓" if em else "✗"
                f.write(f"  {status} {system_name:<25} {pred:<50} (EM:{em}, F1:{f1:.3f}, Time:{time:.3f}s)\n")
        
        f.write("\n" + "=" * 150 + "\n")

def _save_performance_summary(all_results: Dict, test_data: List[Dict], filepath: str):
    """Save JSON performance summary"""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(test_data),
        'systems': {}
    }
    
    for system_name, results in all_results['systems'].items():
        predictions = results['predictions']
        times = results['times']
        
        em_scores = [float(exact_match(p, t['answer'])) for p, t in zip(predictions, test_data)]
        f1_scores = [f1_score(p, t['answer']) for p, t in zip(predictions, test_data)]
        
        summary['systems'][system_name] = {
            'em_score': sum(em_scores) / len(em_scores) if em_scores else 0,
            'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            'avg_time': sum(times) / len(times) if times else 0,
            'total_time': sum(times),
            'errors': len(results.get('errors', [])),
            'samples': len(predictions)
        }
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)

def _save_comparison_table(all_results: Dict, test_data: List[Dict], filepath: str):
    """Save formatted comparison table"""
    
    systems = list(all_results['systems'].keys())
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE SYSTEM COMPARISON - 500 SAMPLES\n\n")
        
        # By question type
        types = {}
        for item in test_data:
            qtype = item['type']
            if qtype not in types:
                types[qtype] = []
            types[qtype].append(item)
        
        f.write("BY QUESTION TYPE:\n")
        f.write("=" * 80 + "\n\n")
        
        for qtype, items in types.items():
            f.write(f"{qtype.upper()} ({len(items)} samples):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'System':<25} {'EM':<10} {'F1':<10} {'Avg Time':<12}\n")
            f.write("-" * 80 + "\n")
            
            indices = [test_data.index(item) for item in items]
            
            for system_name in systems:
                predictions = [all_results['systems'][system_name]['predictions'][i] for i in indices]
                times = [all_results['systems'][system_name]['times'][i] for i in indices]
                
                em_scores = [float(exact_match(p, item['answer'])) for p, item in zip(predictions, items)]
                f1_scores = [f1_score(p, item['answer']) for p, item in zip(predictions, items)]
                
                em_pct = (sum(em_scores) / len(em_scores) * 100) if em_scores else 0
                f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0
                time_avg = sum(times) / len(times) if times else 0
                
                f.write(f"{system_name:<25} {em_pct:>8.1f}% {f1_avg:>8.3f} {time_avg:>10.2f}s\n")
            
            f.write("\n")

def _save_detailed_log(all_results: Dict, test_data: List[Dict], filepath: str):
    """Save detailed per-sample log"""
    
    systems = list(all_results['systems'].keys())
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("DETAILED SAMPLE-BY-SAMPLE LOG\n")
        f.write("=" * 200 + "\n\n")
        
        for idx, item in enumerate(test_data):
            f.write(f"{'='*200}\n")
            f.write(f"SAMPLE {idx+1}\n")
            f.write(f"{'='*200}\n\n")
            
            f.write(f"Question: {item['question']}\n")
            f.write(f"Type: {item['type']}\n")
            f.write(f"Gold Answer: {item['answer']}\n\n")
            
            f.write("SYSTEM RESPONSES:\n")
            f.write("-" * 200 + "\n\n")
            
            for system_name in systems:
                pred = all_results['systems'][system_name]['predictions'][idx]
                em = exact_match(pred, item['answer'])
                f1 = f1_score(pred, item['answer'])
                time = all_results['systems'][system_name]['times'][idx]
                
                status = "✓ CORRECT" if em else "✗ INCORRECT"
                f.write(f"{system_name} [{status}]\n")
                f.write(f"  Prediction: {pred}\n")
                f.write(f"  EM Score: {float(em)}\n")
                f.write(f"  F1 Score: {f1:.4f}\n")
                f.write(f"  Time: {time:.3f}s\n\n")
            
            f.write("\n")

def main():
    """Main evaluation loop with batching"""
    
    # Configuration
    config_path = 'configs/experiment_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Settings
    set_seed(config['experiment']['random_seed'])
    num_samples = int(os.getenv('NUM_SAMPLES', config['experiment']['num_samples']))
    batch_size = int(os.getenv('BATCH_SIZE', 50))
    output_dir = os.path.join('results', 'batch_evaluation')
    os.makedirs(output_dir, exist_ok=True)
    live_csv_path = os.path.join(output_dir, 'live_results.csv')
    # Initialize/overwrite live CSV with header
    with open(live_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'batch',
            'global_index',
            'system',
            'question',
            'gold_answer',
            'prediction',
            'exact_match',
            'f1_score',
            'time_seconds'
        ])
    
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH EVALUATION - AGENTIC SELF-RAG")
    logger.info(f"{'='*80}")
    logger.info(f"Total Samples: {num_samples}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Batches: {(num_samples + batch_size - 1) // batch_size}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Live per-sample CSV: {live_csv_path}")
    logger.info(f"{'='*80}\n")
    
    # Load data
    logger.info("Loading test data...")
    test_data = load_hotpotqa(
        split=config['dataset']['split'],
        num_samples=num_samples,
        cache_dir=config['dataset']['cache_dir'],
        random_seed=config['experiment']['random_seed']
    )
    logger.info(f"Loaded {len(test_data)} samples")
    
    # Initialize systems
    logger.info("\nInitializing systems...")
    all_contexts = [item['context']['sentences'] for item in test_data]
    
    systems = {}
    
    logger.info("1. Vanilla RAG...")
    vanilla_rag = VanillaRAG()
    vanilla_rag.create_vectorstore(all_contexts)
    systems['Vanilla RAG'] = vanilla_rag
    
    logger.info("2. Simplified Self-RAG...")
    simplified_self_rag = SimplifiedSelfRAG(vanilla_rag)
    systems['Simplified Self-RAG'] = simplified_self_rag
    
    logger.info("3. Published Self-RAG...")
    published_self_rag = PublishedSelfRAG()
    published_self_rag.rag.create_vectorstore(all_contexts)
    systems['Published Self-RAG'] = published_self_rag
    
    logger.info("4. Agentic Self-RAG...")
    agentic_self_rag = AgenticSelfRAG()
    agentic_self_rag.setup_vectorstore(all_contexts)
    systems['Agentic Self-RAG'] = agentic_self_rag
    
    logger.info("5. Ultra Agentic RAG...")
    ultra_agentic_rag = UltraAgenticRAG()
    ultra_agentic_rag.setup_vectorstore(all_contexts)
    systems['Ultra Agentic RAG'] = ultra_agentic_rag
    
    logger.info("\nSystems initialized!\n")
    
    # Batch processing
    evaluator = BatchEvaluator(batch_size=batch_size, live_csv_path=live_csv_path)
    all_results = {
        'systems': {name: {'predictions': [], 'times': [], 'errors': []} for name in systems.keys()},
        'batches': []
    }
    
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_data))
        
        batch_results = evaluator.process_batch(systems, test_data, batch_start, batch_end)
        
        # Accumulate results
        for system_name in systems.keys():
            all_results['systems'][system_name]['predictions'].extend(
                batch_results[system_name]['predictions']
            )
            all_results['systems'][system_name]['times'].extend(
                batch_results[system_name]['times']
            )
            all_results['systems'][system_name]['errors'].extend(
                batch_results[system_name]['errors']
            )
        
        all_results['batches'].append({
            'start': batch_start,
            'end': batch_end,
            'size': batch_end - batch_start
        })
        
        # Progress indicator
        progress_pct = ((batch_idx + 1) / num_batches) * 100
        logger.info(f"\n📊 Overall Progress: {progress_pct:.1f}% ({batch_end}/{len(test_data)} samples)\n")
    
    # Save comprehensive results
    logger.info(f"\n{'='*80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*80}\n")
    save_comprehensive_results(all_results, test_data, output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ BATCH EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
