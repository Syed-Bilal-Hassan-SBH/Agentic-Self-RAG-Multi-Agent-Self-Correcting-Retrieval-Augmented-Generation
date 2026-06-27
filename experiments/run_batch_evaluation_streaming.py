#!/usr/bin/env python3
# experiments/run_batch_evaluation_streaming.py
"""
Streaming batch evaluation - appends results progressively as samples are processed
Creates real-time side-by-side CSV and log files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vanilla_rag import VanillaRAG
from src.self_rag import SimplifiedSelfRAG
from src.agentic_self_rag import AgenticSelfRAG
from src.ultra_agentic_rag import UltraAgenticRAG
from src.baselines.published_self_rag import PublishedSelfRAG
from src.evaluation.metrics import exact_match, f1_score
from src.utils.repro import set_seed
from src.utils.data_utils import load_hotpotqa
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingBatchEvaluator:
    """Evaluates samples and streams results to files in real-time"""
    
    def __init__(self, output_dir='results/batch_evaluation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files with headers
        self.sidebyside_csv = self.output_dir / 'sidebyside_results_500.csv'
        self.detailed_csv = self.output_dir / 'detailed_results_500.csv'
        self.log_file = self.output_dir / 'streaming_log_500.txt'
        
        self.sample_count = 0
        self.sidebyside_headers_written = False
        self.detailed_headers_written = False
        
        # Clear existing files
        if self.sidebyside_csv.exists():
            self.sidebyside_csv.unlink()
        if self.detailed_csv.exists():
            self.detailed_csv.unlink()
        
        # Write log header
        with open(self.log_file, 'w') as f:
            f.write(f"STREAMING EVALUATION LOG - Started {datetime.now()}\n")
            f.write("=" * 100 + "\n\n")
    
    def log_sample(self, sample_id, question, gold_answer, question_type, results_per_system):
        """Log a single sample's results to files (incremental appending)"""
        
        self.sample_count += 1
        
        # 1. Append to side-by-side CSV (one row with all systems)
        sidebyside_row = {
            'sample_id': sample_id,
            'question': question[:200],
            'gold_answer': gold_answer,
            'question_type': question_type
        }
        
        for system_name, result in results_per_system.items():
            sidebyside_row[f'{system_name}_prediction'] = result.get('prediction', '')
            sidebyside_row[f'{system_name}_em'] = result.get('exact_match', 0)
            sidebyside_row[f'{system_name}_f1'] = round(result.get('f1_score', 0.0), 4)
            sidebyside_row[f'{system_name}_time'] = round(result.get('time_seconds', 0.0), 4)
        
        # Append to CSV file (write header only on first sample)
        df_sidebyside = pd.DataFrame([sidebyside_row])
        df_sidebyside.to_csv(
            self.sidebyside_csv, 
            mode='a', 
            header=not self.sidebyside_headers_written,
            index=False
        )
        self.sidebyside_headers_written = True
        
        # 2. Append to detailed CSV (one row per system)
        detailed_rows = []
        for system_name, result in results_per_system.items():
            detailed_row = {
                'sample_id': sample_id,
                'system': system_name,
                'question': question[:100],
                'question_type': question_type,
                'prediction': result.get('prediction', ''),
                'gold_answer': gold_answer,
                'exact_match': result.get('exact_match', 0),
                'f1_score': round(result.get('f1_score', 0.0), 4),
                'time_seconds': round(result.get('time_seconds', 0.0), 4)
            }
            detailed_rows.append(detailed_row)
        
        df_detailed = pd.DataFrame(detailed_rows)
        df_detailed.to_csv(
            self.detailed_csv,
            mode='a',
            header=not self.detailed_headers_written,
            index=False
        )
        self.detailed_headers_written = True
        
        # 3. Append to streaming log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[Sample {sample_id + 1}] {question[:80]}\n")
            f.write(f"  Type: {question_type}\n")
            f.write(f"  Gold Answer: {gold_answer}\n")
            f.write("  Results:\n")
            
            for system_name, result in results_per_system.items():
                em_status = "✓" if result.get('exact_match') else "✗"
                f.write(f"    [{em_status}] {system_name:<20} ")
                f.write(f"Pred: {result.get('prediction', '')[:50]:<50} ")
                f.write(f"F1: {result.get('f1_score', 0):.3f} ")
                f.write(f"Time: {result.get('time_seconds', 0):.3f}s\n")
    
    def finalize_results(self, all_results, test_data):
        """Generate final summary after all samples processed"""
        
        logger.info("\n" + "=" * 100)
        logger.info("FINALIZING RESULTS")
        logger.info("=" * 100)
        
        # Generate summary statistics
        systems = list(all_results['systems'].keys())
        summary_data = {}
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n\n" + "=" * 100 + "\n")
            f.write("FINAL SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'System':<25} {'EM %':<10} {'F1 Score':<12} {'Avg Time (s)':<15} {'Total Samples':<15}\n")
            f.write("-" * 100 + "\n")
            
            for system_name in systems:
                predictions = all_results['systems'][system_name]['predictions']
                times = all_results['systems'][system_name]['times']
                
                em_scores = [float(exact_match(p, t['answer'])) for p, t in zip(predictions, test_data)]
                f1_scores = [f1_score(p, t['answer']) for p, t in zip(predictions, test_data)]
                
                em_pct = (sum(em_scores) / len(em_scores) * 100) if em_scores else 0
                f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0
                time_avg = sum(times) / len(times) if times else 0
                
                f.write(f"{system_name:<25} {em_pct:>8.1f}% {f1_avg:>10.3f} {time_avg:>13.2f}s {len(predictions):>13}\n")
                
                summary_data[system_name] = {
                    'exact_match': round(em_pct, 2),
                    'f1_score': round(f1_avg, 3),
                    'avg_time': round(time_avg, 3),
                    'total_samples': len(predictions)
                }
            
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Finalized: {datetime.now()}\n")
        
        # Save summary as JSON
        summary_path = self.output_dir / 'streaming_summary_500.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(test_data),
                'systems': summary_data
            }, f, indent=2)
        
        logger.info(f"✓ Side-by-side CSV: {self.sidebyside_csv}")
        logger.info(f"✓ Detailed CSV: {self.detailed_csv}")
        logger.info(f"✓ Streaming log: {self.log_file}")
        logger.info(f"✓ Summary JSON: {summary_path}")


def main():
    """Main evaluation loop"""
    
    # Configuration
    set_seed(42)
    num_samples = int(os.environ.get('NUM_SAMPLES', 500))
    batch_size = int(os.environ.get('BATCH_SIZE', 50))
    
    logger.info(f"\n{'='*100}")
    logger.info(f"STREAMING BATCH EVALUATION")
    logger.info(f"{'='*100}")
    logger.info(f"Samples: {num_samples} | Batch Size: {batch_size} | Batches: {num_samples // batch_size}")
    
    # Initialize systems
    logger.info("\nInitializing RAG systems...")
    
    # Initialize Vanilla RAG first (needed by others)
    vanilla_rag = VanillaRAG()
    
    systems = {
        'Vanilla RAG': vanilla_rag,
        'Simplified Self-RAG': SimplifiedSelfRAG(vanilla_rag),
        'Published Self-RAG': PublishedSelfRAG(vanilla_rag),
        'Agentic Self-RAG': AgenticSelfRAG(),
        'Ultra Agentic RAG': UltraAgenticRAG()
    }
    
    # Load data
    logger.info("Loading HotpotQA data...")
    test_data = load_hotpotqa('validation', num_samples)
    logger.info(f"Loaded {len(test_data)} samples")
    
    # Initialize streaming evaluator
    evaluator = StreamingBatchEvaluator()
    
    # Initialize storage for batch results
    all_results = {
        'systems': {
            system_name: {
                'predictions': [],
                'times': [],
                'errors': []
            } for system_name in systems.keys()
        }
    }
    
    # Process samples in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_data = test_data[batch_start:batch_end]
        
        logger.info(f"\n[Batch {batch_idx+1}/{num_batches}] Processing samples {batch_start+1}-{batch_end}")
        logger.info("-" * 100)
        
        for sample_idx, item in enumerate(tqdm(batch_data, desc="  Processing", leave=False)):
            global_sample_id = batch_start + sample_idx
            results_per_system = {}
            
            for system_name, system in systems.items():
                try:
                    start_time = time.time()
                    
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
                    elif system_name == 'Agentic Self-RAG':
                        result = system.answer(item['question'])
                        pred = result['answer']
                    elif system_name == 'Ultra Agentic RAG':
                        result = system.answer(item['question'])
                        pred = result['answer']
                    else:
                        pred = ''
                    
                    elapsed = time.time() - start_time
                    
                    # Compute metrics
                    em = exact_match(pred, item['answer'])
                    f1 = f1_score(pred, item['answer'])
                    
                    results_per_system[system_name] = {
                        'prediction': pred,
                        'exact_match': int(em),
                        'f1_score': f1,
                        'time_seconds': elapsed
                    }
                    
                    # Store for batch results
                    all_results['systems'][system_name]['predictions'].append(pred)
                    all_results['systems'][system_name]['times'].append(elapsed)
                    
                except Exception as e:
                    logger.error(f"Error with {system_name}: {e}")
                    results_per_system[system_name] = {
                        'prediction': f'ERROR: {str(e)[:50]}',
                        'exact_match': 0,
                        'f1_score': 0.0,
                        'time_seconds': 0.0
                    }
                    all_results['systems'][system_name]['predictions'].append('')
                    all_results['systems'][system_name]['times'].append(0)
                    all_results['systems'][system_name]['errors'].append(str(e))
            
            # Log this sample to CSV and log files (streaming)
            evaluator.log_sample(
                global_sample_id,
                item['question'],
                item['answer'],
                item['type'],
                results_per_system
            )
        
        # Batch summary
        logger.info(f"✓ Batch {batch_idx+1} complete")
    
    # Finalize and generate summary
    evaluator.finalize_results(all_results, test_data)
    
    logger.info(f"\n{'='*100}")
    logger.info("✅ STREAMING EVALUATION COMPLETE")
    logger.info(f"{'='*100}")
    logger.info(f"\nResults saved to: {evaluator.output_dir}/")
    logger.info(f"  • sidebyside_results_500.csv - All systems side-by-side")
    logger.info(f"  • detailed_results_500.csv - Per-system breakdown")
    logger.info(f"  • streaming_log_500.txt - Real-time sample logs")
    logger.info(f"  • streaming_summary_500.json - Final summary\n")


if __name__ == '__main__':
    main()
