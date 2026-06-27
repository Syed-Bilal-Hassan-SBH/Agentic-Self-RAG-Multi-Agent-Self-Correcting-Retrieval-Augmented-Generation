# experiments/quick_evaluation.py
"""
Quick evaluation on 100 samples for fast iteration
Then scale to 500 once validated
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

from src.vanilla_rag import VanillaRAG
from src.self_rag import SimplifiedSelfRAG
from src.agentic_self_rag import AgenticSelfRAG
from src.baselines.published_self_rag import PublishedSelfRAG
from src.evaluation.metrics import exact_match, f1_score
from src.utils.repro import set_seed


def quick_eval(num_samples=100):
    """Run quick evaluation"""
    
    set_seed(42)
    
    # Load data
    with open('data/hotpotqa_cache.json', 'r') as f:
        data = json.load(f)
    
    data = data[:num_samples]
    print(f"\n✓ Loaded {len(data)} samples")
    
    # Prepare contexts
    all_contexts = [' '.join(item['context']['sentences']) if isinstance(item['context'], dict)
                    else item['context'] for item in data]
    
    print("\nInitializing systems...")
    vanilla_rag = VanillaRAG()
    vanilla_rag.create_vectorstore(all_contexts)
    
    simplified_self_rag = SimplifiedSelfRAG(vanilla_rag)
    published_self_rag = PublishedSelfRAG(vanilla_rag)
    
    agentic_rag = AgenticSelfRAG()
    agentic_rag.setup_vectorstore(all_contexts)
    
    systems = {
        'Vanilla RAG': vanilla_rag,
        'Simplified Self-RAG': simplified_self_rag,
        'Published Self-RAG': published_self_rag,
        'Agentic Self-RAG': agentic_rag
    }
    
    # Quick evaluation
    results = {}
    
    for system_name, system in systems.items():
        em_scores = []
        f1_scores = []
        
        print(f"\n{system_name}:")
        for item in tqdm(data, desc=system_name, leave=False):
            try:
                if system_name == 'Vanilla RAG':
                    result = system.answer_question(item['question'])
                    pred = result['answer']
                elif system_name == 'Published Self-RAG':
                    result = system.answer_with_self_rag(item['question'])
                    pred = result['answer']
                elif system_name == 'Simplified Self-RAG':
                    result = system.answer_with_reflection(item['question'])
                    pred = result['answer']
                else:
                    result = system.answer(item['question'])
                    pred = result['answer']
                
                em = exact_match(pred, item['answer'])
                f1 = f1_score(pred, item['answer'])
                
                em_scores.append(float(em))
                f1_scores.append(f1)
                
            except Exception as e:
                em_scores.append(0.0)
                f1_scores.append(0.0)
        
        em_mean = sum(em_scores) / len(em_scores) if em_scores else 0
        f1_mean = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        results[system_name] = {
            'EM': em_mean,
            'F1': f1_mean
        }
        
        print(f"  EM: {em_mean:.1%}, F1: {f1_mean:.3f}")
    
    # Display comparison
    print("\n" + "="*70)
    print(f"QUICK EVALUATION RESULTS ({num_samples} samples)")
    print("="*70)
    
    df = pd.DataFrame(results).T
    df = df.sort_values('EM', ascending=False)
    print(df.to_string())
    
    # Save
    with open('results/main_evaluation/quick_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results/main_evaluation/quick_results.json")
    
    return results


if __name__ == '__main__':
    quick_eval(100)
