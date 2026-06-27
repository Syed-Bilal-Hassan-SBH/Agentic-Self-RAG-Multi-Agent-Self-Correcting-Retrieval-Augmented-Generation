# experiments/run_ablations.py
"""
Ablation study: Measure contribution of each component
Tests 5 variants to prove novelty and understand system behavior
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vanilla_rag import VanillaRAG
from src.agentic_self_rag import AgenticSelfRAG
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical_tests import compare_systems, print_comparison
from src.utils.repro import set_seed
from src.utils.data_utils import load_hotpotqa
import json
import logging
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AblatedAgenticSelfRAG(AgenticSelfRAG):
    """Agentic Self-RAG with component ablations"""
    
    def __init__(self, ablate_component: str = None, **kwargs):
        """
        Args:
            ablate_component: Component to remove
                - 'query_analyzer': Skip multi-hop detection
                - 'retrieval_critic': Skip retrieval evaluation
                - 'answer_verifier': Skip answer verification
                - 'multi_hop': Disable multi-hop reasoning
                - 'adaptive_iteration': Use fixed iteration
        """
        super().__init__(**kwargs)
        self.ablate_component = ablate_component
        logger.info(f"Ablation: {ablate_component if ablate_component else 'None (full system)'}")
    
    def _analyze_query(self, state):
        """Optionally skip query analysis"""
        if self.ablate_component == 'query_analyzer':
            # Skip analysis, assume single-hop
            state["query_analysis"] = {
                "is_multi_hop": False,
                "complexity": "unknown",
                "sub_questions": [],
                "reasoning": "Analysis skipped (ablation)"
            }
            state["use_multi_hop"] = False
            state["sub_questions"] = []
            return state
        else:
            return super()._analyze_query(state)
    
    def _critique_retrieval(self, state):
        """Optionally skip retrieval critique"""
        if self.ablate_component == 'retrieval_critic':
            # Skip critique, assume sufficient
            state["retrieval_critique"] = {
                "overall_relevance": "unknown",
                "sufficiency": "sufficient",
                "avg_relevance": 0.5,
                "coverage": 0.5,
                "recommendation": "proceed"
            }
            return state
        else:
            return super()._critique_retrieval(state)
    
    def _verify_answer(self, state):
        """Optionally skip answer verification"""
        if self.ablate_component == 'answer_verifier':
            # Skip verification, accept answer
            state["verification"] = {
                "verdict": "SUPPORTED",
                "confidence": 0.5,
                "is_supported": True
            }
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            # Always stop immediately
            state["final_answer"] = state["answer"]
            state["final_verdict"] = "SUPPORTED"
            state["final_confidence"] = 0.5
            
            return state
        else:
            return super()._verify_answer(state)


def run_ablation_study(config_path: str = 'configs/experiment_config.yaml'):
    """Run comprehensive ablation study"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['experiment']['random_seed'])
    
    num_samples = config['experiment']['num_samples']
    
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION STUDY")
    logger.info(f"{'='*80}\n")
    
    # Load test data
    logger.info(f"Loading {num_samples} test samples...")
    test_data = load_hotpotqa(
        split=config['dataset']['split'],
        num_samples=num_samples,
        cache_dir=config['dataset']['cache_dir'],
        random_seed=config['experiment']['random_seed']
    )
    
    # Prepare vectorstore
    logger.info("Preparing vectorstore...")
    vanilla_rag = VanillaRAG()
    all_contexts = [item['context']['sentences'] for item in test_data]
    vanilla_rag.create_vectorstore(all_contexts)
    
    # Define ablations
    ablations = {
        'Full System': None,
        'No Query Analyzer': 'query_analyzer',
        'No Retrieval Critic': 'retrieval_critic',
        'No Answer Verifier': 'answer_verifier',
        'No Multi-Hop': 'multi_hop',
        'Fixed Iteration': 'adaptive_iteration'
    }
    
    results = {}
    
    # Run each ablation
    for ablation_name, ablate_component in ablations.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Ablation: {ablation_name}")
        logger.info(f"{'='*80}")
        
        # Initialize system
        if ablate_component == 'multi_hop':
            system = AblatedAgenticSelfRAG(use_multi_hop=False)
        elif ablate_component == 'adaptive_iteration':
            system = AblatedAgenticSelfRAG(use_adaptive_iteration=False, max_iterations=2)
        else:
            system = AblatedAgenticSelfRAG(ablate_component=ablate_component)
        
        system.setup_vectorstore(all_contexts)
        
        # Evaluate
        predictions = []
        gold_answers = []
        
        for item in tqdm(test_data, desc=ablation_name):
            try:
                result = system.answer(item['question'])
                predictions.append(result['answer'])
                gold_answers.append(item['answer'])
            except Exception as e:
                logger.error(f"Error: {e}")
                predictions.append("")
                gold_answers.append(item['answer'])
        
        # Compute metrics
        metrics = compute_metrics(predictions, gold_answers)
        
        logger.info(f"\n{ablation_name} Results:")
        logger.info(f"  EM: {metrics['EM']:.1%} ± {metrics['EM_std']:.3f}")
        logger.info(f"  F1: {metrics['F1']:.3f} ± {metrics['F1_std']:.3f}")
        
        results[ablation_name] = {
            'metrics': metrics,
            'ablate_component': ablate_component
        }
    
    # Statistical comparisons (all vs full)
    logger.info(f"\n{'='*80}")
    logger.info("STATISTICAL SIGNIFICANCE TESTS (vs Full System)")
    logger.info(f"{'='*80}")
    
    full_scores = results['Full System']['metrics']['F1_scores']
    
    comparisons = []
    for ablation_name, result in results.items():
        if ablation_name == 'Full System':
            continue
        
        ablation_scores = result['metrics']['F1_scores']
        
        comparison = compare_systems(
            full_scores, ablation_scores,
            'Full System', ablation_name
        )
        
        print_comparison(comparison)
        comparisons.append(comparison)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}\n")
    
    full_em = results['Full System']['metrics']['EM']
    full_f1 = results['Full System']['metrics']['F1']
    
    print(f"{'Configuration':<30} {'EM':<10} {'ΔEM':<12} {'F1':<10} {'ΔF1':<10}")
    print("-" * 72)
    
    for ablation_name, result in results.items():
        metrics = result['metrics']
        em_delta = metrics['EM'] - full_em
        f1_delta = metrics['F1'] - full_f1
        
        print(f"{ablation_name:<30} "
              f"{metrics['EM']:.1%}     "
              f"{em_delta:+.1%}      "
              f"{metrics['F1']:.3f}    "
              f"{f1_delta:+.3f}")
    
    print("\n" + "="*80 + "\n")
    
    # Save results
    output_dir = 'results/ablations'
    os.makedirs(output_dir, exist_ok=True)
    
    save_results = {
        ablation_name: {
            'EM': result['metrics']['EM'],
            'F1': result['metrics']['F1'],
            'EM_std': result['metrics']['EM_std'],
            'F1_std': result['metrics']['F1_std'],
            'ablate_component': result['ablate_component']
        }
        for ablation_name, result in results.items()
    }
    
    with open(f'{output_dir}/ablation_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    with open(f'{output_dir}/statistical_comparisons.json', 'w') as f:
        json.dump(comparisons, f, indent=2)
logger.info("✓ Ablation study complete!")
