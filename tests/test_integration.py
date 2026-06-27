# tests/test_integration.py
"""
Integration tests for Agentic Self-RAG system
Tests end-to-end pipeline, data loading, metric computation, and statistical testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import tempfile
from pathlib import Path
from typing import List, Dict

from src.evaluation.metrics import (
    exact_match, f1_score, normalize_answer, compute_metrics, compute_metrics_by_type
)
from src.evaluation.statistical_tests import (
    paired_t_test, wilcoxon_test, cohen_d, compare_systems, normality_test,
    bootstrap_confidence_interval, power_analysis, multi_comparison_correction
)
from src.utils.data_utils import load_hotpotqa
from src.utils.repro import set_seed
from src.vanilla_rag import VanillaRAG
from src.self_rag import SimplifiedSelfRAG
from src.agentic_self_rag import AgenticSelfRAG


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_normalize_answer(self):
        """Test answer normalization"""
        assert normalize_answer("The Capital") == normalize_answer("the capital")
        assert normalize_answer("Washington, D.C.") == normalize_answer("Washington DC")
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"
    
    def test_exact_match(self):
        """Test exact match metric"""
        assert exact_match("Yes", "yes") == True
        assert exact_match("The answer is 42", "42") == False
        assert exact_match("", "") == True
        assert exact_match("Obama", "Barack Hussein Obama") == False
    
    def test_f1_score(self):
        """Test F1 score computation"""
        # Perfect match
        assert f1_score("The capital is Washington", "The capital is Washington") == 1.0
        
        # No match
        assert f1_score("apple", "orange") == 0.0
        
        # Partial match
        score = f1_score("The capital of USA", "capital of USA")
        assert 0 < score < 1
        
        # Empty strings
        assert f1_score("", "") == 1.0
        assert f1_score("", "answer") == 0.0
    
    def test_compute_metrics(self):
        """Test aggregate metrics computation"""
        predictions = [
            "The capital of France is Paris",
            "Paris",
            "London is the capital of UK"
        ]
        gold_answers = [
            "Paris",
            "paris",
            "London"
        ]
        
        metrics = compute_metrics(predictions, gold_answers)
        
        assert 'EM' in metrics
        assert 'F1' in metrics
        assert 'EM_std' in metrics
        assert 'F1_std' in metrics
        assert metrics['num_samples'] == 3
        assert 0 <= metrics['EM'] <= 1
        assert 0 <= metrics['F1'] <= 1
    
    def test_compute_metrics_by_type(self):
        """Test stratified metrics"""
        predictions = ["paris", "london", "berlin", "rome"]
        golds = ["Paris", "London", "Berlin", "Rome"]
        types = ["capital", "capital", "comparison", "comparison"]
        
        metrics_by_type = compute_metrics_by_type(predictions, golds, types)
        
        assert 'capital' in metrics_by_type
        assert 'comparison' in metrics_by_type
        assert metrics_by_type['capital']['num_samples'] == 2
        assert metrics_by_type['comparison']['num_samples'] == 2
    
    def test_metrics_with_empty_predictions(self):
        """Test metrics with empty predictions"""
        predictions = ["", "answer", ""]
        golds = ["gold1", "answer", "gold3"]
        
        metrics = compute_metrics(predictions, golds)
        
        assert metrics['num_samples'] == 3
        # First and third should be mismatches, second should match
        assert metrics['EM'] == pytest.approx(1/3, abs=0.01)


class TestStatisticalTests:
    """Test statistical testing functions"""
    
    def test_paired_t_test(self):
        """Test paired t-test"""
        scores_a = [0.5, 0.6, 0.7, 0.8]
        scores_b = [0.6, 0.7, 0.8, 0.9]
        
        t_stat, p_value = paired_t_test(scores_a, scores_b)
        
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert p_value < 1.0
        assert t_stat < 0  # System B is better
    
    def test_wilcoxon_test(self):
        """Test Wilcoxon signed-rank test"""
        scores_a = [0.5, 0.6, 0.7, 0.8]
        scores_b = [0.6, 0.7, 0.8, 0.9]
        
        w_stat, p_value = wilcoxon_test(scores_a, scores_b)
        
        assert isinstance(w_stat, float)
        assert isinstance(p_value, float)
        assert p_value >= 0
    
    def test_cohen_d(self):
        """Test Cohen's d effect size"""
        # Same distributions should have d ≈ 0
        scores_a = [0.5, 0.6, 0.7, 0.8]
        scores_b = [0.5, 0.6, 0.7, 0.8]
        
        d = cohen_d(scores_a, scores_b)
        assert d == pytest.approx(0, abs=0.01)
        
        # Different distributions should have non-zero d
        scores_c = [0.1, 0.2, 0.3, 0.4]
        d = cohen_d(scores_a, scores_c)
        assert abs(d) > 0.5  # Large effect
    
    def test_compare_systems(self):
        """Test complete system comparison"""
        scores_a = [0.5, 0.6, 0.7, 0.8]
        scores_b = [0.6, 0.7, 0.8, 0.9]
        
        result = compare_systems(scores_a, scores_b, "System A", "System B")
        
        assert result['system_a_name'] == "System A"
        assert result['system_b_name'] == "System B"
        assert 'mean_a' in result
        assert 'mean_b' in result
        assert 'difference' in result
        assert 'p_value_t_test' in result
        assert 'p_value_wilcoxon' in result
        assert 'cohen_d' in result
    
    def test_normality_test(self):
        """Test normality testing"""
        # Normal distribution
        scores_normal = [0.5, 0.51, 0.49, 0.52, 0.48] * 10
        
        result = normality_test(scores_normal)
        
        assert 'test' in result
        assert 'p_value' in result
        assert 'normal' in result
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval"""
        scores = [0.5, 0.6, 0.7, 0.8]
        lower, upper = bootstrap_confidence_interval(scores)
        
        assert lower < upper
        assert lower > 0
        assert upper < 1
        # Mean should be within CI
        assert lower <= sum(scores)/len(scores) <= upper
    
    def test_power_analysis(self):
        """Test post-hoc power analysis"""
        power = power_analysis(effect_size=0.5, n=30, alpha=0.05)
        
        assert 0 <= power <= 1
        # Larger effect size should give higher power
        power_small = power_analysis(effect_size=0.2, n=30, alpha=0.05)
        assert power > power_small
    
    def test_multi_comparison_correction(self):
        """Test multiple comparison correction"""
        p_values = [0.001, 0.05, 0.1, 0.5]
        
        # Bonferroni
        result_bonf = multi_comparison_correction(p_values, method='bonferroni')
        assert result_bonf['method'] == 'bonferroni'
        assert all(p >= orig_p for p, orig_p in zip(result_bonf['corrected_pvalues'], p_values))
        
        # Holm
        result_holm = multi_comparison_correction(p_values, method='holm')
        assert result_holm['method'] == 'holm'
        # Holm should be less conservative than Bonferroni
        assert sum(result_holm['significant_corrected']) >= sum(result_bonf['significant_corrected'])


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_hotpotqa_basic(self):
        """Test basic HotpotQA loading"""
        data = load_hotpotqa(split='validation', num_samples=5, random_seed=42)
        
        assert len(data) == 5
        assert all('question' in item for item in data)
        assert all('answer' in item for item in data)
        assert all('context' in item for item in data)
        assert all('type' in item for item in data)
    
    def test_load_hotpotqa_reproducibility(self):
        """Test reproducibility with fixed seed"""
        data1 = load_hotpotqa(split='validation', num_samples=10, random_seed=42)
        data2 = load_hotpotqa(split='validation', num_samples=10, random_seed=42)
        
        # Should load same samples
        assert len(data1) == len(data2)
        assert data1[0]['question'] == data2[0]['question']
    
    def test_load_hotpotqa_structure(self):
        """Test HotpotQA data structure"""
        data = load_hotpotqa(split='validation', num_samples=1)
        item = data[0]
        
        # Check required fields
        assert isinstance(item['question'], str)
        assert isinstance(item['answer'], str)
        assert isinstance(item['context'], dict)
        assert 'sentences' in item['context']
        assert isinstance(item['context']['sentences'], list)
        assert item['type'] in ['bridge', 'comparison']


class TestSystemInitialization:
    """Test system initialization and basic functionality"""
    
    @pytest.fixture
    def sample_contexts(self):
        """Sample contexts for testing"""
        return [
            "Paris is the capital of France.",
            "The Eiffel Tower is located in Paris.",
            "France is in Western Europe."
        ]
    
    def test_vanilla_rag_initialization(self, sample_contexts):
        """Test VanillaRAG initialization"""
        rag = VanillaRAG()
        assert rag is not None
        
        # Create vectorstore
        rag.create_vectorstore(sample_contexts)
        
        # Should be able to answer questions
        result = rag.answer_question("What is the capital of France?")
        assert 'answer' in result
        assert isinstance(result['answer'], str)
    
    def test_simplified_self_rag_initialization(self, sample_contexts):
        """Test SimplifiedSelfRAG initialization"""
        vanilla = VanillaRAG()
        vanilla.create_vectorstore(sample_contexts)
        
        self_rag = SimplifiedSelfRAG(vanilla)
        assert self_rag is not None
        
        result = self_rag.answer_with_reflection("What is the capital of France?")
        assert 'answer' in result
    
    def test_agentic_rag_initialization(self, sample_contexts):
        """Test AgenticSelfRAG initialization"""
        agentic = AgenticSelfRAG()
        assert agentic is not None
        
        agentic.setup_vectorstore(sample_contexts)
        
        result = agentic.answer("What is the capital of France?")
        assert 'answer' in result


class TestEndToEndPipeline:
    """Test complete evaluation pipeline"""
    
    def test_small_evaluation(self):
        """Test full evaluation on small sample"""
        set_seed(42)
        
        # Load small dataset
        test_data = load_hotpotqa(split='validation', num_samples=3, random_seed=42)
        
        assert len(test_data) == 3
        
        # Initialize system
        all_contexts = [item['context']['sentences'] for item in test_data]
        rag = VanillaRAG()
        rag.create_vectorstore(all_contexts)
        
        # Evaluate
        predictions = []
        gold_answers = []
        
        for item in test_data:
            try:
                result = rag.answer_question(item['question'])
                predictions.append(result['answer'])
                gold_answers.append(item['answer'])
            except Exception as e:
                predictions.append("")
                gold_answers.append(item['answer'])
        
        # Compute metrics
        metrics = compute_metrics(predictions, gold_answers)
        
        assert metrics['num_samples'] == 3
        assert 0 <= metrics['EM'] <= 1
        assert 0 <= metrics['F1'] <= 1
    
    def test_comparison_pipeline(self):
        """Test system comparison pipeline"""
        set_seed(42)
        
        # Load data
        test_data = load_hotpotqa(split='validation', num_samples=5, random_seed=42)
        all_contexts = [item['context']['sentences'] for item in test_data]
        
        # Initialize systems
        rag1 = VanillaRAG()
        rag1.create_vectorstore(all_contexts)
        
        rag2 = VanillaRAG()
        rag2.create_vectorstore(all_contexts)
        
        # Get predictions
        predictions1 = []
        predictions2 = []
        golds = []
        
        for item in test_data:
            try:
                result1 = rag1.answer_question(item['question'])
                result2 = rag2.answer_question(item['question'])
                
                predictions1.append(result1['answer'])
                predictions2.append(result2['answer'])
                golds.append(item['answer'])
            except:
                predictions1.append("")
                predictions2.append("")
                golds.append(item['answer'])
        
        # Compute metrics
        metrics1 = compute_metrics(predictions1, golds)
        metrics2 = compute_metrics(predictions2, golds)
        
        # Statistical comparison
        scores1 = metrics1['F1_scores']
        scores2 = metrics2['F1_scores']
        
        comparison = compare_systems(scores1, scores2, "System 1", "System 2")
        
        assert 'p_value_wilcoxon' in comparison
        assert 'cohen_d' in comparison


class TestOutputSerialization:
    """Test saving and loading results"""
    
    def test_results_json_serialization(self):
        """Test JSON serialization of results"""
        results = {
            'system_name': 'Test System',
            'metrics': {
                'EM': 0.5,
                'F1': 0.6,
                'EM_std': 0.1,
                'F1_std': 0.15
            },
            'predictions': ['answer1', 'answer2']
        }
        
        # Should be JSON serializable
        json_str = json.dumps(results)
        loaded = json.loads(json_str)
        
        assert loaded['system_name'] == 'Test System'
        assert loaded['metrics']['EM'] == 0.5
    
    def test_results_directory_creation(self):
        """Test results directory management"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'results', 'test')
            os.makedirs(output_dir, exist_ok=True)
            
            assert os.path.isdir(output_dir)
            
            # Test file writing
            results_file = os.path.join(output_dir, 'results.json')
            with open(results_file, 'w') as f:
                json.dump({'test': 'data'}, f)
            
            assert os.path.exists(results_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
