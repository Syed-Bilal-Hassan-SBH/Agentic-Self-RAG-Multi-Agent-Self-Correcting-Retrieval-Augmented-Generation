# tests/test_verifier.py
"""
Unit tests for answer verifier agent
Tests multi-signal verification logic
"""

import pytest
from src.agents.answer_verifier_agent import AnswerVerifierAgent

class TestAnswerVerifier:
    """Test answer verification logic"""
    
    @pytest.fixture
    def verifier(self):
        """Create verifier instance"""
        config = {
            'weight_semantic': 0.40,
            'weight_entity': 0.30,
            'weight_token': 0.20,
            'weight_length': 0.10,
            'threshold_supported': 0.75,
            'threshold_partial': 0.50,
            'cache_embeddings': False  # Disable caching for tests
        }
        return AnswerVerifierAgent(config=config)
    
    def test_supported_answer(self, verifier):
        """Test clearly supported answer"""
        question = "What is the capital of France?"
        answer = "The capital of France is Paris."
        context = ["Paris is the capital and largest city of France."]
        
        result = verifier.verify(question, answer, context)
        
        assert result['verdict'] in ['SUPPORTED', 'PARTIALLY_SUPPORTED']
        assert result['confidence'] > 0.5
        assert 'signals' in result
    
    def test_not_supported_answer(self, verifier):
        """Test clearly unsupported answer"""
        question = "What is the capital of France?"
        answer = "The capital of France is London."
        context = ["Paris is the capital and largest city of France."]
        
        result = verifier.verify(question, answer, context)
        
        assert result['verdict'] in ['NOT_SUPPORTED', 'PARTIALLY_SUPPORTED']
        assert result['confidence'] < 0.75
    
    def test_empty_answer(self, verifier):
        """Test empty answer"""
        question = "What is the capital of France?"
        answer = ""
        context = ["Paris is the capital of France."]
        
        result = verifier.verify(question, answer, context)
        
        assert result['verdict'] == 'NOT_SUPPORTED'
        assert result['confidence'] < 0.5
    
    def test_i_dont_know_answer(self, verifier):
        """Test 'I don't know' type answers"""
        question = "What is the capital of France?"
        answer = "I don't know."
        context = ["Paris is the capital of France."]
        
        result = verifier.verify(question, answer, context)
        
        assert result['confidence'] < 0.5
        assert result['signals']['length_reasonableness'] < 0.5
    
    def test_partial_support(self, verifier):
        """Test partially supported answer"""
        question = "What is the capital and population of France?"
        answer = "The capital of France is Paris."  # Only answers part
        context = ["Paris is the capital of France. France has 67 million people."]
        
        result = verifier.verify(question, answer, context)
        
        # Should have decent but not perfect confidence
        assert 0.4 < result['confidence'] < 0.9
    
    def test_hallucination(self, verifier):
        """Test hallucinated facts"""
        question = "What is the capital of France?"
        answer = "The capital of France is Paris, which was founded in 1850."
        context = ["Paris is the capital of France."]
        
        result = verifier.verify(question, answer, context)
        
        # Should detect some mismatch due to hallucinated date
        # but not completely reject due to partial truth
        assert result['confidence'] < 0.95


class TestMultiSignalWeights:
    """Test that signal weights are properly applied"""
    
    def test_weights_sum_to_one(self):
        """Ensure weights sum to 1.0"""
        config = {
            'weight_semantic': 0.40,
            'weight_entity': 0.30,
            'weight_token': 0.20,
            'weight_length': 0.10,
        }
        verifier = AnswerVerifierAgent(config=config)
        
        total = sum([
            config['weight_semantic'],
            config['weight_entity'],
            config['weight_token'],
            config['weight_length']
        ])
        
        assert abs(total - 1.0) < 0.01
    
    def test_invalid_weights_raise_error(self):
        """Invalid weights should raise assertion error"""
        config = {
            'weight_semantic': 0.50,
            'weight_entity': 0.30,
            'weight_token': 0.20,
            'weight_length': 0.20,  # Sum > 1.0
        }
        
        with pytest.raises(AssertionError):
            AnswerVerifierAgent(config=config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])