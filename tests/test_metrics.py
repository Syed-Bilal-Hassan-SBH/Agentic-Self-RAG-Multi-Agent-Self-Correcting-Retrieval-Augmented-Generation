# tests/test_metrics.py
"""
Unit tests for evaluation metrics
Tests edge cases and correctness of EM/F1 implementations
"""

import pytest
from src.evaluation.metrics import (
    normalize_answer,
    exact_match,
    f1_score,
    compute_metrics
)

class TestNormalization:
    """Test answer normalization"""
    
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"
    
    def test_remove_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"
    
    def test_remove_articles(self):
        assert normalize_answer("the cat and a dog") == "cat and dog"
    
    def test_whitespace_fix(self):
        assert normalize_answer("hello  world   test") == "hello world test"
    
    def test_combined(self):
        input_text = "The quick, brown fox!"
        expected = "quick brown fox"
        assert normalize_answer(input_text) == expected


class TestExactMatch:
    """Test exact match metric"""
    
    def test_identical_strings(self):
        assert exact_match("Paris", "Paris") == True
    
    def test_case_insensitive(self):
        assert exact_match("Paris", "paris") == True
    
    def test_punctuation_ignored(self):
        assert exact_match("Paris.", "Paris") == True
    
    def test_articles_ignored(self):
        assert exact_match("the Paris", "Paris") == True
    
    def test_different_strings(self):
        assert exact_match("Paris", "London") == False
    
    def test_partial_match(self):
        assert exact_match("Paris France", "Paris") == False
    
    def test_empty_strings(self):
        assert exact_match("", "") == True
        assert exact_match("Paris", "") == False


class TestF1Score:
    """Test F1 score metric"""
    
    def test_identical_strings(self):
        assert f1_score("Paris", "Paris") == 1.0
    
    def test_no_overlap(self):
        assert f1_score("Paris", "London") == 0.0
    
    def test_partial_overlap(self):
        pred = "Paris France"
        gold = "Paris Germany"
        f1 = f1_score(pred, gold)
        assert 0.0 < f1 < 1.0  # Should have partial overlap
    
    def test_subset(self):
        pred = "Paris"
        gold = "Paris France"
        f1 = f1_score(pred, gold)
        # Precision = 1.0 (all pred tokens in gold)
        # Recall = 0.5 (only 1 of 2 gold tokens)
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.667
        assert abs(f1 - 0.667) < 0.01
    
    def test_superset(self):
        pred = "Paris France"
        gold = "Paris"
        f1 = f1_score(pred, gold)
        # Precision = 0.5, Recall = 1.0, F1 = 0.667
        assert abs(f1 - 0.667) < 0.01
    
    def test_empty_prediction(self):
        assert f1_score("", "Paris") == 0.0
    
    def test_empty_gold(self):
        assert f1_score("Paris", "") == 0.0
    
    def test_both_empty(self):
        assert f1_score("", "") == 1.0


class TestComputeMetrics:
    """Test aggregate metrics computation"""
    
    def test_perfect_predictions(self):
        predictions = ["Paris", "London", "Berlin"]
        golds = ["Paris", "London", "Berlin"]
        
        metrics = compute_metrics(predictions, golds)
        
        assert metrics['EM'] == 1.0
        assert metrics['F1'] == 1.0
        assert metrics['num_samples'] == 3
    
    def test_no_correct_predictions(self):
        predictions = ["Paris", "London", "Berlin"]
        golds = ["Rome", "Madrid", "Vienna"]
        
        metrics = compute_metrics(predictions, golds)
        
        assert metrics['EM'] == 0.0
        assert metrics['F1'] < 0.1  # Should be very low
    
    def test_mixed_predictions(self):
        predictions = ["Paris", "London", "Berlin"]
        golds = ["Paris", "Madrid", "Berlin"]
        
        metrics = compute_metrics(predictions, golds)
        
        assert metrics['EM'] == 2/3  # 2 out of 3 exact matches
        assert 0.6 < metrics['F1'] < 1.0  # Partial credit
    
    def test_length_mismatch_raises_error(self):
        predictions = ["Paris", "London"]
        golds = ["Paris"]
        
        with pytest.raises(AssertionError):
            compute_metrics(predictions, golds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])