# tests/test_reproducibility.py
"""
Test deterministic behavior with fixed seeds
Ensures reproducible results across runs
"""

import pytest
from src.utils.repro import set_seed
from src.evaluation.metrics import f1_score
import numpy as np
import random

class TestReproducibility:
    """Test deterministic behavior"""
    
    def test_numpy_seed(self):
        """Test NumPy random seed"""
        set_seed(42)
        vals1 = [np.random.rand() for _ in range(10)]
        
        set_seed(42)
        vals2 = [np.random.rand() for _ in range(10)]
        
        assert vals1 == vals2
    
    def test_python_random_seed(self):
        """Test Python random seed"""
        set_seed(42)
        vals1 = [random.random() for _ in range(10)]
        
        set_seed(42)
        vals2 = [random.random() for _ in range(10)]
        
        assert vals1 == vals2
    
    def test_pytorch_seed(self):
        """Test PyTorch seed if available"""
        try:
            import torch
            
            set_seed(42)
            vals1 = [torch.rand(1).item() for _ in range(10)]
            
            set_seed(42)
            vals2 = [torch.rand(1).item() for _ in range(10)]
            
            assert vals1 == vals2
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_metric_determinism(self):
        """Test that metrics are deterministic"""
        pred = "The capital of France is Paris"
        gold = "Paris is the capital of France"
        
        # Run multiple times
        scores = [f1_score(pred, gold) for _ in range(5)]
        
        # All should be identical
        assert len(set(scores)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])