# src/utils/repro.py
"""
Reproducibility utilities for deterministic experiments
Sets all relevant random seeds for numpy, torch, random, transformers
"""

import random
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (if available)
    - Transformers (if available)
    
    Also sets deterministic flags for CUDA operations.
    
    Args:
        seed: Random seed value
    """
    logger.info(f"Setting random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info("PyTorch seeds set")
    except ImportError:
        logger.warning("PyTorch not available")
    
    # Transformers
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
        logger.info("Transformers seed set")
    except ImportError:
        logger.warning("Transformers not available")
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info(f"✓ All seeds set to {seed}")


def get_git_commit():
    """Get current git commit SHA for reproducibility"""
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit
    except Exception:
        return "unknown"


def get_environment_info():
    """Collect environment information for metadata"""
    import sys
    import platform
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'git_commit': get_git_commit(),
    }
    
    # Package versions
    try:
        import torch
        info['torch_version'] = torch.__version__
    except Exception:
        pass
    
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except Exception:
        pass
    
    try:
        import langchain
        info['langchain_version'] = langchain.__version__
    except Exception:
        pass
    
    return info