# src/config.py
"""Central configuration for reproducibility"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    """LLM configuration"""
    name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GROQ_API_KEY")

@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_k: int = 5
    multi_hop_k: int = 10  # More passages for multi-hop
    similarity_threshold: float = 0.5
    diversity_weight: float = 0.3

@dataclass
class AgenticConfig:
    """Agentic RAG configuration"""
    max_iterations: int = 4
    confidence_threshold_high: float = 0.85  # Early stop
    confidence_threshold_low: float = 0.50   # Plateau detection
    use_multi_hop: bool = True
    use_adaptive_iteration: bool = True
    
@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    num_samples: int = 500
    random_seed: int = 42
    stratify_by_type: bool = True
    save_predictions: bool = True
    
@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "models"
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

# Global config instance
config = Config()