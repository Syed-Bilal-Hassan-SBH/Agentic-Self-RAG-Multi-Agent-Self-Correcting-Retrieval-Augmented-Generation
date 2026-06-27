"""
Agentic Self-RAG: Main package
Multi-agent RAG system with self-evaluation and iterative refinement
"""

from src.vanilla_rag import VanillaRAG
from src.self_rag import SelfRAG, SimplifiedSelfRAG
from src.agentic_self_rag import AgenticSelfRAG
from src.config import Config

__all__ = [
    "VanillaRAG",
    "SelfRAG",
    "SimplifiedSelfRAG",
    "AgenticSelfRAG",
    "Config",
]

__version__ = "1.0.0"
