# src/agents/retrieval_critic_agent.py
"""
Retrieval Critic Agent with quantitative IR metrics
Improvement: Uses embeddings + coverage analysis (not just LLM heuristics)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from dotenv import load_dotenv
import json
import re
import logging
import torch

load_dotenv()
logger = logging.getLogger(__name__)

class RetrievalCriticAgent:
    """
    Evaluates retrieval quality using quantitative metrics + LLM judgment
    
    Metrics:
    - Semantic similarity (query-passage)
    - Coverage (how many passages are relevant)
    - Diversity (are passages redundant?)
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        # Lazy load embedding model
        self._embed_model = None
        
        # FIXED PROMPT - No nested JSON in template variables
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a retrieval quality expert. Evaluate retrieved passages and respond ONLY with valid JSON."),
            ("user", """Question: {question}

Retrieved Passages:
{passages}

Evaluate the passages and respond with ONLY this JSON (no markdown):
{{"overall_relevance": "high", "sufficiency": "sufficient", "contradictions_found": false, "recommendation": "proceed"}}

Your response:""")
        ])
    
    def critique(self, question: str, passages: List[str]) -> Dict:
        """
        Critique retrieval quality with quantitative + qualitative analysis
        
        Args:
            question: Query string
            passages: Retrieved passages
            
        Returns:
            Critique with metrics and recommendation
        """
        # Compute quantitative metrics
        metrics = self._compute_ir_metrics(question, passages)
        
        # Try LLM-based critique (optional enhancement)
        try:
            llm_critique = self._llm_critique(question, passages)
        except Exception as e:
            logger.warning(f"LLM critique failed: {e}")
            llm_critique = {}
        
        # Combine signals
        combined = self._combine_critiques(metrics, llm_critique)
        
        logger.debug(f"Retrieval critique: {combined['overall_relevance']} relevance, {combined['sufficiency']}")
        
        return combined
    
    def _compute_ir_metrics(self, question: str, passages: List[str]) -> Dict:
        """Compute quantitative IR metrics"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer for retrieval critique")
        
        from sentence_transformers import util
        
        # Encode query and passages
        q_emb = self._embed_model.encode(question, convert_to_tensor=True)
        p_embs = self._embed_model.encode(passages, convert_to_tensor=True)
        
        # Relevance: cosine similarity to query
        similarities = util.cos_sim(q_emb, p_embs)[0]
        relevance_scores = [float(sim) for sim in similarities]
        
        avg_relevance = float(similarities.mean())
        max_relevance = float(similarities.max())
        
        # Coverage: fraction of highly relevant passages
        highly_relevant = sum(1 for s in similarities if s > 0.5)
        coverage = highly_relevant / len(passages) if passages else 0.0
        
        # Diversity: measure redundancy
        if len(passages) > 1:
            p_sims = util.cos_sim(p_embs, p_embs)
            triu_indices = torch.triu_indices(len(passages), len(passages), offset=1)
            avg_pairwise = float(p_sims[triu_indices[0], triu_indices[1]].mean())
            diversity = 1.0 - avg_pairwise
        else:
            diversity = 1.0
        
        return {
            'relevance_scores': relevance_scores,
            'avg_relevance': avg_relevance,
            'max_relevance': max_relevance,
            'coverage': coverage,
            'diversity': diversity,
            'num_passages': len(passages)
        }
    
    def _llm_critique(self, question: str, passages: List[str]) -> Dict:
        """LLM-based critique (backup/enhancement)"""
        passages_text = "\n\n".join([
            f"Passage {i+1}: {p[:150]}..." 
            for i, p in enumerate(passages[:5])
        ])
        
        chain = self.prompt | self.llm
        response = chain.invoke({
            "question": question,
            "passages": passages_text
        })
        
        content = response.content.strip()
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        return json.loads(content)
    
    def _combine_critiques(self, metrics: Dict, llm_critique: Dict) -> Dict:
        """Combine quantitative metrics with qualitative LLM judgment"""
        # Use metrics as primary signal
        if metrics['avg_relevance'] > 0.6 and metrics['coverage'] > 0.4:
            overall_relevance = "high"
            sufficiency = "sufficient"
        elif metrics['avg_relevance'] > 0.4:
            overall_relevance = "medium"
            sufficiency = "partial"
        else:
            overall_relevance = "low"
            sufficiency = "insufficient"
        
        return {
            'relevance_scores': metrics['relevance_scores'],
            'overall_relevance': overall_relevance,
            'sufficiency': sufficiency,
            'avg_relevance': metrics['avg_relevance'],
            'max_relevance': metrics['max_relevance'],
            'coverage': metrics['coverage'],
            'diversity': metrics['diversity'],
            'contradictions_found': llm_critique.get('contradictions_found', False),
            'recommendation': 'proceed' if sufficiency != 'insufficient' else 'retrieve_more',
            'missing_information': llm_critique.get('missing_information', 'none')
        }
