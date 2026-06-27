# src/agents/answer_verifier_agent.py
"""
Answer Verifier Agent with multi-signal verification
COMPLETE REWRITE: Robust multi-signal approach using embeddings + entities + content overlap
Fixed: All answers no longer marked "SUPPORTED" - realistic discrimination
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Optional
from dotenv import load_dotenv
import json
import re
import nltk
import logging
import numpy as np
from functools import lru_cache

load_dotenv()
logger = logging.getLogger(__name__)

class AnswerVerifierAgent:
    """
    Multi-signal answer verification with deterministic behavior
    
    Signals:
    1. Semantic similarity (40%): Sentence embeddings + cosine similarity
    2. Named entity overlap (30%): spaCy NER Jaccard similarity
    3. Content word overlap (20%): Token-level Jaccard (no stopwords)
    4. Length reasonableness (10%): Answer length in reasonable range
    
    Verdict thresholds (configurable):
    - SUPPORTED: combined_score >= 0.75
    - PARTIALLY_SUPPORTED: 0.50 <= combined_score < 0.75
    - NOT_SUPPORTED: combined_score < 0.50
    """
    
    def __init__(self, 
                 model_name: str = "llama-3.3-70b-versatile",
                 config: Optional[Dict] = None):
        """
        Initialize verifier with configurable weights and thresholds
        
        Args:
            model_name: LLM model name (fallback only)
            config: Configuration dict with weights and thresholds
        """
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        # Load configuration
        if config is None:
            config = {}
        
        self.config = {
            'weight_semantic': config.get('weight_semantic', 0.40),
            'weight_entity': config.get('weight_entity', 0.30),
            'weight_token': config.get('weight_token', 0.20),
            'weight_length': config.get('weight_length', 0.10),
            'threshold_supported': config.get('threshold_supported', 0.75),
            'threshold_partial': config.get('threshold_partial', 0.50),
            'cache_embeddings': config.get('cache_embeddings', True),
        }
        
        # Validate weights sum to 1.0
        total_weight = sum([
            self.config['weight_semantic'],
            self.config['weight_entity'],
            self.config['weight_token'],
            self.config['weight_length']
        ])
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"
        
        # Lazy load models
        self._embed_model = None
        self._nlp = None
        self._stopwords = None
        
        logger.info("Answer Verifier initialized with multi-signal approach")
    
    def _load_embedder(self):
        """Lazy load sentence transformer"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer: all-MiniLM-L6-v2")
        return self._embed_model
    
    def _load_nlp(self):
        """Lazy load spaCy model"""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, entity overlap disabled")
                self._nlp = None
        return self._nlp
    
    def _load_stopwords(self):
        """Lazy load stopwords"""
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words('english'))
            except:
                # Fallback to common stopwords
                self._stopwords = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
                }
        return self._stopwords
    
    @lru_cache(maxsize=1024)
    def _get_embedding(self, text: str):
        """Cached embedding computation"""
        model = self._load_embedder()
        return model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    
    def verify(self, question: str, answer: str, context: List[str]) -> Dict:
        """
        Verify if answer is supported by context using multi-signal approach
        
        Args:
            question: Original question
            answer: Generated answer
            context: List of context passages
            
        Returns:
            Verification result with verdict, confidence, and signal breakdown
        """
        # Truncate for efficiency
        answer_text = answer[:500] if len(answer) > 500 else answer
        context_text = "\n".join(context[:5]) if isinstance(context, list) else str(context)[:2000]
        
        # Compute multi-signal verification
        result = self._multi_signal_verify(answer_text, context_text)
        
        # Add metadata
        result['question'] = question[:100]
        result['answer_length'] = len(answer.split())
        result['context_passages'] = len(context) if isinstance(context, list) else 1
        
        logger.debug(f"Verification: {result['verdict']} (confidence={result['confidence']:.3f})")
        
        return result
    
    def _multi_signal_verify(self, answer: str, context: str) -> Dict:
        """
        Multi-signal verification implementation
        
        Combines:
        1. Semantic similarity (embeddings)
        2. Named entity overlap
        3. Content word overlap
        4. Length reasonableness
        
        Args:
            answer: Answer text
            context: Context text
            
        Returns:
            Verification result with signals and verdict
        """
        signals = {}
        
        # Signal 1: Semantic similarity (40%)
        try:
            from sentence_transformers import util
            model = self._load_embedder()
            
            if self.config['cache_embeddings']:
                answer_emb = self._get_embedding(answer)
                context_emb = self._get_embedding(context)
            else:
                answer_emb = model.encode(answer, convert_to_tensor=True)
                context_emb = model.encode(context, convert_to_tensor=True)
            
            semantic_sim = float(util.cos_sim(answer_emb, context_emb)[0][0])
            signals['semantic_similarity'] = semantic_sim
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            signals['semantic_similarity'] = 0.5  # Neutral fallback
        
        # Signal 2: Named entity overlap (30%)
        try:
            nlp = self._load_nlp()
            if nlp is not None:
                answer_doc = nlp(answer[:500])
                context_doc = nlp(context[:2000])
                
                answer_ents = {ent.text.lower() for ent in answer_doc.ents}
                context_ents = {ent.text.lower() for ent in context_doc.ents}
                
                if len(answer_ents) > 0:
                    entity_overlap = len(answer_ents & context_ents) / len(answer_ents)
                else:
                    entity_overlap = 0.5 
                signals['entity_overlap'] = entity_overlap
            else:
                signals['entity_overlap'] = 0.5
        except Exception as e:
            logger.warning(f"Entity overlap failed: {e}")
            signals['entity_overlap'] = 0.5
        
        # Signal 3: Content word overlap (20%)
        try:
            stopwords = self._load_stopwords()
            
            answer_content = {
                w.lower() for w in answer.split()
                if len(w) > 3 and w.lower() not in stopwords
            }
            context_content = {
                w.lower() for w in context.split()
                if len(w) > 3 and w.lower() not in stopwords
            }
            
            if len(answer_content) > 0:
                content_overlap = len(answer_content & context_content) / len(answer_content)
            else:
                content_overlap = 0.5
            
            signals['content_word_overlap'] = content_overlap
        except Exception as e:
            logger.warning(f"Content overlap failed: {e}")
            signals['content_word_overlap'] = 0.5
        
        # Signal 4: Length reasonableness (10%)
        answer_len = len(answer.split())
        is_reasonable = 5 < answer_len < 300  # Not too short, not too long
        is_not_empty = len(answer.strip()) > 0
        is_not_idk = "don't know" not in answer.lower() and "i don't" not in answer.lower()
        
        length_score = 1.0 if (is_reasonable and is_not_empty and is_not_idk) else 0.3
        signals['length_reasonableness'] = length_score
        
        # Weighted combination
        combined_score = (
            self.config['weight_semantic'] * signals['semantic_similarity'] +
            self.config['weight_entity'] * signals['entity_overlap'] +
            self.config['weight_token'] * signals['content_word_overlap'] +
            self.config['weight_length'] * signals['length_reasonableness']
        )
        
        # Determine verdict based on thresholds
        if combined_score >= self.config['threshold_supported']:
            verdict = "SUPPORTED"
            is_supported = True
        elif combined_score >= self.config['threshold_partial']:
            verdict = "PARTIALLY_SUPPORTED"
            is_supported = False
        else:
            verdict = "NOT_SUPPORTED"
            is_supported = False
        
        return {
            "verdict": verdict,
            "is_supported": is_supported,
            "confidence": float(combined_score),
            "signals": {k: float(v) for k, v in signals.items()},
            "unsupported_claims": [],  # Placeholder for future enhancement
            "hallucinations": []  # Placeholder for future enhancement
        } # Neutral