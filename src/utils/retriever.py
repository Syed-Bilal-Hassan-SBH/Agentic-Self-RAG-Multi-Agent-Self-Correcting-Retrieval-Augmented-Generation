# src/utils/retriever.py
"""
Retriever implementations for RAG systems
FAISS for dense retrieval, HuggingFace embeddings, answer extraction
"""

import re
import logging
import numpy as np
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# Stub retriever classes for compatibility
class FAISSRetriever:
    """FAISS-based dense retriever"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        self.model_name = model_name
        logger.info(f"Initialized FAISS retriever with model: {model_name}")
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for similar passages"""
        return [f"Retrieved passage {i+1}" for i in range(k)]


class HFRetriever:
    """HuggingFace embeddings based retriever"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        self.model_name = model_name
        logger.info(f"Initialized HF retriever with model: {model_name}")
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for similar passages"""
        return [f"Retrieved passage {i+1}" for i in range(k)]


class AnswerExtractor:
    """
    Extracts the core answer from verbose LLM outputs
    
    Strategies:
    1. Sentence segmentation - identify likely answer sentences
    2. LLM-based extraction - use a smaller model to extract key answer
    3. Pattern matching - extract names, numbers, dates, yes/no
    4. Fallback - use first non-reasoning sentence
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting concise answers from verbose text.

Given a verbose answer, extract ONLY the core answer in as few words as possible.
Return ONLY the extracted answer, nothing else.

Examples:
- Verbose: "Let's think step by step. First, we need to understand... The answer is John Smith."
  Extracted: "John Smith"
  
- Verbose: "After analyzing the evidence, the year was 1876."
  Extracted: "1876"
  
- Verbose: "Considering both possibilities, yes, they are both board games."
  Extracted: "yes" """),
            ("user", "Extract the core answer from this text:\n\n{answer}")
        ])
    
    def extract(self, verbose_answer: str, max_length: int = 300) -> str:
        """
        Extract concise answer from verbose output
        
        Args:
            verbose_answer: Full verbose LLM output
            max_length: Maximum length of extracted answer in characters (increased default)
            
        Returns:
            Concise answer string
        """
        if not verbose_answer:
            return verbose_answer.strip()
        
        # If already short, return as-is
        if len(verbose_answer) < 150:
            return verbose_answer.strip()
        
        # Try strategy 1: Sentence-based extraction (most reliable)
        sentence_result = self._extract_by_sentence(verbose_answer, max_length)
        if sentence_result and len(sentence_result) > 3:
            logger.debug(f"✓ Sentence extraction successful: {sentence_result[:100]}")
            return sentence_result
        
        # Try strategy 2: Pattern matching for common answer types
        pattern_result = self._extract_by_pattern(verbose_answer)
        if pattern_result and len(pattern_result) > 3:
            logger.debug(f"✓ Pattern extraction successful: {pattern_result}")
            return pattern_result
        
        # Try strategy 3: LLM-based extraction (more expensive)
        try:
            llm_result = self._extract_by_llm(verbose_answer, max_length)
            if llm_result and len(llm_result) > 3:
                logger.debug(f"✓ LLM extraction successful: {llm_result}")
                return llm_result
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        # Fallback: Return first substantial sentence + next if needed
        sentences = re.split(r'[.!?]+', verbose_answer)
        result = []
        char_count = 0
        
        for sent in sentences:
            cleaned = sent.strip()
            if not cleaned or len(cleaned) < 3:
                continue
            
            # Skip reasoning/explanation sentences
            if any(word in cleaned.lower() for word in 
                   ['let', 'think', 'step', 'first', 'analyze', 'consider', 'reasoning', 'evidence', 'based on']):
                continue
            
            result.append(cleaned)
            char_count += len(cleaned) + 2
            
            # Stop if we have enough content
            if char_count > 50 or len(result) >= 2:
                break
        
        if result:
            final_answer = '. '.join(result)
            return final_answer[:max_length].strip()
        
        # Last resort: Return first max_length chars
        return verbose_answer[:max_length].strip()
    
    def _extract_by_pattern(self, text: str) -> Optional[str]:
        """
        Extract answer using regex patterns for common types
        """
        text_lower = text.lower()
        
        # Pattern 1: "the answer is X" or "answer: X"
        match = re.search(r'(?:the\s+)?answer(?:\s+is)?[:\s]+([^.!?]+)', text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if 5 < len(candidate) < 200:
                return candidate
        
        # Pattern 2: Yes/No questions
        if 'yes' in text_lower and 'no' not in text_lower[-100:]:
            return 'yes'
        if 'no' in text_lower and 'yes' not in text_lower[-100:]:
            return 'no'
        
        # Pattern 3: Extract years (4 digits)
        match = re.search(r'\b(19|20)\d{2}\b', text)
        if match:
            return match.group(0)
        
        # Pattern 4: Extract proper nouns/capitalized phrases
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if matches:
            # Filter out common words at sentence start
            for match in matches:
                if match not in ['The', 'According', 'Based', 'Therefore', 'However', 'First', 'Because']:
                    return match
        
        return None
    
    def _extract_by_sentence(self, text: str, max_length: int = 300) -> Optional[str]:
        """
        Extract shortest non-reasoning sentence(s)
        Returns 1-2 sentences as the answer
        """
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        candidates = []
        for sent in sentences:
            cleaned = sent.strip()
            if not cleaned:
                continue
            
            # Skip reasoning/explanation sentences
            reasoning_words = [
                'let', 'think', 'step', 'first', 'analyze', 'consider', 'evidence',
                'reason', 'believe', 'seem', 'appear', 'according', 'based', 'found',
                'would', 'could', 'might', 'suggests', 'indicates', 'important',
                'therefore', 'however', 'because', 'thus'
            ]
            
            words_lower = cleaned.lower().split()
            has_reasoning = any(word in reasoning_words for word in words_lower[:3])  # Check first 3 words
            
            # Be more lenient: accept sentences 10-300 chars
            if not has_reasoning and 10 < len(cleaned) < max_length:
                candidates.append(cleaned)
        
        # Return 1-2 sentences combined
        if candidates:
            # Sort by length (prefer medium-length over very short)
            sorted_candidates = sorted(candidates, key=lambda x: (abs(len(x) - 50), len(x)))
            
            # Take first candidate, or combine first two if both are short
            if len(sorted_candidates[0]) < 30 and len(sorted_candidates) > 1:
                return sorted_candidates[0] + ". " + sorted_candidates[1]
            else:
                return sorted_candidates[0]
        
        return None
    
    def _extract_by_llm(self, verbose_answer: str, max_length: int) -> Optional[str]:
        """
        Use LLM to extract answer (expensive but accurate)
        """
        try:
            chain = self.extraction_prompt | self.llm
            response = chain.invoke({"answer": verbose_answer[:2000]})  # Truncate for API limits
            
            extracted = response.content.strip()
            
            # Clean up
            extracted = extracted.strip('"\'')
            
            if extracted and len(extracted) < max_length:
                return extracted
        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")
        
        return None


def extract_answer(verbose_answer: str, use_llm: bool = True) -> str:
    """
    Convenience function to extract answer
    
    Args:
        verbose_answer: Full LLM output
        use_llm: Whether to use LLM for extraction (slower but more accurate)
        
    Returns:
        Concise answer
    """
    # For very short answers, return as-is
    if len(verbose_answer) < 50:
        return verbose_answer.strip()
    
    extractor = AnswerExtractor()
    # Disable LLM extraction if not explicitly requested (too expensive)
    if not use_llm:
        extractor.extraction_prompt = None
    
    return extractor.extract(verbose_answer)
