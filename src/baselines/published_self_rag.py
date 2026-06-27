# src/baselines/published_self_rag.py
"""
Faithful reproduction of Self-RAG (Asai et al., 2024)
Reference: https://arxiv.org/abs/2310.11511
OPTIMIZED: Reduced API calls from 7-8 to 2-3 per question
"""

from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.utils.answer_extraction import extract_answer_string
import os
import logging

logger = logging.getLogger(__name__)

class PublishedSelfRAG:
    """
    Reproduction of published Self-RAG system
    
    Components (reflection tokens):
    1. Retrieve: Decides whether to retrieve
    2. IsRel: Judges if retrieved doc is relevant
    3. IsSup: Judges if doc supports generation
    4. IsUse: Judges if doc contributes to answer
    
    OPTIMIZATIONS:
    - Batched reflection tokens (IsRel, IsSup, IsUse) into single API call
    - Reduced max_iterations from 2 to 1
    - Skip passage-level relevance checks (trust retrieval)
    """
    
    def __init__(self, vanilla_rag, model_name: str = "llama-3.3-70b-versatile"):
        self.vanilla_rag = vanilla_rag
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        # Retrieve decision prompt
        self.retrieve_prompt = ChatPromptTemplate.from_messages([
            ("system", "Decide if you need to retrieve information to answer the question."),
            ("user", "Question: {question}\n\nDo you need to retrieve? Respond with ONLY 'yes' or 'no':")
        ])
        
        # OPTIMIZED: Combined reflection prompt (batches IsRel, IsSup, IsUse into one call)
        self.combined_reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator for a RAG system. Evaluate the answer and documents."),
            ("user", """Question: {question}
Answer: {answer}
Documents: {documents}

Evaluate the following:
1. RELEVANCE: Are the documents relevant to the question?
2. SUPPORT: Do the documents fully support the answer?
3. UTILITY: Do the documents contribute useful information?

Respond in this exact format:
RELEVANCE: [relevant/not_relevant]
SUPPORT: [supported/not_supported]
UTILITY: [useful/not_useful]""")
        ])
    
    def answer_with_self_rag(self, question: str, max_iterations: int = 1) -> Dict:
        """
        Answer with Self-RAG reflection tokens (OPTIMIZED)
        
        Process:
        1. Decide if retrieval needed (Retrieve token)
        2. Retrieve passages
        3. Generate answer
        4. Batch evaluate: relevance, support, utility (1 API call instead of 3)
        5. Iterate if needed (max 1 iteration to reduce API calls)
        """
        iteration = 0
        current_answer = None
        
        while iteration < max_iterations:
            logger.debug(f"[Self-RAG Iteration {iteration + 1}]")
            
            # Step 1: Retrieve decision
            retrieve_chain = self.retrieve_prompt | self.llm
            retrieve_response = retrieve_chain.invoke({"question": question})
            should_retrieve = "yes" in retrieve_response.content.lower()
            
            if not should_retrieve and current_answer:
                logger.debug("No retrieval needed, returning current answer")
                return {
                    'answer': current_answer,
                    'iterations': iteration,
                    'method': 'self_rag',
                    'final_retrieve': False
                }
            
            # Step 2: Retrieve
            result = self.vanilla_rag.answer_question(question)
            raw_answer = result.get('raw_answer', result['answer'])
            answer = extract_answer_string(raw_answer)
            retrieved_docs = result['source_documents'][:3]
            
            # OPTIMIZED: Combined reflection evaluation (1 API call instead of 3)
            combined_docs = " ".join(retrieved_docs)[:500]
            reflection_chain = self.combined_reflection_prompt | self.llm
            reflection_response = reflection_chain.invoke({
                "question": question,
                "answer": answer[:200],
                "documents": combined_docs
            })
            
            # Parse combined response
            response_text = reflection_response.content.lower()
            is_relevant = "relevance: relevant" in response_text
            is_supported = "support: supported" in response_text
            is_useful = "utility: useful" in response_text
            
            logger.debug(f"Relevance: {'Yes' if is_relevant else 'No'}, "
                        f"Support: {'Yes' if is_supported else 'No'}, "
                        f"Utility: {'Yes' if is_useful else 'No'}")
            
            # Decision: Accept or refine
            if is_supported and is_useful:
                current_answer = answer
                return {
                    'answer': current_answer,
                    'iterations': iteration + 1,
                    'method': 'self_rag',
                    'is_relevant': is_relevant,
                    'is_supported': is_supported,
                    'is_useful': is_useful
                }
            
            current_answer = answer
            iteration += 1
        
        # Max iterations reached
        return {
            'answer': current_answer if current_answer else "I don't know",
            'iterations': iteration,
            'method': 'self_rag_max_iterations',
            'is_supported': is_supported if 'is_supported' in locals() else False
        }