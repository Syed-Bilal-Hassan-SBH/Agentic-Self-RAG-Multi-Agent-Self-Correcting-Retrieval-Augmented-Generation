# src/agents/multi_hop_agent.py
"""
Multi-Hop Reasoning Agent
Novel contribution: Explicit sub-question decomposition and combination
"""

from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

logger = logging.getLogger(__name__)

class MultiHopAgent:
    """
    Agent for explicit multi-hop question answering
    
    Process:
    1. Receive decomposed sub-questions from QueryAnalyzer
    2. Answer each sub-question with targeted retrieval
    3. Combine sub-answers using chain-of-thought reasoning
    
    Novel contribution over Self-RAG:
    - Explicit decomposition (not just reflection)
    - Sub-question targeted retrieval
    - Structured evidence combination
    """
    
    def __init__(self, rag_system, model_name: str = "llama-3.3-70b-versatile"):
        self.rag = rag_system
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        # Prompt for combining sub-answers
        self.combination_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at synthesizing information from multiple sources.

Given answers to sub-questions, provide a final answer to the original question.

Guidelines:
1. Be direct and complete
2. Include all relevant information from the sub-answers
3. Synthesize logically when combining information
4. Use natural language, not bullet points

Provide only the final answer, no explanation of your reasoning:"""),
            ("user", """Original Question: {original_question}

Sub-question Answers:
{sub_answers}

Final answer:""")
        ])
    
    def answer_with_decomposition(self, 
                                  original_question: str,
                                  sub_questions: List[str]) -> Dict:
        """
        Answer multi-hop question via decomposition
        
        Args:
            original_question: Original complex question
            sub_questions: List of decomposed sub-questions
            
        Returns:
            Dict with final answer, sub-answers, and evidence
        """
        logger.info(f"Multi-hop answering with {len(sub_questions)} sub-questions")
        
        # Step 1: Answer each sub-question
        sub_answers = []
        all_evidence = []
        
        for i, sub_q in enumerate(sub_questions):
            logger.debug(f"  Answering sub-question {i+1}: {sub_q}")
            
            # Retrieve and answer
            result = self.rag.answer_question(sub_q)
            
            sub_answer = {
                'sub_question': sub_q,
                'answer': result['answer'],
                'evidence': result['source_documents'][:2]  # Top 2 passages
            }
            sub_answers.append(sub_answer)
            all_evidence.extend(result['source_documents'][:2])
        
        # Step 2: Combine sub-answers with chain-of-thought
        sub_answers_text = "\n\n".join([
            f"Q{i+1}: {sa['sub_question']}\nA{i+1}: {sa['answer']}"
            for i, sa in enumerate(sub_answers)
        ])
        
        combination_chain = self.combination_prompt | self.llm
        final_response = combination_chain.invoke({
            "original_question": original_question,
            "sub_answers": sub_answers_text
        })
        
        final_answer = final_response.content
        
        logger.info(f"Multi-hop answer generated ({len(final_answer)} chars)")
        
        return {
            'answer': final_answer,
            'sub_answers': sub_answers,
            'all_evidence': all_evidence,
            'method': 'multi-hop-decomposition',
            'num_hops': len(sub_questions)
        }