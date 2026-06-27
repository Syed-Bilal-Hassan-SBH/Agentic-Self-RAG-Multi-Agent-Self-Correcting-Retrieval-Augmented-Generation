# src/improved_agents.py
"""
Improved agent prompts and behaviors for better performance
Focuses on answer quality, confidence calibration, and reduction of hallucinations
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import logging

logger = logging.getLogger(__name__)


class ImprovedQueryAnalyzer:
    """Enhanced query analyzer with better decomposition"""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        # Enhanced prompt with clearer examples
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query analyzer for multi-hop question answering.

Your job is to:
1. Identify if a question needs multi-hop reasoning (requires info from 2+ entities/facts)
2. Decompose it into simple atomic sub-questions
3. Assess complexity level

Multi-hop examples:
- "Who is the spouse of the director of Titanic?" → Need: (1) Director of Titanic (2) Their spouse
- "Were X and Y from the same country?" → Need: (1) X's country (2) Y's country

Single-hop examples:
- "What is the capital of France?" → Direct factual lookup
- "Who directed Titanic?" → Direct entity/relation lookup

Return ONLY valid JSON (no markdown):"""),
            ("user", """Analyze: {question}

Respond with ONLY this JSON structure (no code blocks, no other text):
{{
  "is_multi_hop": <boolean>,
  "complexity": "<simple|medium|complex>",
  "sub_questions": [<list of atomic sub-questions if multi-hop, else empty>],
  "decomposition_reasoning": "<brief explanation of the decomposition strategy>"
}}""")
        ])
    
    def analyze(self, question: str) -> dict:
        """Analyze query and return structured breakdown"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"question": question})
            content = response.content.strip()
            
            # Clean up markdown if present
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            # Extract JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            
            result = json.loads(content)
            logger.debug(f"Query analysis: multi_hop={result.get('is_multi_hop')}, "
                        f"complexity={result.get('complexity')}")
            return result
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "is_multi_hop": False,
                "complexity": "simple",
                "sub_questions": [],
                "decomposition_reasoning": "Error in analysis, treating as simple"
            }


class ImprovedAnswerVerifier:
    """Enhanced answer verifier with better confidence scoring and hallucination detection"""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert answer verification system. Your job is to:

1. Check if the answer is supported by the given context
2. Assess hallucination risk (fabricated information not in context)
3. Evaluate answer completeness
4. Assign a confidence score (0.0 to 1.0)

Scoring guidance:
- 0.0-0.3: Poor quality, hallucinatory, or unsupported
- 0.3-0.6: Partially correct but missing info or has issues
- 0.6-0.8: Good quality, mostly supported
- 0.8-1.0: Excellent, fully supported and complete

Return ONLY valid JSON (no markdown):"""),
            ("user", """Context: {context}

Question: {question}

Proposed Answer: {answer}

Respond with ONLY this JSON (no code blocks):
{{
  "is_supported": <boolean - is answer supported by context?>,
  "hallucination_risk": "<low|medium|high>",
  "confidence": <0.0 to 1.0>,
  "issues": [<list of issues if any>],
  "verdict": "<accept|revise|reject>",
  "reasoning": "<brief explanation>"
}}""")
        ])
    
    def verify(self, context: str, question: str, answer: str) -> dict:
        """Verify answer quality and confidence"""
        try:
            chain = self.verification_prompt | self.llm
            response = chain.invoke({
                "context": context[:2000],  # Limit context size
                "question": question,
                "answer": answer
            })
            content = response.content.strip()
            
            # Clean JSON
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            
            result = json.loads(content)
            return result
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                "is_supported": False,
                "hallucination_risk": "high",
                "confidence": 0.0,
                "issues": ["Verification error"],
                "verdict": "reject",
                "reasoning": "System error in verification"
            }


class ImprovedAnswerGenerator:
    """Enhanced answer generation with better grounding and direct answers"""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a factual QA system. Your job is to answer questions accurately and concisely.

CRITICAL INSTRUCTIONS:
1. Answer ONLY what is explicitly stated or directly inferable from the given passages
2. Do NOT generate or hallucinate information
3. If you don't know, say "I cannot answer based on the provided information"
4. For yes/no questions, answer with just "yes" or "no" + brief justification
5. For factual questions, provide the specific fact directly
6. Be concise - one or two sentences maximum

Passages (may be multiple):
{context}

Use ONLY information from these passages."""),
            ("user", "Question: {question}\n\nAnswer:")
        ])
    
    def generate(self, context: str, question: str) -> str:
        """Generate answer grounded in context"""
        try:
            chain = self.answer_prompt | self.llm
            response = chain.invoke({
                "context": context[:3000],  # Limit context
                "question": question
            })
            answer = response.content.strip()
            
            # Ensure answer is not too long (indicates hallucination)
            if len(answer) > 500:
                logger.warning(f"Answer too long ({len(answer)} chars), likely hallucinated")
                answer = answer[:500]
            
            return answer
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer"


class ImprovedRetrievalCritic:
    """Enhanced retrieval critic with better passage ranking"""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a passage relevance evaluator for QA systems.

For each passage, assess:
1. Relevance: Does it contain information needed to answer the question?
2. Quality: Is the information clear and factual?
3. Reliability: Is it from a trustworthy source (typical passages are)?

Score each passage: 0.0 (useless) to 1.0 (highly relevant)

Return ONLY valid JSON (no markdown):"""),
            ("user", """Question: {question}

Passages to evaluate:
{passages}

Return ONLY this JSON format (no code blocks):
{{
  "passages_scores": [<list of numbers 0.0-1.0>],
  "top_passages": [<indices of top 2-3 passages>],
  "reasoning": "<brief assessment>"
}}""")
        ])
    
    def critique(self, question: str, passages: list) -> dict:
        """Evaluate passage relevance"""
        try:
            passages_text = "\n\n".join([f"[{i}] {p[:300]}" for i, p in enumerate(passages)])
            
            chain = self.critique_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "passages": passages_text
            })
            content = response.content.strip()
            
            # Clean JSON
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            
            result = json.loads(content)
            return result
        except Exception as e:
            logger.error(f"Retrieval critique failed: {e}")
            return {
                "passages_scores": [0.5] * len(passages),
                "top_passages": list(range(min(3, len(passages)))),
                "reasoning": "Error in critique, using default ranking"
            }
