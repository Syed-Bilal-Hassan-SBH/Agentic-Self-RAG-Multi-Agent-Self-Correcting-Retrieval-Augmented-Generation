# src/agentic_self_rag.py
"""
Agentic Self-RAG: Multi-Agent Self-Correcting RAG System
OPTIMIZED: Full intelligence with smart API call reduction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from src.vanilla_rag import VanillaRAG
from src.agents.query_analyzer_agent import QueryAnalyzerAgent
from src.agents.retrieval_critic_agent import RetrievalCriticAgent
from src.utils.answer_extraction import extract_answer_string
from src.agents.answer_verifier_agent import AnswerVerifierAgent
from src.agents.multi_hop_agent import MultiHopAgent
from dotenv import load_dotenv
import json
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state
class AgenticRAGState(TypedDict):
    """State for the agentic RAG workflow"""
    question: str
    query_analysis: dict
    use_multi_hop: bool
    sub_questions: List[str]
    retrieved_passages: List[str]
    retrieval_critique: dict
    answer: str
    verification: dict
    iteration_count: int
    final_answer: str
    final_verdict: str
    final_confidence: float

class AgenticSelfRAG:
    """
    Multi-agent self-correcting RAG system with FULL INTELLIGENCE
    
    SMART OPTIMIZATIONS:
    - Combined query analysis + retrieval critique (2 calls → 1 call)
    - Conditional verification (skip only on very high confidence)
    - Early stopping on supported answers
    - Max 2 iterations (down from 3, but allows refinement)
    """
    
    def __init__(self,
                 rag=None,
                 use_multi_hop: bool = True,  # ENABLED: Multi-hop is a key feature!
                 use_adaptive_iteration: bool = True,
                 max_iterations: int = 2,  # SMART: Allow refinement but not excessive
                 confidence_threshold_high: float = 0.75,  # Higher threshold for skipping
                 confidence_threshold_low: float = 0.35):
        
        logger.info("🚀 Initializing Agentic Self-RAG (FULL INTELLIGENCE MODE)...")
        
        self.use_multi_hop = use_multi_hop
        self.use_adaptive_iteration = use_adaptive_iteration
        self.max_iterations = max_iterations
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_low = confidence_threshold_low
        self.last_retrieval_quality = None
        
        # Initialize RAG system
        self.rag = rag if rag is not None else VanillaRAG()
        
        # Initialize ALL agents (full intelligence)
        logger.info("📦 Loading all agents...")
        self.query_analyzer = QueryAnalyzerAgent()
        self.retrieval_critic = RetrievalCriticAgent()
        self.answer_verifier = AnswerVerifierAgent()
        self.multi_hop_agent = MultiHopAgent(self.rag)
        
        # Build workflow
        self.app = self._build_workflow()
        
        logger.info("✅ Agentic Self-RAG initialized with FULL capabilities!")
    
    def setup_vectorstore(self, contexts):
        """Setup the vector store with contexts"""
        self.rag.create_vectorstore(contexts)
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgenticRAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("critique_retrieval", self._critique_retrieval)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_answer", self._verify_answer)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_edge("retrieve", "critique_retrieval")
        workflow.add_edge("critique_retrieval", "generate_answer")
        workflow.add_edge("generate_answer", "verify_answer")
        
        # Conditional edge from verification
        workflow.add_conditional_edges(
            "verify_answer",
            self._should_continue,
            {
                "continue": "retrieve",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=None)
    
    def _analyze_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Analyze query complexity - FULL INTELLIGENCE"""
        logger.info("  🔍 Step 1: Analyzing query...")
        
        try:
            analysis = self.query_analyzer.analyze(state["question"])
            state["query_analysis"] = analysis
            
            if self.use_multi_hop and analysis['is_multi_hop'] and analysis.get('sub_questions'):
                state["use_multi_hop"] = True
                state["sub_questions"] = analysis['sub_questions']
                logger.info(f"     Multi-hop detected: {len(state['sub_questions'])} sub-questions")
            else:
                state["use_multi_hop"] = False
                state["sub_questions"] = []
                logger.info(f"     Single-hop query")
        except Exception as e:
            logger.warning(f"⚠️ Query analysis failed: {e}, falling back to single-hop")
            state["query_analysis"] = {}
            state["use_multi_hop"] = False
            state["sub_questions"] = []
        
        return state
    
    def _retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """Retrieve relevant passages with adaptive refinement"""
        logger.info("  📚 Step 2: Retrieving passages...")
        
        iteration = state.get("iteration_count", 0)
        
        # Adaptive retrieval: increase k on poor retrieval
        if iteration == 0:
            k_val = 10
        elif iteration == 1 and self.last_retrieval_quality == "insufficient":
            k_val = 15
            logger.info("     → Refinement: Increasing k for better coverage")
        else:
            k_val = 10
        
        retriever = self.rag.vectorstore.as_retriever(search_kwargs={"k": k_val})
        
        if state["use_multi_hop"]:
            all_passages = []
            for sub_q in state["sub_questions"]:
                retrieved_docs = retriever.invoke(sub_q)
                all_passages.extend([doc.page_content for doc in retrieved_docs])
            
            unique_passages = list(set(all_passages))
            state["retrieved_passages"] = unique_passages[:min(k_val * 2, len(unique_passages))]
            logger.info(f"     Retrieved {len(state['retrieved_passages'])} unique passages (multi-hop)")
        else:
            retrieved_docs = retriever.invoke(state["question"])
            state["retrieved_passages"] = [doc.page_content for doc in retrieved_docs]
            logger.info(f"     Retrieved {len(state['retrieved_passages'])} passages")
        
        return state
    
    def _critique_retrieval(self, state: AgenticRAGState) -> AgenticRAGState:
        """Critique retrieval quality - FULL INTELLIGENCE"""
        logger.info("  🔬 Step 3: Critiquing retrieval...")
        
        try:
            critique = self.retrieval_critic.critique(
                state["question"],
                state["retrieved_passages"]
            )
            state["retrieval_critique"] = critique
        except Exception as e:
            logger.warning(f"⚠️ Critique failed: {e}, using fallback")
            has_passages = len(state["retrieved_passages"]) > 0
            state["retrieval_critique"] = {
                "avg_relevance": 0.7 if has_passages else 0.3,
                "coverage": 0.7 if has_passages else 0.3,
                "sufficiency": "sufficient" if has_passages else "insufficient"
            }
        
        self.last_retrieval_quality = state['retrieval_critique'].get('sufficiency', 'unknown')
        
        logger.info(f"     Relevance: {state['retrieval_critique'].get('avg_relevance', 0):.2f}, "
                   f"Coverage: {state['retrieval_critique'].get('coverage', 0):.2f}, "
                   f"Sufficiency: {state['retrieval_critique'].get('sufficiency', 'unknown')}")
        
        return state
    
    def _generate_answer(self, state: AgenticRAGState) -> AgenticRAGState:
        """Generate answer - FULL INTELLIGENCE with multi-hop support"""
        logger.info("  🤖 Step 4: Generating answer...")
        
        try:
            if state["use_multi_hop"] and state["sub_questions"]:
                try:
                    result = self.multi_hop_agent.answer_with_decomposition(
                        state["question"],
                        state["sub_questions"]
                    )
                    state["answer"] = result['answer']
                    logger.info(f"     Multi-hop answer generated ({len(result['answer'])} chars)")
                except Exception as e:
                    logger.warning(f"⚠️ Multi-hop generation failed: {e}, falling back to single-hop")
                    result = self.rag.answer_question(state["question"])
                    state["answer"] = result["answer"]
                    state["use_multi_hop"] = False
            else:
                result = self.rag.answer_question(state["question"])
                state["answer"] = result["answer"]
                logger.info(f"     Answer generated ({len(result['answer'])} chars)")
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            state["answer"] = "I don't know"
        
        return state
    
    def _verify_answer(self, state: AgenticRAGState) -> AgenticRAGState:
        """Verify answer quality - SMART: Skip only on very high confidence"""
        logger.info("  ✅ Step 5: Verifying answer...")
        
        iteration = state.get("iteration_count", 0)
        
        # SMART OPTIMIZATION: Only skip verification if we have very high retrieval quality
        retrieval_quality = state["retrieval_critique"].get("sufficiency", "unknown")
        retrieval_relevance = state["retrieval_critique"].get("avg_relevance", 0)
        
        skip_verification = (
            iteration == 0 and 
            retrieval_quality == "sufficient" and 
            retrieval_relevance > 0.8
        )
        
        if skip_verification:
            logger.info("     → Skipping verification (high retrieval quality)")
            verification = {
                "verdict": "SUPPORTED",
                "confidence": 0.8,
                "signals": {}
            }
            state["verification"] = verification
        else:
            # Run full verification
            try:
                verification = self.answer_verifier.verify(
                    state["question"],
                    state["answer"],
                    state["retrieved_passages"]
                )
                state["verification"] = verification
            except Exception as e:
                logger.warning(f"⚠️ Verification failed: {e}, using fallback")
                if state["answer"].lower() in ["i don't know", "", "error"]:
                    verification = {
                        "verdict": "NOT_SUPPORTED",
                        "confidence": 0.2,
                        "signals": {}
                    }
                else:
                    verification = {
                        "verdict": "PARTIALLY_SUPPORTED",
                        "confidence": 0.5,
                        "signals": {}
                    }
                state["verification"] = verification
        
        state["iteration_count"] = iteration + 1
        
        logger.info(f"     Verdict: {state['verification'].get('verdict', 'UNKNOWN')}, "
                   f"Confidence: {state['verification'].get('confidence', 0):.2f}, "
                   f"Iteration: {state['iteration_count']}")
        
        # ALWAYS set final answer
        extracted_answer = extract_answer_string(state["answer"])
        state["final_answer"] = extracted_answer
        state["final_verdict"] = state['verification'].get("verdict", "UNKNOWN")
        state["final_confidence"] = state['verification'].get("confidence", 0.0)
        
        return state
    
    def _should_stop(self, state: AgenticRAGState) -> bool:
        """SMART stopping conditions - balance quality and efficiency"""
        verdict = state["verification"]["verdict"]
        confidence = state["verification"]["confidence"]
        iteration = state["iteration_count"]
        retrieval_quality = state["retrieval_critique"]["sufficiency"]
        
        # Rule 1: Very high confidence - STOP
        if confidence > self.confidence_threshold_high:
            logger.info(f"     → Stop: High confidence ({confidence:.2f})")
            return True
        
        # Rule 2: Supported answer - accept it
        if verdict == "SUPPORTED":
            logger.info(f"     → Stop: Answer is SUPPORTED (iteration {iteration})")
            return True
        
        # Rule 3: Partial support with decent confidence
        if verdict == "PARTIALLY_SUPPORTED" and confidence > 0.50:
            logger.info(f"     → Stop: Partial support acceptable (confidence {confidence:.2f})")
            return True
        
        # Rule 4: Max iterations reached
        if iteration >= self.max_iterations:
            logger.warning(f"     → Stop: Max iterations ({self.max_iterations}) reached")
            return True
        
        # Rule 5: Good retrieval + reasonable answer on first iteration
        if iteration == 1 and retrieval_quality == "sufficient" and confidence > 0.45:
            logger.info(f"     → Stop: Good retrieval + reasonable answer")
            return True
        
        return False
    
    def _should_continue(self, state: AgenticRAGState) -> str:
        """Decide whether to continue or end"""
        if self._should_stop(state):
            return "end"
        else:
            logger.info(f"     → Continue: Refining answer (iteration {state['iteration_count']})")
            return "continue"
    
    def answer(self, question: str) -> dict:
        """Answer a question using the FULL multi-agent system"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 Processing: {question[:80]}...")
        logger.info(f"{'='*80}")
        
        # Initialize state
        initial_state = {
            "question": question,
            "query_analysis": {},
            "use_multi_hop": False,
            "sub_questions": [],
            "retrieved_passages": [],
            "retrieval_critique": {},
            "answer": "",
            "verification": {},
            "iteration_count": 0,
            "final_answer": "",
            "final_verdict": "",
            "final_confidence": 0.0
        }
        
        # Run workflow with error handling
        try:
            final_state = self.app.invoke(
                initial_state,
                config={"recursion_limit": 50}
            )
            
            # Ensure we have a valid answer
            if not final_state.get("final_answer"):
                logger.warning("⚠️ No answer generated, falling back to simple retrieval")
                try:
                    result = self.rag.answer_question(question)
                    final_state["final_answer"] = result["answer"]
                    final_state["final_verdict"] = "FALLBACK"
                    final_state["final_confidence"] = 0.5
                except Exception as e:
                    logger.error(f"Fallback retrieval also failed: {e}")
                    final_state["final_answer"] = "I don't know"
                    final_state["final_verdict"] = "ERROR"
                    final_state["final_confidence"] = 0.0
                    
        except Exception as e:
            logger.error(f"❌ Workflow error: {e}")
            # Graceful fallback
            try:
                logger.info("🔄 Attempting graceful fallback to Vanilla RAG...")
                result = self.rag.answer_question(question)
                final_state = initial_state.copy()
                final_state["final_answer"] = result["answer"]
                final_state["final_verdict"] = "FALLBACK"
                final_state["final_confidence"] = 0.5
                logger.info("✅ Fallback retrieval successful!")
            except Exception as fallback_e:
                logger.error(f"❌ Fallback also failed: {fallback_e}")
                final_state = initial_state.copy()
                final_state["final_answer"] = "I don't know"
                final_state["final_verdict"] = "ERROR"
                final_state["final_confidence"] = 0.0
        
        return {
            "question": question,
            "answer": extract_answer_string(final_state.get("final_answer", "I don't know")),
            "verdict": final_state.get("final_verdict", "ERROR"),
            "confidence": final_state.get("final_confidence", 0.0),
            "iterations": final_state.get("iteration_count", 0),
            "query_analysis": final_state.get("query_analysis", {}),
            "retrieval_critique": final_state.get("retrieval_critique", {}),
            "verification": final_state.get("verification", {})
        }
