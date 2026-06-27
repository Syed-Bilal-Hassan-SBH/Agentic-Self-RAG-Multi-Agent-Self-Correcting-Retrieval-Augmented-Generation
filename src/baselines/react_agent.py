# src/baselines/react_agent.py
"""
ReAct (Reasoning + Acting) Agent Implementation
Baseline agent that uses explicit reasoning followed by action selection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Optional, Any
import time
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from src.utils.data_utils import normalize_answer
from src.utils.logging_utils import setup_logger
from src.utils.retriever import FAISSRetriever

logger = setup_logger("react_agent", log_file="logs/react_agent.log")


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent
    
    Follows the ReAct pattern:
    1. Thought: Reason about what to do
    2. Action: Choose an action (retrieve, synthesize, finish)
    3. Observation: See the result
    4. Repeat until final answer
    
    This is a simplified version compared to the full ReAct paper,
    but demonstrates the core reasoning + acting pattern.
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        retriever: Optional[FAISSRetriever] = None,
        max_steps: int = 5,
        temperature: float = 0.0
    ):
        """
        Initialize ReAct Agent
        
        Args:
            model_name: Groq model to use
            retriever: FAISS retriever for context
            max_steps: Maximum reasoning/action cycles
            temperature: Model temperature for deterministic output
        """
        self.model_name = model_name
        self.retriever = retriever
        self.max_steps = max_steps
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize prompts
        self._init_prompts()
        
        logger.info(f"ReActAgent initialized with model: {model_name}")
    
    def _init_prompts(self):
        """Initialize prompt templates for reasoning and action"""
        
        self.thought_action_prompt = PromptTemplate(
            input_variables=["question", "retrieved_context", "conversation_history"],
            template="""You are a helpful assistant that uses ReAct (Reasoning + Acting) to answer questions.

Current Question: {question}

Retrieved Context:
{retrieved_context}

Previous Reasoning:
{conversation_history}

Now, provide:
1. Your THOUGHT about what to do next (reason about the question)
2. Your ACTION choice: 'retrieve' for more info, 'synthesize' to answer, or 'finish' when done

Format your response as:
Thought: [Your reasoning]
Action: [retrieve/synthesize/finish]
If Action is 'retrieve': What to search for?
If Action is 'finish': What is the final answer?"""
        )
        
        self.answer_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Based on the following context, answer the question directly and concisely.

Question: {question}

Context:
{context}

Answer:"""
        )
    
    def _retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context for a query"""
        if not self.retriever:
            return ""
        
        try:
            results = self.retriever.search(query, k=top_k)
            if results:
                context_text = "\n".join([f"- {result}" for result in results])
                return context_text
            return "No relevant documents found."
        except Exception as e:
            logger.warning(f"Retrieval error: {e}")
            return ""
    
    def _parse_thought_action(self, response: str) -> tuple:
        """Parse LLM response to extract thought and action"""
        lines = response.split("\n")
        
        thought = ""
        action = "finish"
        action_input = ""
        
        for i, line in enumerate(lines):
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action_text = line.replace("Action:", "").strip()
                # Extract action type
                if "retrieve" in action_text.lower():
                    action = "retrieve"
                    # Look for the search query
                    if i + 1 < len(lines) and "What to search" in lines[i + 1]:
                        action_input = lines[i + 1].split("?")[-1].strip()
                elif "synthesize" in action_text.lower():
                    action = "synthesize"
                elif "finish" in action_text.lower():
                    action = "finish"
                    if i + 1 < len(lines) and "answer" in lines[i + 1].lower():
                        action_input = lines[i + 1].split(":")[-1].strip()
        
        return thought, action, action_input
    
    def answer(
        self,
        question: str,
        context: Optional[str] = None,
        num_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Answer a question using ReAct pattern
        
        Args:
            question: The question to answer
            context: Optional initial context
            num_hops: Expected number of reasoning steps
        
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"ReAct answering: {question[:80]}...")
        
        # Initialize conversation history
        messages: List[BaseMessage] = []
        conversation_history = ""
        retrieved_docs = []
        step_count = 0
        final_answer = None
        
        # Initial context retrieval
        if not context:
            context = self._retrieve_context(question, top_k=5)
        retrieved_docs.append(context)
        
        # Main reasoning loop
        for step in range(self.max_steps):
            step_count += 1
            
            # Get LLM's thought and action
            thought_action_input = self.thought_action_prompt.format(
                question=question,
                retrieved_context=context,
                conversation_history=conversation_history
            )
            
            response = self.llm.invoke([HumanMessage(content=thought_action_input)])
            response_text = response.content
            
            # Parse response
            thought, action, action_input = self._parse_thought_action(response_text)
            
            # Update conversation history
            conversation_history += f"\nStep {step_count}:\nThought: {thought}\nAction: {action}"
            
            # Execute action
            if action == "retrieve":
                # Retrieve more context
                if action_input:
                    context = self._retrieve_context(action_input, top_k=5)
                    retrieved_docs.append(context)
                    conversation_history += f"\nObservation: Retrieved {len(context.split())} words of context"
                else:
                    conversation_history += "\nObservation: No search query provided"
            
            elif action == "synthesize":
                # Continue reasoning with current context
                conversation_history += "\nObservation: Continuing with available context"
            
            elif action == "finish":
                # Generate final answer
                answer_input = self.answer_prompt.format(
                    question=question,
                    context=context
                )
                
                final_response = self.llm.invoke([HumanMessage(content=answer_input)])
                final_answer = final_response.content.strip()
                break
        
        # If loop ended without explicit finish
        if not final_answer:
            answer_input = self.answer_prompt.format(
                question=question,
                context=context
            )
            final_response = self.llm.invoke([HumanMessage(content=answer_input)])
            final_answer = final_response.content.strip()
        
        elapsed_time = time.time() - start_time
        
        return {
            "answer": final_answer,
            "question": question,
            "reasoning_steps": step_count,
            "retrieved_documents": retrieved_docs,
            "reasoning_history": conversation_history,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def batch_answer(
        self,
        questions: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions
        
        Args:
            questions: List of questions
            contexts: Optional list of contexts
        
        Returns:
            List of answer dictionaries
        """
        results = []
        
        for i, question in enumerate(questions):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.answer(question, context=context)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(questions)} questions")
        
        return results


def main():
    """Test ReAct Agent"""
    from src.utils.data_utils import load_hotpotqa
    
    # Initialize retriever
    retriever = FAISSRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize agent
    agent = ReActAgent(retriever=retriever, max_steps=5)
    
    # Load sample questions
    data = load_hotpotqa(split="validation", sample_size=5)
    
    questions = [item["question"] for item in data]
    
    # Answer questions
    logger.info(f"Answering {len(questions)} questions with ReAct Agent...")
    results = agent.batch_answer(questions)
    
    # Print results
    for i, result in enumerate(results):
        logger.info(f"\nQuestion {i+1}: {result['question']}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Reasoning Steps: {result['reasoning_steps']}")
        logger.info(f"Time: {result['elapsed_time']:.2f}s")


if __name__ == "__main__":
    main()
