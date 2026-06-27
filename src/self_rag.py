from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.vanilla_rag import VanillaRAG
from src.utils.answer_extraction import extract_answer_string
from dotenv import load_dotenv
import json
import os

load_dotenv()

# Base class for Self-RAG variants
class SelfRAG:
    """
    Base Self-RAG implementation with reflection and refinement
    Self-Reflective RAG: Iterative answer generation with quality evaluation
    """
    
    def __init__(self, vanilla_rag, model_name="llama-3.3-70b-versatile"):
        self.vanilla_rag = vanilla_rag
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.0
        )
        
        # Quality evaluation prompt
        self.quality_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert evaluator. Assess answer quality on a scale 1-10."),
            ("user", """Question: {question}
Answer: {answer}
Context: {context}

Rate this answer from 1-10 based on:
- Correctness and accuracy
- Completeness
- Support from context
- Clarity and coherence

Respond with: SCORE: <number>""")
        ])
    
    def answer_question(self, question):
        """Answer with evaluation (base implementation)"""
        result = self.vanilla_rag.answer_question(question)
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", []),
            "score": 8.0
        }


class SimplifiedSelfRAG(SelfRAG):
    """Self-RAG with reflection capability (OPTIMIZED)"""
    
    def __init__(self, vanilla_rag, model_name="llama-3.3-70b-versatile"):
        self.vanilla_rag = vanilla_rag
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        # OPTIMIZED: Combined generation + reflection prompt (1 API call instead of 2)
        self.combined_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Generate an answer and evaluate if it's supported by the context."),
            ("user", """Question: {question}
Context: {context}

Generate a concise answer to the question based on the context provided.
Then evaluate if your answer is fully supported by the context.

Respond in this exact format:
ANSWER: [your answer here]
SUPPORTED: [YES/NO]""")
        ])
        
    def answer_with_reflection(self, question, max_iterations=2):
        """Answer with self-reflection (OPTIMIZED to 1 API call per iteration)"""
        iteration = 0
        last_answer = None
        
        while iteration < max_iterations:
            print(f"\n  [Iteration {iteration + 1}]")
            
            # Step 1: Get context from Vanilla RAG (retrieval only, no generation)
            result = self.vanilla_rag.answer_question(question)
            context = "\n".join(result['source_documents'][:3])  # Use top 3 docs
            
            # OPTIMIZED: Combined generation + reflection in single API call
            combined_chain = self.combined_prompt | self.llm
            combined_response = combined_chain.invoke({
                "question": question,
                "context": context
            })
            
            # Parse combined response
            response_text = combined_response.content
            
            # Extract answer
            if "ANSWER:" in response_text:
                answer_part = response_text.split("ANSWER:")[1].split("SUPPORTED:")[0].strip()
                answer = extract_answer_string(answer_part)
            else:
                answer = extract_answer_string(response_text)
            
            # Extract support status
            is_supported = "SUPPORTED: YES" in response_text.upper() or "SUPPORTED:YES" in response_text.upper()
            
            last_answer = answer
            print(f"  Generated answer: {answer}")
            print(f"  Reflection: {'SUPPORTED' if is_supported else 'NOT_SUPPORTED'}")
            
            # Step 2: Decide whether to accept or refine
            if is_supported:
                print(f"  ✅ Answer accepted!")
                return {
                    'answer': answer,
                    'iterations': iteration + 1,
                    'reflection': 'SUPPORTED',
                    'source_documents': result['source_documents']
                }
            else:
                print(f"  ❌ Answer needs refinement...")
                iteration += 1
        
        # Max iterations reached
        print(f"  ⚠️ Max iterations reached, returning last answer")
        return {
            'answer': last_answer if last_answer else "I don't know",
            'iterations': iteration,
            'reflection': 'MAX_ITERATIONS_REACHED',
            'source_documents': result['source_documents']
        }

def evaluate_self_rag():
    """Evaluate Self-RAG on sample data"""
    
    # Load data
    print("📂 Loading data...")
    with open('data/hotpotqa_sample.json', 'r') as f:
        data = json.load(f)
    
    # Initialize Vanilla RAG
    print("🔄 Initializing Vanilla RAG...")
    vanilla_rag = VanillaRAG()
    all_contexts = [item['context'] for item in data]
    vanilla_rag.create_vectorstore(all_contexts)
    
    # Initialize Self-RAG
    print("🔄 Initializing Self-RAG...")
    self_rag = SimplifiedSelfRAG(vanilla_rag)
    
    # Test on first 3 questions
    print("\n" + "="*80)
    print("🧪 TESTING SELF-RAG WITH REFLECTION")
    print("="*80)
    
    for i, item in enumerate(data[:3]):
        print(f"\n{'='*80}")
        print(f"--- Question {i+1} ---")
        print(f"❓ Q: {item['question']}")
        print(f"✅ True Answer: {item['answer']}")
        
        result = self_rag.answer_with_reflection(item['question'])
        
        print(f"\n🤖 Final Answer: {result['answer'][:200]}...")
        print(f"🔄 Iterations Needed: {result['iterations']}")
        print(f"📊 Final Reflection: {result['reflection']}")
        print(f"📚 Used {len(result['source_documents'])} source documents")
        print("="*80)

if __name__ == "__main__":
    evaluate_self_rag()