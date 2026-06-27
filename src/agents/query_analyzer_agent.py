from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import os

load_dotenv()

class QueryAnalyzerAgent:
    """Agent that analyzes query complexity and decomposes multi-hop questions"""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query analysis expert. Analyze questions and determine:
1. Whether they require multi-hop reasoning (information from multiple sources)
2. How to decompose complex questions into simpler sub-questions
3. The complexity level

Examples:
- "What is the capital of France?" → Simple, no decomposition
- "Who is the spouse of the director of Titanic?" → Multi-hop: (1) Who directed Titanic? (2) Who is their spouse?
- "Which is taller, Eiffel Tower or Statue of Liberty?" → Comparison: (1) Height of Eiffel Tower? (2) Height of Statue of Liberty?

Respond ONLY in valid JSON format."""),
            ("user", """Analyze this question: {question}

Respond in this exact JSON format (no markdown, just JSON):
{{
  "is_multi_hop": true or false,
  "complexity": "simple" or "medium" or "complex",
  "sub_questions": ["sub-question 1", "sub-question 2"],
  "reasoning": "brief explanation"
}}""")
        ])
        
    def analyze(self, question):
        """Analyze a query and return structured analysis"""
        chain = self.prompt | self.llm
        response = chain.invoke({"question": question})
        
        try:
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            if content.startswith("json"):
                content = content[4:].strip()
            
            result = json.loads(content)
            return result
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            return {
                "is_multi_hop": False,
                "complexity": "simple",
                "sub_questions": [question],
                "reasoning": "Failed to parse, treating as simple"
            }

def test_query_analyzer():
    """Test Query Analyzer on HotpotQA questions"""
    
    # Load data
    with open('data/hotpotqa_sample.json', 'r') as f:
        data = json.load(f)
    
    agent = QueryAnalyzerAgent()
    
    print("="*80)
    print("🧪 TESTING QUERY ANALYZER AGENT")
    print("="*80)
    
    for i, item in enumerate(data[:5]):
        print(f"\n--- Question {i+1} ---")
        print(f"Question: {item['question']}")
        print(f"Type: {item['type']}")  # HotpotQA labels (bridge/comparison)
        
        result = agent.analyze(item['question'])
        
        print(f"\n🔍 Analysis:")
        print(f"  Multi-hop: {result['is_multi_hop']}")
        print(f"  Complexity: {result['complexity']}")
        
        if result['is_multi_hop']:
            print(f"  Sub-questions:")
            for j, sub_q in enumerate(result['sub_questions'], 1):
                print(f"    {j}. {sub_q}")
        
        print(f"  Reasoning: {result['reasoning']}")
        print("-"*80)

if __name__ == "__main__":
    test_query_analyzer()