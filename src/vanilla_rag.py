from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.answer_extraction import extract_answer_string
import json
from dotenv import load_dotenv
import os

load_dotenv()

class VanillaRAG:
    def __init__(self, model_name="llama-3.3-70b-versatile"):  # UPDATED MODEL
        """Initialize Vanilla RAG system with Groq"""
        print("[INIT] Initializing Vanilla RAG with Groq...")
        
        # Use Groq LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        
        # Use free HuggingFace embeddings
        print("[LOADING] Loading embeddings model (this may take 30 seconds)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = None
        print("[OK] RAG initialized!")
        
    def create_vectorstore(self, contexts):
        """Create FAISS vectorstore from context passages"""
        print("[PROCESSING] Creating vector store...")
        
        # Flatten all context passages into documents
        documents = []
        for ctx in contexts:
            # HotpotQA context structure: list of lists or list of strings
            for item in ctx:
                # Handle nested lists
                if isinstance(item, list):
                    for sentence in item:
                        if isinstance(sentence, str) and sentence.strip():
                            documents.append(Document(page_content=sentence))
                elif isinstance(item, str) and item.strip():
                    documents.append(Document(page_content=item))
        
        print(f"[INFO] Total passages: {len(documents)}")
        
        # Create vectorstore
        print("[PROCESSING] Building FAISS index...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("[OK] Vector store created!")
        
    def answer_question(self, question):
        """Answer a question using RAG"""
        if self.vectorstore is None:
            raise ValueError("[ERROR] Vector store not initialized!")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # IMPROVED prompt - explicitly ask for concise answer only
        template = """Use the following context to answer the question directly and concisely.

IMPORTANT:
1. Provide ONLY the answer itself, nothing more
2. Do NOT explain or provide reasoning
3. If multiple valid answers exist, provide the most specific one
4. If the answer requires a full phrase, keep it as short as possible
5. If you cannot find the answer in the context, respond with exactly: "I don't know"

Context: {context}

Question: {question}

Answer (one or two words maximum):"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Format retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create chain using LCEL (LangChain Expression Language)
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Get answer
        print(f"  🔍 Retrieving relevant passages...")
        retrieved_docs = retriever.invoke(question)
        
        print(f"  🤖 Generating answer...")
        raw_answer = rag_chain.invoke(question)
        
        # Extract and clean the answer
        answer = extract_answer_string(raw_answer)
        
        return {
            'answer': answer,
            'raw_answer': raw_answer,
            'source_documents': [doc.page_content for doc in retrieved_docs]
        }

def evaluate_vanilla_rag():
    """Test Vanilla RAG on sample data"""
    # Load data
    print("📂 Loading data...")
    with open('data/hotpotqa_sample.json', 'r') as f:
        data = json.load(f)
    
    # Initialize RAG
    rag = VanillaRAG()
    
    # Create vectorstore
    all_contexts = [item['context'] for item in data]
    rag.create_vectorstore(all_contexts)
    
    # Test on first 3 questions
    print("\n" + "="*80)
    print("🧪 TESTING VANILLA RAG WITH GROQ + VECTOR SEARCH")
    print("="*80)
    
    for i, item in enumerate(data[:3]):
        print(f"\n--- Question {i+1} ---")
        print(f"❓ Q: {item['question']}")
        print(f"✅ True Answer: {item['answer']}")
        
        result = rag.answer_question(item['question'])
        
        print(f"🤖 RAG Answer: {result['answer']}")
        print(f"\n📚 Retrieved {len(result['source_documents'])} passages:")
        for j, doc in enumerate(result['source_documents'][:3]):
            print(f"  {j+1}. {doc[:100]}...")
        print("-"*80)

if __name__ == "__main__":
    evaluate_vanilla_rag()