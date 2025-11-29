import os
import dotenv

# Load environment variables first
dotenv.load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set in environment variables or .env file.")
    exit(1)

# Import after loading env vars because rag_agent initializes embeddings at module level
from rag_agent import rag_node

def test_rag():
    print("Testing RAG Agent...")
    
    questions = [
        "How do I create a profile?",
        "What is video spoofing?",
        "How to use the cookie robot?",
        "How to create proxy?",
    ]
    
    for q in questions:
        print(f"\nQuestion: {q}")
        state = {"question": q}
        try:
            result = rag_node(state)
            print(f"Answer: {result['final_answer']}")
            print(f"Sources: {result.get('rag_context')}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_rag()
