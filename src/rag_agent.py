import os
import glob
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import FakeEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from vllm_manager import get_vllm_manager
from monitoring import monitor

# Configuration
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs_Vision")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "vision_docs"

# Initialize embeddings
if os.getenv("OPENAI_API_KEY"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
elif HuggingFaceEmbeddings:
    print("⚠️ OpenAI API key not found. Using HuggingFace embeddings.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
else:
    print("⚠️ OpenAI API key not found and HuggingFace embeddings not available. Using FakeEmbeddings.")
    embeddings = FakeEmbeddings(size=1536)

def load_and_process_docs():
    """Load markdown files from the docs directory and split them."""
    print(f"Loading documents from {DOCS_DIR}...")
    
    # Find all markdown files
    md_files = glob.glob(os.path.join(DOCS_DIR, "**/*.md"), recursive=True)
    
    all_splits = []
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split by headers first
            md_header_splits = markdown_splitter.split_text(content)
            
            # Add metadata
            for doc in md_header_splits:
                doc.metadata["source"] = file_path
                
            # Then split by characters
            splits = text_splitter.split_documents(md_header_splits)
            all_splits.extend(splits)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"Processed {len(md_files)} files into {len(all_splits)} chunks.")
    return all_splits

def initialize_vector_store():
    """Initialize or load the Chroma vector store."""
    
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
    else:
        print("Creating new vector store...")
        splits = load_and_process_docs()
        if not splits:
            print("No documents found to index.")
            return None
            
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_DIR,
            collection_name=COLLECTION_NAME
        )
        
    return vectorstore

# Initialize global vector store
vectorstore = initialize_vector_store()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) if vectorstore else None

class LocalLLM(BaseChatModel):
    """Wrapper for VLLMManager to be compatible with LangChain"""
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        vllm_manager = get_vllm_manager()
        
        # Convert LangChain messages to dict format
        formatted_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
                
            formatted_messages.append({"role": role, "content": msg.content})
            
        response_text = vllm_manager.generate_response(
            messages=formatted_messages,
            temperature=kwargs.get("temperature", 0),
            stop=stop
        )
        
        generation = ChatGeneration(message=BaseMessage(content=response_text, type="ai"))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "vllm_local"

@monitor.trace(name="rag_agent")
def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for RAG processing.
    """
    question = state["question"]
    
    if not retriever:
        state["final_answer"] = "I'm sorry, but the documentation database is not available."
        return state
    
    # Define the RAG prompt
    template = """You are a helpful assistant for the 'Vision' browser automation tool.
    Use the following pieces of retrieved context to answer the question.
    If the context contains code examples, please include them in your answer formatted as code blocks.
    If you don't know the answer, just say that you don't know.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = LocalLLM(temperature=0)
    
    # Retrieve context
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    state["final_answer"] = answer
    
    # Deduplicate sources while preserving order
    seen = set()
    unique_sources = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)
            
    state["rag_context"] = unique_sources
    
    return state

if __name__ == "__main__":
    # Test the RAG agent
    test_state = {"question": "How do I create a profile?"}
    result = rag_node(test_state)
    print(f"Question: {test_state['question']}")
    print(f"Answer: {result['final_answer']}")
    print(f"Sources: {result.get('rag_context')}")
