import sys
import os

# Ensure the workspace directory is in python path
sys.path.append(os.path.abspath('.'))

from rag_engine import RAGEngine
from langchain_core.documents import Document

def run_test():
    print("Step 1: Testing imports...")
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        print("Imports successful!")
    except Exception as e:
        print(f"Import error: {e}")
        sys.exit(1)

    print("\nStep 2: Initializing Hugging Face embeddings...")
    try:
        # This will download the sentence-transformer model if not present
        embeddings = RAGEngine.get_embeddings("huggingface")
        print("Hugging Face embeddings initialized successfully!")
    except Exception as e:
        print(f"Embeddings error: {e}")
        sys.exit(1)

    print("\nStep 3: Creating mock documents...")
    docs = [
        Document(page_content="Antigravity is a powerful agentic AI coding assistant designed by the Google DeepMind team.", metadata={"source": "test_doc.pdf", "page": 0}),
        Document(page_content="Streamlit is an open-source Python library that makes it easy to create custom web apps for machine learning and data science.", metadata={"source": "test_doc.pdf", "page": 1}),
        Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors, developed by Meta's AI research group.", metadata={"source": "test_doc.pdf", "page": 2})
    ]
    print(f"Created {len(docs)} mock documents.")

    print("\nStep 4: Building FAISS index...")
    try:
        db = RAGEngine.create_vector_store(docs, "huggingface")
        print("FAISS vector store created successfully!")
    except Exception as e:
        print(f"FAISS creation error: {e}")
        sys.exit(1)

    print("\nStep 5: Performing similarity query...")
    try:
        query = "Who designed Antigravity?"
        results = db.similarity_search(query, k=1)
        print(f"Query: '{query}'")
        print(f"Top result content: '{results[0].page_content}'")
        print(f"Top result metadata: {results[0].metadata}")
        
        # Verify the top result is the correct one
        assert "DeepMind" in results[0].page_content, "Search failed to retrieve the most relevant document!"
        print("\nVerification Test PASSED successfully!")
    except Exception as e:
        print(f"Query/Assertion error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
