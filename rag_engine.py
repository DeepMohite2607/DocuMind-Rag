import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Provider imports
# Note: We import dynamically inside functions or safely here to avoid crashes if imports fail.
# We will catch import errors gracefully in case a provider SDK isn't fully set up yet.

class RAGEngine:
    def __init__(self):
        pass

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Loads and parses a PDF file using PyPDFLoader."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

    @staticmethod
    def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Splits documents into smaller chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        return splitter.split_documents(docs)

    @staticmethod
    def get_embeddings(provider: str, api_key: Optional[str] = None):
        """Initializes and returns the appropriate embedding model based on provider."""
        provider = provider.lower()
        if api_key:
            api_key = api_key.strip()

        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API Key is required for OpenAI embeddings.")
            os.environ["OPENAI_API_KEY"] = api_key
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=api_key)
            
        elif provider == "gemini":
            if not api_key:
                raise ValueError("Google Gemini API Key is required for Gemini embeddings.")
            os.environ["GOOGLE_API_KEY"] = api_key
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            # Check model parameter if needed, default is fine
            return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
            
        elif provider == "huggingface":
            # Uses sentence-transformers locally
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    @classmethod
    def create_vector_store(cls, docs: List[Document], provider: str, api_key: Optional[str] = None) -> FAISS:
        """Creates a FAISS vector store from documents using selected embedding provider."""
        embeddings = cls.get_embeddings(provider, api_key)
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    @classmethod
    def load_vector_store(cls, folder_path: str, provider: str, api_key: Optional[str] = None) -> FAISS:
        """Loads a FAISS vector store from local folder."""
        embeddings = cls.get_embeddings(provider, api_key)
        # allow_dangerous_deserialization is required for loading FAISS index files locally
        return FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)

    @staticmethod
    def get_llm(provider: str, model_name: str, api_key: Optional[str] = None, temperature: float = 0.2):
        """Initializes and returns the LLM based on provider and configurations."""
        provider = provider.lower()
        if api_key:
            api_key = api_key.strip()

        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API Key is required.")
            os.environ["OPENAI_API_KEY"] = api_key
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)
            
        elif provider == "gemini":
            if not api_key:
                raise ValueError("Google Gemini API Key is required.")
            os.environ["GOOGLE_API_KEY"] = api_key
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
            
        elif provider == "huggingface":
            if not api_key:
                raise ValueError("Hugging Face Hub API Token is required to use HuggingFace inference.")
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            from langchain_huggingface import HuggingFaceEndpoint
            # E.g. "meta-llama/Meta-Llama-3-8B-Instruct" or "mistralai/Mistral-7B-Instruct-v0.2"
            return HuggingFaceEndpoint(
                repo_id=model_name,
                temperature=temperature,
                huggingfacehub_api_token=api_key,
                max_new_tokens=512,
                top_p=0.95
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Formats a list of documents into a single context string with metadata markings."""
        formatted = []
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "Uploaded PDF"))
            page = doc.metadata.get("page", 0) + 1  # convert 0-indexed to 1-indexed
            formatted.append(f"--- Chunk {i+1} [Source: {source}, Page: {page}] ---\n{doc.page_content}\n")
        return "\n".join(formatted)

    @classmethod
    def query_rag(cls, question: str, vector_store: FAISS, provider: str, model_name: str, 
                  api_key: Optional[str] = None, temperature: float = 0.2, top_k: int = 4) -> Dict[str, Any]:
        """Queries the RAG pipeline. Returns response text and retrieved source documents with metadata."""
        # 1. Retrieve sources
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retriever.invoke(question)
        
        # 2. Get similarity scores if possible
        # FAISS search_with_score returns list of (doc, score) tuples.
        # Score is L2 distance, where smaller is closer.
        raw_docs_with_scores = vector_store.similarity_search_with_score(question, k=top_k)
        
        # 3. Setup LLM
        llm = cls.get_llm(provider, model_name, api_key, temperature)
        
        # 4. Define Prompt
        # Let's craft a prompt that instructs the LLM to use the source guidelines.
        template = """You are a helpful and highly accurate document assistant.
Answer the user's question based strictly on the provided context chunks.
If the context does not contain the answer, state that you do not know or that the information is not present in the documents. Do not make up facts or hallucinate.

Ensure you cite the source file and page numbers when answering, matching the format '[Source: filename, Page: page_num]' from the context.

Context:
{context}

Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # 5. Build and execute chain
        formatted_context = cls.format_docs(retrieved_docs)
        
        chain = (
            {"context": lambda x: formatted_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        # 6. Process sources details for UI
        sources = []
        for doc, score in raw_docs_with_scores:
            sources.append({
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "Uploaded PDF")),
                "page": doc.metadata.get("page", 0) + 1,
                "score": float(score)  # L2 distance score
            })
            
        return {
            "answer": answer,
            "sources": sources
        }
