import os
import tempfile
import streamlit as st
from rag_engine import RAGEngine


def format_rag_error(error: Exception, provider: str) -> str:
    message = str(error)
    normalized = message.lower()

    if "api_key_invalid" in normalized or "api key not valid" in normalized:
        provider_name = provider.lower()
        return (
            f"{provider_name.title()} rejected the API key. "
            "Verify that the key is active, copied without extra spaces, and issued for the correct provider. "
            "If you want to avoid cloud auth issues, switch the model provider to HuggingFace (Local, Free)."
        )

    if "api key" in normalized and "invalid" in normalized:
        return (
            f"{provider.title()} rejected the API key. "
            "Check the key value in the sidebar and try again."
        )

    return message

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DocuMind RAG - PDF QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Aesthetics
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        /* Apply fonts */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        /* Main Container Styling */
        .main {
            background-color: #0b0f19;
        }

        /* Titles and Headers */
        h1, h2, h3 {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }
        
        .hero-title {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.15rem;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Glassmorphic Cards */
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        /* Source Card Styling */
        .source-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .source-card:hover {
            border-color: rgba(99, 102, 241, 0.5);
            background: rgba(30, 41, 59, 0.7);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
            transform: translateY(-2px);
        }
        .source-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: #818cf8;
            margin-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 4px;
        }
        .source-content {
            font-size: 0.9rem;
            color: #cbd5e1;
            line-height: 1.5;
            font-style: italic;
        }
        .source-badge {
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .score-badge {
            background: rgba(236, 72, 153, 0.15);
            color: #f9a8d4;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        /* Status Pill */
        .status-ready {
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid rgba(16, 185, 129, 0.2);
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .status-empty {
            background: rgba(239, 68, 68, 0.1);
            color: #f87171;
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "active_embeddings_provider" not in st.session_state:
    st.session_state.active_embeddings_provider = None

inject_custom_css()

# Sidebar Layout
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/documents.png", width=80)
    st.title("Settings & Documents")
    
    # 1. EMBEDDING PROVIDER SETUP
    st.subheader("1. Embedding Model (Index)")
    embed_provider = st.selectbox(
        "Embedding Provider",
        options=["HuggingFace (Local, Free)", "Gemini (Cloud)", "OpenAI (Cloud)"],
        help="Select model to compute document embeddings. HuggingFace runs 100% locally and free."
    )
    
    embed_api_key = ""
    embed_provider_code = ""
    if "HuggingFace" in embed_provider:
        embed_provider_code = "huggingface"
    elif "Gemini" in embed_provider:
        embed_provider_code = "gemini"
        embed_api_key = st.text_input("Gemini API Key (Embed)", type="password", help="Enter Google AI Studio Key")
    elif "OpenAI" in embed_provider:
        embed_provider_code = "openai"
        embed_api_key = st.text_input("OpenAI API Key (Embed)", type="password", help="Enter OpenAI API Key")

    # 2. CHAT LLM PROVIDER SETUP
    st.subheader("2. Chat LLM Model")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["Gemini", "OpenAI", "HuggingFace"],
        index=0,
        help="Select LLM provider for generating final responses."
    )

    llm_api_key = ""
    llm_model_name = ""
    
    if llm_provider == "OpenAI":
        llm_api_key = st.text_input("OpenAI API Key (LLM)", type="password", help="Enter your OpenAI API key")
        llm_model_name = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        )
    elif llm_provider == "Gemini":
        llm_api_key = st.text_input("Gemini API Key (LLM)", type="password", help="Enter your Google AI Studio API key")
        llm_model_name = st.selectbox(
            "LLM Model",
            options=["gemini-3.5-flash", "gemini-3.1-pro", "gemini-3.1-flash-lite", "gemini-2.5-flash"]
        )
    elif llm_provider == "HuggingFace":
        llm_api_key = st.text_input("Hugging Face API Token", type="password", help="Enter HF Hub Token")
        llm_model_name = st.text_input(
            "HF Repo ID",
            value="meta-llama/Meta-Llama-3-8B-Instruct"
        )

    # 3. HYPERPARAMETERS
    st.subheader("3. RAG Parameters")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=4000, value=800, step=100)
    with col2:
        chunk_overlap = st.number_input("Overlap", min_value=0, max_value=1000, value=150, step=50)

    temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k = st.slider("Source Chunks to Retrieve", min_value=1, max_value=10, value=4, step=1)

    # 4. DOCUMENT UPLOAD
    st.subheader("4. Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDFs to index in the knowledge store."
    )

    # Document Processing Logic
    if uploaded_files:
        # Check if the user is attempting to change embedding models mid-session
        if st.session_state.vector_store is not None and st.session_state.active_embeddings_provider != embed_provider_code:
            st.warning("⚠️ You are trying to add files with a different Embedding Provider. Please clear the database index first.")
        else:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            if new_files:
                # Check for API keys if cloud embedding model selected
                if embed_provider_code in ["openai", "gemini"] and not embed_api_key:
                    st.error(f"Please enter the Embedding API Key for {embed_provider} before indexing.")
                else:
                    with st.spinner("Processing & indexing PDFs..."):
                        all_docs = []
                        for uploaded_file in new_files:
                            # Write to a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name

                            try:
                                docs = RAGEngine.load_pdf(tmp_path)
                                # Fix metadata to show original filename
                                for doc in docs:
                                    doc.metadata["source"] = uploaded_file.name
                                
                                st.session_state.total_pages += len(docs)
                                all_docs.extend(docs)
                                st.session_state.processed_files.add(uploaded_file.name)
                            finally:
                                os.unlink(tmp_path)

                        if all_docs:
                            chunks = RAGEngine.split_documents(all_docs, chunk_size, chunk_overlap)
                            st.session_state.total_chunks += len(chunks)

                            # Build/update FAISS index
                            try:
                                if st.session_state.vector_store is None:
                                    st.session_state.vector_store = RAGEngine.create_vector_store(
                                        chunks, embed_provider_code, embed_api_key
                                    )
                                    st.session_state.active_embeddings_provider = embed_provider_code
                                else:
                                    st.session_state.vector_store.add_documents(chunks)
                                st.success("Successfully processed and indexed PDFs!")
                            except Exception as e:
                                st.error(f"Error initializing embeddings / vector store: {e}")
                                # Rollback processed files
                                for f in new_files:
                                    st.session_state.processed_files.discard(f.name)

    # Status Dashboard
    st.subheader("5. Database Status")
    if st.session_state.vector_store is not None:
        st.markdown(f'<div class="status-ready">Ready (Indexed)</div>', unsafe_allow_html=True)
        st.markdown(f"🧬 **Embedding Model:** {st.session_state.active_embeddings_provider.upper()}")
        st.markdown(f"📊 **Files Indexed:** {len(st.session_state.processed_files)}")
        st.markdown(f"📄 **Total Pages:** {st.session_state.total_pages}")
        st.markdown(f"🧩 **Total Chunks:** {st.session_state.total_chunks}")
        
        # Reset Button
        if st.button("Clear Database Index", type="secondary"):
            st.session_state.vector_store = None
            st.session_state.processed_files = set()
            st.session_state.total_pages = 0
            st.session_state.total_chunks = 0
            st.session_state.chat_history = []
            st.session_state.active_embeddings_provider = None
            st.rerun()
    else:
        st.markdown(f'<div class="status-empty">Empty</div>', unsafe_allow_html=True)
        st.caption("Upload PDFs to initialize the vector database.")


# Main Application Interface
st.markdown('<div class="hero-title">DocuMind RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Ask questions, locate facts, and extract insight from your PDF files instantly</div>', unsafe_allow_html=True)

# Chat Area Setup
if st.session_state.vector_store is None:
    # Welcome Layout
    st.markdown("""
        <div class="glass-card">
            <h3>Welcome to DocuMind!</h3>
            <p>Get started with document-driven Q&A in three simple steps:</p>
            <ol>
                <li>Choose your <b>Embedding Model</b> (select <i>HuggingFace (Local, Free)</i> to avoid API cost & quota errors).</li>
                <li>Upload your PDF documents in the sidebar uploader.</li>
                <li>Set up your preferred <b>Chat LLM Model</b> (e.g. Gemini Pro/Flash or OpenAI) and API key to start querying.</li>
            </ol>
            <hr style="opacity: 0.15; margin: 20px 0;">
            <p style="font-size: 0.9rem; color: #94a3b8;">
                💡 <b>Developer Tip:</b> Google Gemini currently offers a generous free-tier API key in Google AI Studio. Combined with local HuggingFace embeddings, you can run this entire RAG system for 100% free!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Capabilities & Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.1); border-radius: 12px; padding: 20px; height: 100%;">
                <h4 style="color: #818cf8; margin-top: 0;">🔍 Semantic Retrieval</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">Uses state-of-the-art dense embeddings to fetch the exact passages containing relevant context, regardless of exact keyword matches.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background: rgba(168, 85, 247, 0.05); border: 1px solid rgba(168, 85, 247, 0.1); border-radius: 12px; padding: 20px; height: 100%;">
                <h4 style="color: #c084fc; margin-top: 0;">📍 Source Attribution</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">Every answer lists its source references with page numbers and proximity metrics, keeping model generation grounded and verifiable.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="background: rgba(236, 72, 153, 0.05); border: 1px solid rgba(236, 72, 153, 0.1); border-radius: 12px; padding: 20px; height: 100%;">
                <h4 style="color: #f472b6; margin-top: 0;">🔐 Hybrid Model Selection</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">Mix and match embedding models and LLM providers. Keep embeddings local and secure while leveraging advanced LLMs for summarization.</p>
            </div>
        """, unsafe_allow_html=True)

else:
    # Render Active Chat
    st.subheader("Converse with your Knowledge Base")
    
    # Reset chat history button
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔍 Inspected Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">
                                    <span>📄 {src['source']} (Page {src['page']})</span>
                                    <span class="score-badge">Distance Score: {src['score']:.4f}</span>
                                </div>
                                <div class="source-content">"...{src['content']}..."</div>
                            </div>
                        """, unsafe_allow_html=True)

    # Handle user query
    user_query = st.chat_input("Ask a question about the uploaded documents...")
    if user_query:
        # Check API Key validity for Chat LLM
        if llm_provider in ["OpenAI", "Gemini"] and not llm_api_key:
            st.warning(f"Please provide an API key in the sidebar for {llm_provider} before asking questions.")
        elif llm_provider == "HuggingFace" and not llm_api_key:
            st.warning("Please provide a Hugging Face API Token in the sidebar before asking questions.")
        else:
            # Render user message
            with st.chat_message("user"):
                st.write(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Retrieve, synthesize, and render assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document context and synthesizing response..."):
                    try:
                        result = RAGEngine.query_rag(
                            question=user_query,
                            vector_store=st.session_state.vector_store,
                            provider=llm_provider,
                            model_name=llm_model_name,
                            api_key=llm_api_key,
                            temperature=temperature,
                            top_k=top_k
                        )
                        
                        answer = result["answer"]
                        sources = result["sources"]
                        
                        st.write(answer)
                        
                        with st.expander("🔍 Inspected Sources"):
                            for src in sources:
                                st.markdown(f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span>📄 {src['source']} (Page {src['page']})</span>
                                            <span class="score-badge">Distance Score: {src['score']:.4f}</span>
                                        </div>
                                        <div class="source-content">"...{src['content']}..."</div>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Error querying RAG system: {format_rag_error(e, llm_provider)}")
