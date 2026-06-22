# 📚 DocuMind-RAG

An AI-powered Document Question Answering System built using **Retrieval-Augmented Generation (RAG)**. Upload your documents, ask questions in natural language, and receive context-aware answers generated from the document content.

## 🚀 Overview

DocuMind-RAG enables users to interact with their documents using Large Language Models (LLMs). Instead of relying solely on the model's internal knowledge, the system retrieves relevant document chunks and provides them as context to generate accurate, grounded responses.

This helps reduce hallucinations and improves answer reliability.


## ✨ Features

- 📄 Upload and process PDF documents
- 🔍 Semantic search using vector embeddings
- 🤖 Context-aware question answering
- 📚 Retrieval-Augmented Generation (RAG)
- ⚡ Fast document retrieval with vector database
- 🎯 Accurate responses based on uploaded documents
- 💬 Interactive chat interface
- 🔄 Real-time document querying

🛠️ Tech Stack
Programming Language
Python

AI & Machine Learning
LangChain
OpenAI / Gemini
Sentence Transformers

Vector Database
FAISS / ChromaDB

Backend
FastAPI

Frontend
Streamlit

Libraries
Pandas
NumPy
PyPDF
Transformers

📂 Project Structure
DocuMind-RAG/
│
├── data/                 # Documents
├── embeddings/           # Vector embeddings
├── src/
│   ├── ingestion.py      # Document processing
│   ├── retrieval.py      # Vector search
│   ├── rag_pipeline.py   # RAG workflow
│   └── app.py            # Main application
│
├── requirements.txt
├── .env
└── README.md

⚙️ Installation
1. Clone Repository
git clone https://github.com/DeepMohite2607/DocuMind-Rag.git
cd DocuMind-Rag
2. Create Virtual Environment
python -m venv venv

Activate environment:

venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure Environment Variables

Create a .env file:

OPENAI_API_KEY=your_api_key
GOOGLE_API_KEY=your_api_key

▶️ Run Application

For Streamlit:

streamlit run app.py

For FastAPI:

uvicorn app:app --reload

🔄 RAG Pipeline
Upload document
Extract text from PDF
Split text into chunks
Generate embeddings
Store embeddings in vector database
User asks a question
Retrieve relevant chunks
Pass chunks + query to LLM
Generate context-aware response

📊 Use Cases
Research Paper Analysis
Enterprise Knowledge Base
Resume Question Answering
Legal Document Search
Educational Content Assistant
Policy & Compliance Search

📈 Future Enhancements
Multi-PDF support
Conversation memory
Source citation tracking
Hybrid search (BM25 + Vector Search)
Multi-modal RAG
LangGraph agent workflow

🤝 Contributing

Contributions are welcome!

Fork the repository
Create a feature branch
Commit changes
Push to branch
Open a Pull Request

## 🏗️ Architecture

```text
User Query
     │
     ▼
Embedding Model
     │
     ▼
Vector Database
     │
     ▼
Relevant Document Chunks
     │
     ▼
Large Language Model (LLM)
     │
     ▼
Generated Answer
