# PDF RAG Pipeline

A robust, production-ready Retrieval-Augmented Generation (RAG) pipeline designed for indexing PDF files (both small and large) and answering user queries with precise page-level citations.

Built using **FastAPI**, **LangChain**, and **SentenceTransformers**, this pipeline supports pluggable vector stores, cross-encoder reranking, and choice of LLM backends (OpenAI, Anthropic, or local Ollama).

---

## Features

- **Document Ingestion**: Extracts text page-by-page from PDFs using `pdfplumber` (with `pypdf` fallback), cleans text, and splits it using a custom recursive character chunker.
- **Semantic Search**: Converts text chunks into 384-dimensional dense vectors using the CPU-friendly `BAAI/bge-small-en-v1.5` embeddings.
- **Pluggable Vector Databases**:
  - **FAISS**: Local in-memory vector store with a sidecar JSON file for chunk metadata.
  - **ChromaDB**: Local persistent embedded database.
  - **Pinecone**: Cloud-managed vector database.
- **Cross-Encoder Reranking**: Utilizes `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank retrieved context chunks, improving response accuracy.
- **LLM Generators**: Pluggable support for OpenRouter (`meta-llama/llama-3.3-70b-instruct`, `deepseek/deepseek-chat`, `anthropic/claude-3.5-sonnet`, etc.), OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-haiku`), or local Ollama (`llama3`).
- **Response Citations**: Prompt engineering forces the LLM to base its answers strictly on context and include page-level citations (e.g., `(p. 12)`).

---

## Directory Layout

```text
RAG Pipeline/
│
├── app/                      # Main package directory
│   ├── __init__.py           # Declares app package
│   ├── chunker.py            # PDF parser and recursive text chunker
│   ├── config.py             # Settings using pydantic-settings
│   ├── embedder.py           # Embeddings wrapper using SentenceTransformers
│   ├── generator.py          # LLM response generation via LangChain
│   ├── main.py               # FastAPI application definition and endpoints
│   ├── pipeline.py           # RAG coordinator pipeline class
│   └── retriever.py          # Vector store implementations (FAISS, Chroma, Pinecone)
│
├── env.example               # Template environment configuration file
├── requirements.txt          # Third-party Python package dependencies
├── test_all.py               # Complete Pytest test suite
└── README.md                 # Project documentation (this file)
```

---

## Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Configure Virtual Environment & Install Dependencies
From the project root directory, activate your virtual environment and install the required packages:

```bash
# If using the pre-configured virtual environment (Windows Powershell):
.\rag_env\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `env.example` to `.env` and fill in your keys and preferences:

```bash
cp env.example .env
```

Open `.env` and configure your desired settings:
- **LLM Provider**: Choose `openrouter`, `openai`, `anthropic`, or `ollama`. Add API keys accordingly.
- **Vector Store**: Choose `faiss`, `chroma`, or `pinecone`.

---

## Running the Application

### 1. Start the FastAPI Server
To launch the FastAPI development server, run:

```bash
uvicorn app.main:app --reload
```

The API documentation will be available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. Access the Interactive Dashboard
To access the beautiful, interactive drag-and-drop web client dashboard:
1. Open the [`index.html`](file:///c:/Users/DELL/OneDrive/Desktop/RAG%20Pipeline/index.html) file directly in any modern web browser.
2. The UI will automatically connect to your running backend at `http://localhost:8000`.
3. Drag & drop your PDF documents into the upload box and ask questions!

---

## API Endpoints

### 1. Ingest PDF
- **Endpoint**: `POST /upload`
- **Request**: Multipart Form Data with a `file` field containing your PDF.
- **Description**: Extracts, chunks, embeds, and indexes the document. Returns a unique `doc_id`.

### 2. Ask Question (Block Response)
- **Endpoint**: `POST /query`
- **Request Body**:
  ```json
  {
    "question": "What is the key takeaway of this document?",
    "top_k": 6
  }
  ```
- **Response**: The LLM's answer with citations, plus a list of retrieved source text excerpts (including page numbers).

### 3. Ask Question (Streaming Response)
- **Endpoint**: `POST /query/stream`
- **Request Body**: Same as `/query`.
- **Response**: A raw text/event stream yielding response tokens in real-time.

### 4. Delete Document
- **Endpoint**: `DELETE /documents/{doc_id}`
- **Description**: Deletes a document and its vectors from the database.

### 5. Statistics & Health
- **Endpoints**: `GET /stats`, `GET /health`
- **Description**: Monitor the index sizes, chosen models, and system health status.

---

## Running Tests

Verify the configuration and code correctness by running the full test suite:

```bash
python -m pytest test_all.py -v
```
