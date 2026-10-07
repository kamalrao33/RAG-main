"""
FastAPI application — PDF RAG Pipeline
Routes:
  POST  /upload           Upload a PDF
  POST  /query            Ask a question
  POST  /query/stream     Streaming answer
  GET   /documents        List indexed documents
  DELETE /documents/{id}  Remove a document
  GET   /stats            Index statistics
  GET   /health           Health check
"""
import logging
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ── Shared pipeline singleton ──────────────────────────────────────────────────
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised")
    return _pipeline


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    settings = get_settings()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Create directories
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_index_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Vector store: %s | LLM: %s", settings.vector_store,
                settings.llm_provider)

    # Warm up embedding model
    _pipeline = RAGPipeline(settings)
    _pipeline.embedder.load()

    logger.info("Pipeline ready. Chunks indexed: %d",
                _pipeline.store.count())
    yield

    # Save index on shutdown
    _pipeline.store.save()
    logger.info("Shutdown — index saved.")


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF RAG Pipeline",
    version="1.0.0",
    description=(
        "Upload any PDF — small or large — and ask questions about it. "
        "Powered by FAISS vector search, semantic chunking, cross-encoder "
        "reranking, and your choice of LLM."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Stamp processing time on every response
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(
        round((time.perf_counter() - t0) * 1000, 1)
    )
    return response


# ── Request / Response models ──────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    elapsed_ms: float
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3,
                          description="Question about the uploaded PDF")
    top_k: int = Field(6, ge=1, le=20,
                       description="Candidates to retrieve before reranking")


class SourceChunk(BaseModel):
    filename: str
    page: int
    text: str
    score: float


class EvaluationMetrics(BaseModel):
    faithfulness: float = Field(..., description="Groundedness / Faithfulness score (0-100%)")
    answer_relevance: float = Field(..., description="Answer relevance to question (0-100%)")
    context_relevance: float = Field(..., description="Context relevance of retrieved sources (0-100%)")
    citation_precision: float = Field(..., description="Citation accuracy and validity (0-100%)")
    overall_score: float = Field(..., description="Composite RAG quality score (0-100%)")
    rating: str = Field(..., description="Rating label: Excellent | Good | Fair | Needs Review")
    cited_pages: list[int] = Field(default_factory=list, description="List of page numbers cited in the answer")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    model: str
    evaluation: EvaluationMetrics


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    vector_store: str
    embedding_model: str
    llm: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "version": settings.app_version,
        "vector_store": settings.vector_store,
        "llm_provider": settings.llm_provider,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF (any size). The file is:
    1. Saved temporarily to disk
    2. Parsed page-by-page
    3. Chunked, embedded, and indexed
    Returns doc_id, page count, and chunk count.
    """
    settings = get_settings()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted (.pdf extension required)",
        )

    # Check file size
    max_bytes = settings.max_upload_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_mb} MB.",
        )

    # Save to temp file (pipeline needs a file path for pdfplumber)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pipeline = get_pipeline()
        result = pipeline.ingest(pdf_path=tmp_path, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed for '%s'", file.filename)
        raise HTTPException(status_code=500,
                            detail=f"Ingestion error: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        pages=result.pages,
        chunks=result.chunks,
        elapsed_ms=result.elapsed_ms,
        message=f"Successfully indexed {result.chunks} chunks from "
                f"{result.pages} pages.",
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Ask a question. The pipeline:
    1. Embeds your question
    2. Retrieves the top-k most similar chunks from FAISS
    3. Re-ranks with a cross-encoder
    4. Feeds top chunks to the LLM and returns a cited answer
    """
    pipeline = get_pipeline()
    if pipeline.store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF first.",
        )
    try:
        result = await pipeline.query(req.question, top_k=req.top_k)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceChunk(**s) for s in result.sources],
        retrieval_ms=result.retrieval_ms,
        rerank_ms=result.rerank_ms,
        generation_ms=result.generation_ms,
        total_ms=result.total_ms,
        model=result.model,
        evaluation=EvaluationMetrics(**result.evaluation),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Same as /query but streams the answer token-by-token.
    Useful for displaying the answer as it's being generated.
    """
    pipeline = get_pipeline()
    if pipeline.store.count() == 0:
        raise HTTPException(status_code=400,
                            detail="No documents indexed yet.")

    async def token_gen():
        try:
            async for token in pipeline.stream_query(req.question,
                                                     top_k=req.top_k):
                yield token
        except Exception as e:
            logger.exception("Stream failed")
            yield f"\n\n[ERROR: {e}]"

    return StreamingResponse(token_gen(), media_type="text/plain")


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all indexed PDF documents."""
    pipeline = get_pipeline()
    return [DocumentInfo(**d) for d in pipeline.list_documents()]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document and all its chunks from the index."""
    pipeline = get_pipeline()
    deleted = pipeline.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404,
                            detail=f"Document '{doc_id}' not found.")
    return {
        "doc_id": doc_id,
        "chunks_deleted": deleted,
        "status": "deleted",
    }


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Index statistics."""
    pipeline = get_pipeline()
    return StatsResponse(**pipeline.stats())


@app.get("/", response_class=FileResponse)
async def read_index():
    """Serve the interactive HTML dashboard."""
    index_path = Path(__file__).parent.parent / "index.html"
    return FileResponse(index_path)
