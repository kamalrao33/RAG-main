"""
RAGPipeline — wires together:
  PDF → Chunker → Embedder → VectorStore → Reranker → Generator
"""
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import CrossEncoder

from app.config import Settings
from app.chunker import pdf_to_chunks, Chunk
from app.embedder import Embedder
from app.retriever import BaseStore, SearchResult, build_store
from app.generator import Generator
from app.evaluator import evaluate_rag_response

logger = logging.getLogger(__name__)


# ── Response dataclasses ───────────────────────────────────────────────────────

@dataclass
class UploadResult:
    doc_id: str
    filename: str
    pages: int
    chunks: int
    elapsed_ms: float


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[dict]          # [{filename, page, text, score}]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    model: str
    evaluation: dict             # Real-time evaluation metrics


# ── Document registry (in-memory; swap for DB in production) ──────────────────

_registry: dict[str, dict] = {}   # doc_id → {filename, pages, chunks, path}


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder  = Embedder(settings)
        self.store: BaseStore = build_store(settings, self.embedder.dim)
        self.generator = Generator(settings)
        self._reranker: CrossEncoder | None = None

    # ── Reranker (lazy load) ────────────────────────────────────────────────────
    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info("Loading cross-encoder reranker")
            self._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
        return self._reranker

    # ── Ingest ─────────────────────────────────────────────────────────────────
    def ingest(self, pdf_path: str, filename: str) -> UploadResult:
        """
        Full ingestion pipeline:
        1. Extract text from every PDF page
        2. Clean and chunk the text
        3. Embed all chunks in batches
        4. Store vectors + metadata in the vector store
        """
        t0 = time.perf_counter()
        doc_id = str(uuid.uuid4())

        # 1 + 2: extract & chunk
        chunks = pdf_to_chunks(
            pdf_path=pdf_path,
            doc_id=doc_id,
            filename=filename,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"No text could be extracted from '{filename}'")

        # 3: embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)

        # 4: store
        self.store.add(chunks, embeddings)
        self.store.save()

        # Register document
        pages = max(c.page for c in chunks)
        _registry[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "pages": pages,
            "chunks": len(chunks),
            "path": pdf_path,
        }

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "Ingested '%s': %d pages, %d chunks in %.0fms",
            filename, pages, len(chunks), elapsed_ms,
        )
        return UploadResult(
            doc_id=doc_id,
            filename=filename,
            pages=pages,
            chunks=len(chunks),
            elapsed_ms=elapsed_ms,
        )

    # ── Query ──────────────────────────────────────────────────────────────────
    async def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """
        Full query pipeline:
        1. Embed the question
        2. ANN search in vector store (retrieval)
        3. Cross-encoder reranking (optional)
        4. LLM generation with retrieved context
        """
        top_k = top_k or self.settings.retrieval_top_k

        # 1: embed query
        q_vec = self.embedder.encode_one(question)

        # 2: retrieve
        t_ret = time.perf_counter()
        candidates = self.store.search(q_vec, top_k=top_k)
        retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 1)

        if not candidates:
            return QueryResult(
                question=question,
                answer="No relevant content found. Please upload a PDF first.",
                sources=[],
                retrieval_ms=retrieval_ms,
                rerank_ms=0.0,
                generation_ms=0.0,
                total_ms=retrieval_ms,
                model=self.generator.model_name,
                evaluation={
                    "faithfulness": 100.0,
                    "answer_relevance": 0.0,
                    "context_relevance": 0.0,
                    "citation_precision": 100.0,
                    "overall_score": 0.0,
                    "rating": "No Context",
                    "cited_pages": [],
                },
            )

        # 3: rerank
        rerank_ms = 0.0
        if self.settings.reranker_enabled and len(candidates) > 1:
            t_rr = time.perf_counter()
            reranker = self._get_reranker()
            pairs = [(question, r.text) for r in candidates]
            scores = reranker.predict(pairs, show_progress_bar=False)
            for result, score in zip(candidates, scores):
                result.rerank_score = float(score)
            candidates = sorted(
                candidates, key=lambda r: r.rerank_score, reverse=True
            )[: self.settings.reranker_top_k]
            rerank_ms = round((time.perf_counter() - t_rr) * 1000, 1)

        # 4: generate
        answer, generation_ms = await self.generator.generate(
            question, candidates
        )

        total_ms = round(retrieval_ms + rerank_ms + generation_ms, 1)

        # 5: evaluate response quality
        evaluation = evaluate_rag_response(
            question=question,
            answer=answer,
            candidates=candidates,
            embedder=self.embedder,
        )

        return QueryResult(
            question=question,
            answer=answer,
            sources=[{
                "filename": r.filename,
                "page": r.page,
                "text": r.text[:300] + ("…" if len(r.text) > 300 else ""),
                "score": round(r.rerank_score or r.score, 4),
            } for r in candidates],
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            model=self.generator.model_name,
            evaluation=evaluation,
        )

    async def stream_query(self, question: str, top_k: int | None = None):
        """Yield answer tokens for streaming response."""
        top_k = top_k or self.settings.retrieval_top_k
        q_vec = self.embedder.encode_one(question)
        candidates = self.store.search(q_vec, top_k=top_k)

        if not candidates:
            yield "No relevant content found. Please upload a PDF first."
            return

        if self.settings.reranker_enabled and len(candidates) > 1:
            reranker = self._get_reranker()
            pairs = [(question, r.text) for r in candidates]
            scores = reranker.predict(pairs, show_progress_bar=False)
            for r, s in zip(candidates, scores):
                r.rerank_score = float(s)
            candidates = sorted(
                candidates, key=lambda r: r.rerank_score, reverse=True
            )[: self.settings.reranker_top_k]

        async for token in self.generator.stream(question, candidates):
            yield token

    # ── Management ─────────────────────────────────────────────────────────────
    def delete_document(self, doc_id: str) -> int:
        deleted = self.store.delete_doc(doc_id)
        self.store.save()
        _registry.pop(doc_id, None)
        return deleted

    def list_documents(self) -> list[dict]:
        return list(_registry.values())

    def stats(self) -> dict:
        return {
            "total_documents": len(_registry),
            "total_chunks": self.store.count(),
            "vector_store": self.settings.vector_store,
            "embedding_model": self.embedder._model_name,
            "llm": self.generator.model_name,
        }
