"""
Tests for the PDF RAG Pipeline.
Run with:  pytest tests/ -v
"""
import io
import os
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

# Make sure tests never call real APIs
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("VECTOR_STORE", "faiss")
os.environ.setdefault("RERANKER_ENABLED", "false")


# ── Chunker tests ──────────────────────────────────────────────────────────────

class TestChunker:
    def test_clean_text_normalises_whitespace(self):
        from app.chunker import clean_text
        raw = "Hello   world\r\n\r\n\r\nNew paragraph"
        cleaned = clean_text(raw)
        assert "\r" not in cleaned
        assert "   " not in cleaned
        assert cleaned.count("\n") <= 2

    def test_split_text_chunk_size(self):
        from app.chunker import _split_text
        long_text = "word " * 500   # 2500 chars
        chunks = _split_text(long_text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 300   # a little slack for overlap

    def test_split_short_text_is_single_chunk(self):
        from app.chunker import _split_text
        chunks = _split_text("Short text.", chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_chunk_ids_are_unique(self):
        from app.chunker import Chunk
        chunks = [
            Chunk("id1", "text one", "doc1", "file.pdf", 1, 0),
            Chunk("id2", "text two", "doc1", "file.pdf", 1, 1),
        ]
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)


# ── Embedder tests ─────────────────────────────────────────────────────────────

class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        from app.config import Settings
        from app.embedder import Embedder
        settings = Settings(embedding_model="BAAI/bge-small-en-v1.5",
                            embedding_device="cpu")
        e = Embedder(settings)
        e.load()
        return e

    def test_encode_returns_correct_shape(self, embedder):
        vecs = embedder.encode(["Hello world", "Test sentence"])
        assert vecs.shape == (2, embedder.dim)
        assert vecs.dtype == np.float32

    def test_encode_one_is_1d(self, embedder):
        vec = embedder.encode_one("Single query")
        assert vec.ndim == 1
        assert vec.shape[0] == embedder.dim

    def test_vectors_are_normalised(self, embedder):
        vec = embedder.encode_one("Test")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_list_returns_empty(self, embedder):
        vecs = embedder.encode([])
        assert vecs.shape == (0, embedder.dim)


# ── FAISS store tests ──────────────────────────────────────────────────────────

class TestFAISSStore:
    @pytest.fixture
    def store(self, tmp_path):
        from app.config import Settings
        from app.retriever import FAISSStore
        settings = Settings(faiss_index_dir=str(tmp_path))
        return FAISSStore(settings, dim=64)

    def _random_chunks(self, n: int, doc_id="doc1") -> list:
        from app.chunker import Chunk
        return [Chunk(
            chunk_id=f"{doc_id}-chunk-{i}",
            text=f"Chunk text {i}",
            doc_id=doc_id,
            filename="test.pdf",
            page=i + 1,
            chunk_index=i,
        ) for i in range(n)]

    def _random_vecs(self, n: int, dim=64) -> np.ndarray:
        rng = np.random.default_rng(42)
        v = rng.standard_normal((n, dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_empty_search_returns_nothing(self, store):
        q = self._random_vecs(1)[0]
        assert store.search(q, top_k=5) == []

    def test_add_increments_count(self, store):
        chunks = self._random_chunks(10)
        embs = self._random_vecs(10)
        store.add(chunks, embs)
        assert store.count() == 10

    def test_search_returns_top_k(self, store):
        chunks = self._random_chunks(20)
        embs = self._random_vecs(20)
        store.add(chunks, embs)
        results = store.search(self._random_vecs(1)[0], top_k=5)
        assert len(results) == 5

    def test_delete_removes_correct_chunks(self, store):
        c_a = self._random_chunks(5, doc_id="doc_a")
        c_b = self._random_chunks(5, doc_id="doc_b")
        store.add(c_a, self._random_vecs(5))
        store.add(c_b, self._random_vecs(5))
        deleted = store.delete_doc("doc_a")
        assert deleted == 5
        assert store.count() == 5

    def test_save_and_reload(self, tmp_path):
        from app.config import Settings
        from app.retriever import FAISSStore
        settings = Settings(faiss_index_dir=str(tmp_path))
        s1 = FAISSStore(settings, dim=64)
        chunks = self._random_chunks(8)
        s1.add(chunks, self._random_vecs(8))
        s1.save()

        s2 = FAISSStore(settings, dim=64)
        assert s2.count() == 8


# ── Pipeline integration tests ─────────────────────────────────────────────────

class TestPipeline:
    @pytest.fixture
    def pipeline(self, tmp_path):
        from app.config import Settings
        from app.pipeline import RAGPipeline
        settings = Settings(
            vector_store="faiss",
            faiss_index_dir=str(tmp_path),
            reranker_enabled=False,
            llm_provider="openai",
            openai_api_key="sk-dummy",
        )
        p = RAGPipeline(settings)
        p.embedder.load()
        return p

    def _make_pdf(self, tmp_path, text: str) -> str:
        """Create a minimal single-page PDF in memory."""
        try:
            from pypdf import PdfWriter
            writer = PdfWriter()
            from pypdf.generic import PageObject
            page = writer.add_blank_page(width=612, height=792)
            path = str(tmp_path / "test.pdf")
            with open(path, "wb") as f:
                writer.write(f)
            return path
        except Exception:
            pytest.skip("Could not create test PDF")

    @pytest.mark.asyncio
    async def test_query_empty_index_returns_message(self, pipeline):
        result = await pipeline.query("What is FAISS?")
        assert "upload" in result.answer.lower() or "No relevant" in result.answer


# ── API tests ──────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient
        from app import main as main_module
        from app.config import Settings

        mock_pipeline = MagicMock()
        mock_pipeline.store.count.return_value = 5
        mock_pipeline.embedder._model_name = "bge-small"
        mock_pipeline.generator.model_name = "gpt-4o-mini"
        mock_pipeline.stats.return_value = {
            "total_documents": 2,
            "total_chunks": 5,
            "vector_store": "faiss",
            "embedding_model": "bge-small",
            "llm": "gpt-4o-mini",
        }
        mock_pipeline.list_documents.return_value = []
        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            main_module._pipeline = mock_pipeline
            yield c

        main_module._pipeline = None

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_chunks"] == 5

    def test_upload_rejects_non_pdf(self, client):
        r = client.post(
            "/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 400

    def test_query_empty_index_returns_400(self, client):
        from app import main as main_module
        main_module._pipeline.store.count.return_value = 0
        r = client.post("/query",
                        json={"question": "What is the document about?"})
        assert r.status_code == 400


# ── Generator tests ────────────────────────────────────────────────────────────

class TestGenerator:
    def test_build_openrouter_llm(self):
        from app.config import Settings
        from app.generator import _build_llm
        settings = Settings(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-v1-dummy",
            openrouter_model="meta-llama/llama-3.3-70b-instruct",
        )
        llm = _build_llm(settings)
        assert llm.model_name == "meta-llama/llama-3.3-70b-instruct"
        assert str(llm.openai_api_base).rstrip("/").endswith("openrouter.ai/api/v1") or "openrouter.ai" in str(llm.openai_api_base)

    @pytest.mark.asyncio
    async def test_generator_prompt_variables(self):
        from app.config import Settings
        from app.generator import Generator
        from app.retriever import SearchResult
        settings = Settings(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-v1-dummy",
            openrouter_model="meta-llama/llama-3.3-70b-instruct",
        )
        gen = Generator(settings)
        gen._ensure_chain()
        # Verify prompt expects only context and question
        prompt = gen._chain.first
        assert set(prompt.input_variables) == {"context", "question"}


# ── Evaluator tests ────────────────────────────────────────────────────────────

class TestEvaluator:
    @pytest.fixture(scope="class")
    def embedder(self):
        from app.config import Settings
        from app.embedder import Embedder
        settings = Settings(embedding_model="BAAI/bge-small-en-v1.5",
                            embedding_device="cpu")
        e = Embedder(settings)
        e.load()
        return e

    def test_evaluate_rag_response_metrics(self, embedder):
        from app.evaluator import evaluate_rag_response
        from app.retriever import SearchResult

        question = "What is FAISS used for?"
        answer = "FAISS is a library for efficient similarity search and clustering of dense vectors (p. 1)."
        candidates = [
            SearchResult(
                chunk_id="c1",
                text="FAISS is a library for efficient similarity search and clustering of dense vectors.",
                filename="faiss.pdf",
                page=1,
                doc_id="doc1",
                score=0.92,
                rerank_score=0.95,
            )
        ]

        metrics = evaluate_rag_response(question, answer, candidates, embedder)
        assert 0 <= metrics["faithfulness"] <= 100
        assert 0 <= metrics["answer_relevance"] <= 100
        assert 0 <= metrics["context_relevance"] <= 100
        assert 0 <= metrics["citation_precision"] <= 100
        assert 0 <= metrics["overall_score"] <= 100
        assert metrics["rating"] in ["Excellent", "Good", "Fair", "Needs Review"]
        assert metrics["cited_pages"] == [1]



