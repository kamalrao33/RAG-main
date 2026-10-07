"""
Pluggable vector store.
Swap backends by changing VECTOR_STORE env var: faiss | chroma | pinecone
"""
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import asdict

from app.chunker import Chunk
from app.config import Settings

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

class SearchResult:
    def __init__(self, chunk_id: str, text: str, filename: str,
                 page: int, doc_id: str, score: float,
                 rerank_score: float | None = None):
        self.chunk_id = chunk_id
        self.text = text
        self.filename = filename
        self.page = page
        self.doc_id = doc_id
        self.score = score
        self.rerank_score = rerank_score

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "filename": self.filename,
            "page": self.page,
            "doc_id": self.doc_id,
            "score": self.score,
            "rerank_score": self.rerank_score,
        }


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]: ...

    @abstractmethod
    def delete_doc(self, doc_id: str) -> int: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def save(self) -> None: ...


# ── FAISS ──────────────────────────────────────────────────────────────────────

class FAISSStore(BaseStore):
    """
    Uses IndexFlatIP (exact inner-product / cosine search with normalised vectors).
    IndexIDMap wraps it so we can assign our own int64 IDs.
    Metadata is stored in a JSON sidecar file alongside the .faiss binary.
    """

    def __init__(self, settings: Settings, dim: int):
        import faiss
        self._faiss = faiss
        self.dim = dim
        self._dir = Path(settings.faiss_index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._idx_path  = self._dir / "index.faiss"
        self._meta_path = self._dir / "index.json"
        self._index: faiss.IndexIDMap | None = None
        self._meta: list[dict] = []
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self._idx_path.exists() and self._meta_path.exists():
            logger.info("Loading existing FAISS index from %s", self._idx_path)
            self._index = self._faiss.read_index(str(self._idx_path))
            with open(self._meta_path) as f:
                self._meta = json.load(f)
            logger.info("FAISS index loaded — %d vectors", self._index.ntotal)
        else:
            logger.info("Creating new FAISS index (dim=%d)", self.dim)
            flat = self._faiss.IndexFlatIP(self.dim)
            self._index = self._faiss.IndexIDMap(flat)
            self._meta = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        # Map UUID → stable int64 via hash
        ids = np.array(
            [hash(c.chunk_id) & 0x7FFFFFFFFFFFFFFF for c in chunks],
            dtype=np.int64,
        )
        self._index.add_with_ids(embeddings, ids)
        for chunk, fid in zip(chunks, ids):
            self._meta.append({
                "fid": int(fid),
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "filename": chunk.filename,
                "page": chunk.page,
                "doc_id": chunk.doc_id,
            })
        logger.debug("Added %d vectors to FAISS", len(chunks))

    def search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(q, k)
        id_map = {m["fid"]: m for m in self._meta}
        results = []
        for score, fid in zip(scores[0], ids[0]):
            if fid < 0:
                continue
            m = id_map.get(int(fid))
            if m:
                results.append(SearchResult(
                    chunk_id=m["chunk_id"],
                    text=m["text"],
                    filename=m["filename"],
                    page=m["page"],
                    doc_id=m["doc_id"],
                    score=float(score),
                ))
        return results

    def delete_doc(self, doc_id: str) -> int:
        to_remove = [m for m in self._meta if m["doc_id"] == doc_id]
        if not to_remove:
            return 0
        fids = np.array([m["fid"] for m in to_remove], dtype=np.int64)
        sel = self._faiss.IDSelectorArray(fids)
        self._index.remove_ids(sel)
        self._meta = [m for m in self._meta if m["doc_id"] != doc_id]
        return len(to_remove)

    def count(self) -> int:
        return self._index.ntotal

    def save(self) -> None:
        self._faiss.write_index(self._index, str(self._idx_path))
        with open(self._meta_path, "w") as f:
            json.dump(self._meta, f)
        logger.info("FAISS index saved (%d vectors)", self._index.ntotal)


# ── ChromaDB ───────────────────────────────────────────────────────────────────

class ChromaStore(BaseStore):
    def __init__(self, settings: Settings):
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._col = client.get_or_create_collection(
            name="pdf_rag",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection ready — %d docs", self._col.count())

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{
                "filename": c.filename,
                "page": c.page,
                "doc_id": c.doc_id,
            } for c in chunks],
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._col.count() == 0:
            return []
        n = min(top_k, self._col.count())
        r = self._col.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n,
        )
        results = []
        for i, doc in enumerate(r["documents"][0]):
            m = r["metadatas"][0][i]
            dist = r["distances"][0][i]
            results.append(SearchResult(
                chunk_id=r["ids"][0][i],
                text=doc,
                filename=m.get("filename", ""),
                page=m.get("page", 0),
                doc_id=m.get("doc_id", ""),
                score=1.0 - dist,
            ))
        return results

    def delete_doc(self, doc_id: str) -> int:
        existing = self._col.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._col.delete(ids=existing["ids"])
        return len(existing["ids"])

    def count(self) -> int:
        return self._col.count()

    def save(self) -> None:
        pass  # Chroma auto-persists


# ── Pinecone ───────────────────────────────────────────────────────────────────

class PineconeStore(BaseStore):
    def __init__(self, settings: Settings, dim: int):
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = [i.name for i in pc.list_indexes()]
        if settings.pinecone_index_name not in existing:
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = pc.Index(settings.pinecone_index_name)
        logger.info("Pinecone index '%s' connected", settings.pinecone_index_name)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        vectors = [
            (c.chunk_id, emb.tolist(), {
                "text": c.text,
                "filename": c.filename,
                "page": c.page,
                "doc_id": c.doc_id,
            })
            for c, emb in zip(chunks, embeddings)
        ]
        # Pinecone recommends batches of ≤100
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i:i + 100])

    def search(self, query_vec: np.ndarray, top_k: int) -> list[SearchResult]:
        r = self._index.query(
            vector=query_vec.tolist(),
            top_k=top_k,
            include_metadata=True,
        )
        results = []
        for m in r.get("matches", []):
            md = m["metadata"]
            results.append(SearchResult(
                chunk_id=m["id"],
                text=md.get("text", ""),
                filename=md.get("filename", ""),
                page=md.get("page", 0),
                doc_id=md.get("doc_id", ""),
                score=m["score"],
            ))
        return results

    def delete_doc(self, doc_id: str) -> int:
        r = self._index.query(
            vector=[0.0] * 384,
            top_k=10_000,
            filter={"doc_id": {"$eq": doc_id}},
            include_metadata=False,
        )
        ids = [m["id"] for m in r.get("matches", [])]
        if ids:
            self._index.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    def save(self) -> None:
        pass  # Pinecone is cloud-managed


# ── Factory ────────────────────────────────────────────────────────────────────

def build_store(settings: Settings, dim: int) -> BaseStore:
    store = settings.vector_store
    if store == "faiss":
        return FAISSStore(settings, dim)
    elif store == "chroma":
        return ChromaStore(settings)
    elif store == "pinecone":
        return PineconeStore(settings, dim)
    raise ValueError(f"Unknown vector store: {store!r}")
