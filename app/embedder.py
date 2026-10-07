"""
Wraps sentence-transformers to convert text → float32 numpy vectors.
Model: BAAI/bge-small-en-v1.5  (384-dim, CPU-friendly, great for retrieval)
"""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import Settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, settings: Settings):
        self._model_name = settings.embedding_model
        self._device = settings.embedding_device
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    # ── Lazy load ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._model:
            return
        logger.info("Loading embedding model '%s' on %s",
                    self._model_name, self._device)
        self._model = SentenceTransformer(self._model_name,
                                          device=self._device)
        # probe to get dimension
        probe = self._model.encode(["hello"], convert_to_numpy=True)
        self._dim = int(probe.shape[1])
        logger.info("Embedder ready — dim=%d", self._dim)

    @property
    def model(self) -> SentenceTransformer:
        if not self._model:
            self.load()
        return self._model

    @property
    def dim(self) -> int:
        if not self._dim:
            self.load()
        return self._dim

    # ── Encoding ───────────────────────────────────────────────────────────────
    def encode(self, texts: list[str],
               batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts.
        Returns float32 numpy array of shape (len(texts), dim).
        normalize_embeddings=True  →  cosine similarity == dot product,
        so we can use FAISS IndexFlatIP for cosine search.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string — used for query embedding."""
        return self.encode([text])[0]
