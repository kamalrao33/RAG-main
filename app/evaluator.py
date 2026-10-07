"""
RAG Evaluation Module.
Computes real-time evaluation metrics for every query:
- Faithfulness / Groundedness (checks hallucination vs retrieved context)
- Answer Relevance (semantic alignment between question and answer)
- Context Precision / Relevance (relevance of retrieved passages to the question)
- Citation Accuracy & Coverage (validity of page citations in the answer)
- Overall Composite RAG Score & Quality Rating
"""
import re
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.embedder import Embedder
    from app.retriever import SearchResult


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences while ignoring page citation periods."""
    cleaned = re.sub(r'\(p\.\s*\d+(?:,\s*p\.\s*\d+)*\)', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_citations(text: str) -> list[int]:
    """Extract all cited page numbers like (p. 3), (p. 3, p. 4), (page 2)."""
    matches = re.findall(r'\((?:p\.|page)?\s*(\d+(?:,\s*(?:p\.|page)?\s*\d+)*)\)', text, re.IGNORECASE)
    pages = []
    for match in matches:
        nums = re.findall(r'\d+', match)
        pages.extend(int(n) for n in nums)
    return sorted(list(set(pages)))


def _sigmoid(x: float) -> float:
    """Map logit score to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_rag_response(
    question: str,
    answer: str,
    candidates: list,
    embedder: "Embedder",
) -> dict:
    """
    Computes RAG evaluation metrics:
    - faithfulness: Groundedness of answer sentences against context (0-100%)
    - answer_relevance: Question-Answer semantic similarity (0-100%)
    - context_relevance: Retrieved context quality (0-100%)
    - citation_precision: Validity of citations vs retrieved pages (0-100%)
    - overall_score: Weighted composite RAG score (0-100%)
    - rating: Excellent | Good | Fair | Low
    - cited_pages: List of unique cited page integers
    """
    if not candidates or "does not contain sufficient information" in answer.lower():
        return {
            "faithfulness": 100.0,
            "answer_relevance": 60.0 if not candidates else 80.0,
            "context_relevance": 0.0 if not candidates else 50.0,
            "citation_precision": 100.0,
            "overall_score": 50.0 if not candidates else 75.0,
            "rating": "Fair" if not candidates else "Good",
            "cited_pages": [],
        }

    # ── 1. Context Relevance ──────────────────────────────────────────────────
    scores = []
    for c in candidates:
        score = getattr(c, "rerank_score", None)
        if score is None or score == 0.0:
            score = getattr(c, "score", 0.5)
        # If cross-encoder logit score, map to sigmoid
        if score > 1.0 or score < 0.0:
            score = _sigmoid(score)
        scores.append(float(score))
    
    context_relevance = float(np.mean(scores)) if scores else 0.5
    # Normalize/scale to realistic percentages
    context_relevance = min(1.0, max(0.0, context_relevance))

    # ── 2. Answer Relevance ───────────────────────────────────────────────────
    q_emb = embedder.encode_one(question)
    a_emb = embedder.encode_one(answer)
    # Cosine similarity for normalized vectors is dot product
    raw_sim = float(np.dot(q_emb, a_emb))
    # Map typical cosine similarity ranges [0.3, 0.9] to [0.5, 1.0] for calibrated score
    ans_rel = max(0.0, min(1.0, (raw_sim - 0.1) / 0.8))

    # ── 3. Faithfulness / Groundedness ────────────────────────────────────────
    sentences = _split_sentences(answer)
    if not sentences:
        faithfulness = 0.9
    else:
        context_texts = [c.text for c in candidates]
        context_embs = embedder.encode(context_texts)  # (N, D)
        sentence_embs = embedder.encode(sentences)     # (M, D)
        
        # Compute similarity matrix (M, N)
        sim_matrix = np.dot(sentence_embs, context_embs.T)
        # Max similarity for each sentence across candidate chunks
        max_sims = np.max(sim_matrix, axis=1)
        # Calibrate groundedness
        calibrated_sims = [(s - 0.2) / 0.75 for s in max_sims]
        faithfulness = float(np.mean([max(0.0, min(1.0, s)) for s in calibrated_sims]))

    # ── 4. Citation Precision & Extraction ────────────────────────────────────
    cited_pages = _extract_citations(answer)
    retrieved_pages = {getattr(c, "page", 1) for c in candidates}
    
    if cited_pages:
        valid_count = sum(1 for p in cited_pages if p in retrieved_pages)
        citation_precision = valid_count / len(cited_pages)
    else:
        # If no citations were present, give a neutral/default score
        citation_precision = 0.85

    # ── 5. Composite Score & Rating ───────────────────────────────────────────
    # Weights: Faithfulness (35%), Answer Relevance (35%), Context Relevance (20%), Citations (10%)
    overall = (
        0.35 * faithfulness +
        0.35 * ans_rel +
        0.20 * context_relevance +
        0.10 * citation_precision
    ) * 100.0

    overall_score = round(max(0.0, min(100.0, overall)), 1)
    faithfulness_pct = round(faithfulness * 100.0, 1)
    ans_rel_pct = round(ans_rel * 100.0, 1)
    context_rel_pct = round(context_relevance * 100.0, 1)
    citation_pct = round(citation_precision * 100.0, 1)

    if overall_score >= 85.0:
        rating = "Excellent"
    elif overall_score >= 70.0:
        rating = "Good"
    elif overall_score >= 50.0:
        rating = "Fair"
    else:
        rating = "Needs Review"

    return {
        "faithfulness": faithfulness_pct,
        "answer_relevance": ans_rel_pct,
        "context_relevance": context_rel_pct,
        "citation_precision": citation_pct,
        "overall_score": overall_score,
        "rating": rating,
        "cited_pages": cited_pages,
    }
