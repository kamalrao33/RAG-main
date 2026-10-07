"""
PDF ingestion: extract text page-by-page, clean it,
then split into overlapping semantic chunks.
"""
import re
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.
    Tries pdfplumber first (better layout), falls back to pypdf.
    Returns list of {page_number, text}.
    """
    path = Path(pdf_path)
    pages = []

    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page": i, "text": text})
        logger.info("Extracted %d pages with pdfplumber", len(pages))
        return pages
    except Exception as e:
        logger.warning("pdfplumber failed (%s), falling back to pypdf", e)

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
        logger.info("Extracted %d pages with pypdf", len(pages))
        return pages
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {e}") from e


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace, remove control characters."""
    # Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control chars except newline/tab
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursive character splitter — same strategy as LangChain's
    RecursiveCharacterTextSplitter but written explicitly so you
    can see exactly how it works.

    Priority of split points: paragraph → sentence → word → character
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            # Hard character split
            return [text[i:i+chunk_size]
                    for i in range(0, len(text), chunk_size - chunk_overlap)]

        sep = seps[0]
        parts = text.split(sep) if sep else list(text)
        chunks, current = [], ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # part itself is too long — recurse
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks

    raw = _split(text, separators)

    # Add overlap: carry the tail of the previous chunk into the next
    if chunk_overlap == 0 or len(raw) <= 1:
        return [c for c in raw if c.strip()]

    overlapped = [raw[0]]
    for i in range(1, len(raw)):
        prev_tail = overlapped[-1][-chunk_overlap:]
        overlapped.append(prev_tail + raw[i])

    return [c for c in overlapped if c.strip()]


# ── Public API ─────────────────────────────────────────────────────────────────

def pdf_to_chunks(
    pdf_path: str,
    doc_id: str,
    filename: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Full pipeline: PDF → pages → clean text → chunks.
    Returns a flat list of Chunk objects with page metadata.
    """
    pages = extract_pages(pdf_path)
    total_text = sum(len(p["text"]) for p in pages)
    logger.info("PDF '%s': %d pages, %d chars total", filename, len(pages), total_text)

    chunks: list[Chunk] = []
    chunk_index = 0

    for page_info in pages:
        page_num = page_info["page"]
        text = clean_text(page_info["text"])
        if not text:
            continue

        splits = _split_text(text, chunk_size, chunk_overlap)
        for split in splits:
            if len(split.strip()) < 30:          # skip trivial fragments
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=split.strip(),
                doc_id=doc_id,
                filename=filename,
                page=page_num,
                chunk_index=chunk_index,
                metadata={"page": page_num, "filename": filename},
            ))
            chunk_index += 1

    logger.info("Created %d chunks from '%s'", len(chunks), filename)
    return chunks
