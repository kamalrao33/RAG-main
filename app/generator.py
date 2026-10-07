"""
LLM answer generation using LangChain.
Supports: OpenAI · Anthropic · Ollama (local)
"""
import logging
import time
from typing import AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import Settings
from app.retriever import SearchResult

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions strictly based on the \
provided PDF document excerpts.

Rules:
- Answer ONLY using information from the provided context.
- Cite the source with (p. {{page}}) after every claim.
- If multiple passages support a claim, cite all relevant pages.
- If the context does not contain enough information, say:
  "The document does not contain sufficient information to answer this question."
- Be concise, accurate, and do not add information from outside the document.
"""

HUMAN_PROMPT = """\
Document excerpts:
{context}

Question: {question}

Answer (with page citations):"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_context(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Excerpt {i} — {r.filename}, p.{r.page}]\n{r.text}"
        )
    return "\n\n".join(parts)


def _build_llm(settings: Settings):
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key,
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": settings.app_name,
            },
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# ── Generator class ────────────────────────────────────────────────────────────

class Generator:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._chain = None

    def _ensure_chain(self):
        if self._chain:
            return
        logger.info(
            "Initialising LLM: provider=%s model=%s",
            self._settings.llm_provider,
            getattr(self._settings,
                    f"{self._settings.llm_provider}_model", "?"),
        )
        llm = _build_llm(self._settings)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm | StrOutputParser()

    async def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> tuple[str, float]:
        """Returns (answer, generation_latency_ms)."""
        self._ensure_chain()
        context = _format_context(results)
        t0 = time.perf_counter()
        answer = await self._chain.ainvoke({
            "context": context,
            "question": question,
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return answer, round(elapsed_ms, 1)

    async def stream(
        self,
        question: str,
        results: list[SearchResult],
    ) -> AsyncIterator[str]:
        """Yield answer tokens one by one."""
        self._ensure_chain()
        context = _format_context(results)
        async for token in self._chain.astream({
            "context": context,
            "question": question,
        }):
            yield token

    @property
    def model_name(self) -> str:
        p = self._settings.llm_provider
        return getattr(self._settings, f"{p}_model", p)
