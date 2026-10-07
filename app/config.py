from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "PDF RAG Pipeline"
    app_version: str = "1.0.0"
    debug: bool = False
    max_upload_mb: int = 100

    # LLM
    llm_provider: Literal["openai", "anthropic", "ollama", "openrouter"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Vector store
    vector_store: Literal["faiss", "chroma", "pinecone"] = "faiss"
    faiss_index_dir: str = "indexes"
    chroma_persist_dir: str = "indexes/chroma"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "pdf-rag"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    retrieval_top_k: int = 6
    reranker_enabled: bool = True
    reranker_top_k: int = 4

    # Cache
    redis_url: str = "redis://localhost:6379"
    cache_enabled: bool = False
    cache_ttl: int = 3600

    # Paths
    upload_dir: str = "uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
