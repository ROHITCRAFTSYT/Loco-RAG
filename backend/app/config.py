"""Application configuration via pydantic-settings.

All values can be overridden by environment variables (see .env.example).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server ---
    app_name: str = "Local LLM Chat with RAG"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]
    data_dir: str = "./data"

    # --- Database ---
    database_url: str = "sqlite:///./data/app.db"

    # --- LLM providers (both speak the OpenAI /v1 API) ---
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    default_provider: str = "ollama"  # "ollama" | "lmstudio"
    default_model: str = "llama3.1"
    request_timeout: float = 300.0

    # --- Embeddings ---
    # embed_provider: "fastembed" (pure-local, bundled) or "ollama" (uses ollama embeddings)
    embed_provider: str = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    ollama_embed_model: str = "nomic-embed-text"

    # --- Vector store ---
    vector_backend: str = "chroma"  # "chroma" | "lancedb"
    chroma_dir: str = "./data/chroma"
    lancedb_dir: str = "./data/lancedb"
    default_collection: str = "default"

    # --- RAG ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieve_top_k: int = 8
    rerank_top_n: int = 4
    enable_rerank: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Memory ---
    memory_summarize_after_tokens: int = 3000

    # --- Web search ---
    websearch_provider: str = "duckduckgo"  # "duckduckgo" | "searxng"
    searxng_base_url: str = "http://localhost:8080"
    websearch_max_results: int = 5

    # --- Voice (speech-to-text) ---
    whisper_model: str = "base"  # tiny | base | small | medium | large-v3


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
