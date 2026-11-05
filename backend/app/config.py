"""Application configuration via pydantic-settings.

All values can be overridden by environment variables (see .env.example).

Settings are validated at import time so a bad deployment fails fast with a
readable error instead of misbehaving later: enum-like values are checked
(a typo such as ``VECTOR_BACKEND=lance`` used to fall back silently to Chroma),
numeric knobs are range-checked, and cross-field invariants are enforced (an
overlap >= chunk size used to degrade chunking to one chunk per token).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "lmstudio"]
EmbedProvider = Literal["fastembed", "ollama"]
VectorBackend = Literal["chroma", "lancedb"]
WebSearchProvider = Literal["duckduckgo", "searxng"]
LogFormat = Literal["text", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_LOWERCASE_FIELDS = (
    "default_provider",
    "embed_provider",
    "vector_backend",
    "websearch_provider",
    "log_format",
)
_URL_FIELDS = ("ollama_base_url", "lmstudio_base_url", "searxng_base_url")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server ---
    app_name: str = "Local LLM Chat with RAG"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]
    data_dir: str = "./data"
    # Cap upload size so a single large file cannot exhaust server memory
    # (uploads are read fully into RAM for parsing).
    max_upload_mb: int = Field(50, gt=0, le=2048)

    # --- Logging ---
    # "json" emits one JSON object per line (for log shippers); "text" is for humans.
    log_format: LogFormat = "text"
    log_level: LogLevel = "INFO"

    # --- Database ---
    database_url: str = "sqlite:///./data/app.db"

    # --- LLM providers (both speak the OpenAI /v1 API) ---
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    default_provider: Provider = "ollama"
    default_model: str = Field("llama3.1", min_length=1)
    request_timeout: float = Field(300.0, gt=0)

    # --- Embeddings ---
    # embed_provider: "fastembed" (pure-local, bundled) or "ollama" (uses ollama embeddings)
    embed_provider: EmbedProvider = "fastembed"
    fastembed_model: str = Field("BAAI/bge-small-en-v1.5", min_length=1)
    ollama_embed_model: str = Field("nomic-embed-text", min_length=1)

    # --- Vector store ---
    vector_backend: VectorBackend = "chroma"
    chroma_dir: str = "./data/chroma"
    lancedb_dir: str = "./data/lancedb"
    default_collection: str = Field("default", min_length=1)

    # --- RAG ---
    chunk_size_tokens: int = Field(512, ge=16, le=8192)
    chunk_overlap_tokens: int = Field(64, ge=0)
    retrieve_top_k: int = Field(8, ge=1, le=200)
    rerank_top_n: int = Field(4, ge=1, le=200)
    enable_rerank: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Maximal Marginal Relevance: diversify the final chunks so the context isn't
    # several near-duplicate passages. lambda 1.0 = pure relevance, 0.0 = pure
    # diversity; 0.5 balances both.
    enable_mmr: bool = False
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)

    # --- Memory ---
    memory_summarize_after_tokens: int = Field(3000, ge=256)

    # --- Web search ---
    websearch_provider: WebSearchProvider = "duckduckgo"
    searxng_base_url: str = "http://localhost:8080"
    websearch_max_results: int = Field(5, ge=1, le=20)

    # --- Voice (speech-to-text) ---
    # Any faster-whisper size name (tiny | base | small | medium | large-v3 | …),
    # a Hugging Face repo id, or a local model path.
    whisper_model: str = Field("base", min_length=1)

    @field_validator(*_LOWERCASE_FIELDS, mode="before")
    @classmethod
    def _lowercase(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase(cls, v: object) -> object:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator(*_URL_FIELDS)
    @classmethod
    def _http_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"must be an http(s) URL, got {v!r}")
        return v.rstrip("/")

    @model_validator(mode="after")
    def _cross_field(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"CHUNK_OVERLAP_TOKENS ({self.chunk_overlap_tokens}) must be smaller than "
                f"CHUNK_SIZE_TOKENS ({self.chunk_size_tokens})"
            )
        if self.rerank_top_n > self.retrieve_top_k:
            raise ValueError(
                f"RERANK_TOP_N ({self.rerank_top_n}) cannot exceed RETRIEVE_TOP_K "
                f"({self.retrieve_top_k}): the final context is cut from the retrieved pool"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
