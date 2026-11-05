"""Settings validation: bad deployments must fail at startup, loudly.

Before validation, a typo like VECTOR_BACKEND=lance silently fell back to
Chroma, and an overlap >= chunk size quietly degraded chunking to one chunk per
token. These pin the fail-fast behaviour and the friendly normalisation.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def make(**overrides) -> Settings:
    # _env_file=None: ignore any developer .env so tests are hermetic.
    return Settings(_env_file=None, **overrides)


def test_defaults_are_valid():
    s = make()
    assert s.vector_backend == "chroma"
    assert s.chunk_overlap_tokens < s.chunk_size_tokens


@pytest.mark.parametrize(
    "field,value",
    [
        ("vector_backend", "lance"),
        ("embed_provider", "openai"),
        ("default_provider", "vllm"),
        ("websearch_provider", "google"),
        ("log_format", "xml"),
        ("log_level", "LOUD"),
    ],
)
def test_unknown_enum_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_enum_values_are_case_and_space_insensitive():
    s = make(vector_backend=" LanceDB ", default_provider="LMStudio", log_level="debug")
    assert s.vector_backend == "lancedb"
    assert s.default_provider == "lmstudio"
    assert s.log_level == "DEBUG"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        make(chunk_size_tokens=128, chunk_overlap_tokens=128)


def test_rerank_top_n_cannot_exceed_retrieve_top_k():
    with pytest.raises(ValidationError, match="RERANK_TOP_N"):
        make(retrieve_top_k=3, rerank_top_n=4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mmr_lambda", 1.5),
        ("mmr_lambda", -0.1),
        ("retrieve_top_k", 0),
        ("max_upload_mb", 0),
        ("request_timeout", 0),
        ("chunk_size_tokens", 4),
    ],
)
def test_numeric_ranges(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_base_urls_must_be_http_and_lose_trailing_slash():
    assert make(ollama_base_url="http://ollama:11434/").ollama_base_url == "http://ollama:11434"
    with pytest.raises(ValidationError, match="http"):
        make(ollama_base_url="ollama:11434")


def test_env_vars_are_validated(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_whisper_accepts_repo_ids_and_paths():
    assert make(whisper_model="distil-large-v3").whisper_model == "distil-large-v3"
    assert make(whisper_model="/models/whisper").whisper_model == "/models/whisper"
