"""Local embedding service.

Two backends, both keeping data on-machine:
- fastembed: bundled ONNX models, zero external dependency (default).
- ollama: uses a pulled embedding model (e.g. nomic-embed-text).
"""
from __future__ import annotations

from functools import lru_cache

import httpx

from app.config import settings


class Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        # Probe dimensionality once.
        self._dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    @property
    def dim(self) -> int:
        return self._dim


class OllamaEmbedder(Embedder):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        with httpx.Client(timeout=120.0) as client:
            for t in texts:
                r = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                )
                r.raise_for_status()
                out.append(r.json()["embedding"])
        return out

    @property
    def dim(self) -> int:
        return self._dim


@lru_cache
def get_embedder() -> Embedder:
    if settings.embed_provider == "ollama":
        return OllamaEmbedder(settings.ollama_base_url, settings.ollama_embed_model)
    return FastEmbedEmbedder(settings.fastembed_model)
