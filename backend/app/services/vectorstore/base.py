"""Common interface for vector stores. Chroma and LanceDB both implement this."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def upsert(self, collection: str, records: list[VectorRecord]) -> None: ...

    def query(
        self, collection: str, embedding: list[float], top_k: int
    ) -> list[QueryHit]: ...

    def all_texts(self, collection: str) -> list[QueryHit]:
        """Return every chunk in a collection (used to build the BM25 index)."""
        ...

    def delete_document(self, collection: str, document_id: str) -> None: ...

    def list_collections(self) -> list[str]: ...

    def drop_collection(self, collection: str) -> None: ...
