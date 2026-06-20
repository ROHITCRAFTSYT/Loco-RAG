"""ChromaDB-backed vector store (embedded, persistent)."""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.services.vectorstore.base import QueryHit, VectorRecord


class ChromaVectorStore:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(
            path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )

    def _coll(self, collection: str):
        return self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        coll = self._coll(collection)
        coll.upsert(
            ids=[r.id for r in records],
            embeddings=[r.embedding for r in records],
            documents=[r.text for r in records],
            metadatas=[r.metadata or {} for r in records],
        )

    def query(self, collection: str, embedding: list[float], top_k: int) -> list[QueryHit]:
        coll = self._coll(collection)
        if coll.count() == 0:
            return []
        res = coll.query(
            query_embeddings=[embedding],
            n_results=min(top_k, coll.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[QueryHit] = []
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # cosine distance -> similarity
            hits.append(QueryHit(id=id_, text=doc, score=1.0 - dist, metadata=meta or {}))
        return hits

    def all_texts(self, collection: str) -> list[QueryHit]:
        coll = self._coll(collection)
        if coll.count() == 0:
            return []
        res = coll.get(include=["documents", "metadatas"])
        return [
            QueryHit(id=i, text=d, score=0.0, metadata=m or {})
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def delete_document(self, collection: str, document_id: str) -> None:
        self._coll(collection).delete(where={"document_id": document_id})

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def drop_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass
