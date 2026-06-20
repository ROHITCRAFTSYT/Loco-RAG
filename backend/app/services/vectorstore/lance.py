"""LanceDB-backed vector store (embedded, file-based, no server)."""
from __future__ import annotations

import lancedb
import pyarrow as pa

from app.services.vectorstore.base import QueryHit, VectorRecord


class LanceVectorStore:
    def __init__(self, uri: str):
        self._db = lancedb.connect(uri)

    def _table(self, collection: str, dim: int | None = None):
        if collection in self._db.table_names():
            return self._db.open_table(collection)
        if dim is None:
            return None
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("metadata", pa.string()),  # JSON-encoded
            ]
        )
        return self._db.create_table(collection, schema=schema)

    @staticmethod
    def _enc(meta: dict) -> str:
        import json

        return json.dumps(meta)

    @staticmethod
    def _dec(meta: str) -> dict:
        import json

        try:
            return json.loads(meta) if meta else {}
        except Exception:
            return {}

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        dim = len(records[0].embedding)
        table = self._table(collection, dim)
        rows = [
            {
                "id": r.id,
                "text": r.text,
                "vector": r.embedding,
                "metadata": self._enc(r.metadata or {}),
            }
            for r in records
        ]
        ids = [r.id for r in records]
        table.delete(f"id IN ({','.join(repr(i) for i in ids)})")
        table.add(rows)

    def query(self, collection: str, embedding: list[float], top_k: int) -> list[QueryHit]:
        table = self._table(collection)
        if table is None:
            return []
        res = table.search(embedding).metric("cosine").limit(top_k).to_list()
        hits: list[QueryHit] = []
        for row in res:
            # LanceDB returns _distance (cosine distance); convert to similarity.
            score = 1.0 - float(row.get("_distance", 0.0))
            hits.append(
                QueryHit(id=row["id"], text=row["text"], score=score, metadata=self._dec(row.get("metadata", "")))
            )
        return hits

    def all_texts(self, collection: str) -> list[QueryHit]:
        table = self._table(collection)
        if table is None:
            return []
        rows = table.to_arrow().to_pylist()
        return [
            QueryHit(id=r["id"], text=r["text"], score=0.0, metadata=self._dec(r.get("metadata", "")))
            for r in rows
        ]

    def delete_document(self, collection: str, document_id: str) -> None:
        table = self._table(collection)
        if table is None:
            return
        # metadata is JSON text; match on the embedded document_id field.
        table.delete(f"metadata LIKE '%\"document_id\": \"{document_id}\"%'")

    def list_collections(self) -> list[str]:
        return list(self._db.table_names())

    def drop_collection(self, collection: str) -> None:
        if collection in self._db.table_names():
            self._db.drop_table(collection)
