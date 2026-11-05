"""Contract tests for both real vector-store backends (embedded, on tmp dirs).

The same assertions run against Chroma and LanceDB so the "swap with one env var"
promise is actually checked: upsert is idempotent, query ranks by cosine and
reports similarity, delete_document removes only that document's chunks, and
collections can be listed and dropped. Vectors are supplied explicitly, so no
embedding model is downloaded.
"""
from __future__ import annotations

import pytest

from app.services.vectorstore.base import VectorRecord


def _rec(doc: str, idx: int, vec: list[float]) -> VectorRecord:
    return VectorRecord(
        id=f"{doc}:{idx}",
        text=f"{doc} chunk {idx}",
        embedding=vec,
        metadata={"document_id": doc, "document": f"{doc}.md", "chunk_index": idx},
    )


@pytest.fixture(params=["chroma", "lancedb"])
def store(request, tmp_path):
    if request.param == "chroma":
        from app.services.vectorstore.chroma import ChromaVectorStore

        return ChromaVectorStore(str(tmp_path / "chroma"))
    from app.services.vectorstore.lance import LanceVectorStore

    return LanceVectorStore(str(tmp_path / "lance"))


RECORDS = [
    _rec("docA", 0, [1.0, 0.0, 0.0]),
    _rec("docA", 1, [0.9, 0.1, 0.0]),
    _rec("docB", 0, [0.0, 1.0, 0.0]),
]


def test_empty_collection_queries_cleanly(store):
    assert store.query("kb-main", [1.0, 0.0, 0.0], 5) == []
    assert store.all_texts("kb-main") == []


def test_query_ranks_by_cosine_similarity(store):
    store.upsert("kb-main", RECORDS)
    hits = store.query("kb-main", [1.0, 0.0, 0.0], 2)
    assert [h.id for h in hits] == ["docA:0", "docA:1"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].metadata["document"] == "docA.md"


def test_upsert_is_idempotent(store):
    store.upsert("kb-main", RECORDS)
    store.upsert("kb-main", RECORDS)
    assert len(store.all_texts("kb-main")) == 3


def test_delete_document_removes_only_that_document(store):
    store.upsert("kb-main", RECORDS)
    store.delete_document("kb-main", "docA")
    remaining = store.all_texts("kb-main")
    assert [h.id for h in remaining] == ["docB:0"]


def test_collections_list_and_drop(store):
    store.upsert("kb-one", RECORDS[:1])
    store.upsert("kb-two", RECORDS[2:])
    assert {"kb-one", "kb-two"} <= set(store.list_collections())
    store.drop_collection("kb-one")
    assert "kb-one" not in store.list_collections()
    store.drop_collection("never-existed")  # no-op, no exception
