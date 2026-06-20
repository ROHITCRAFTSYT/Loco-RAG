"""Seed the default collection with the bundled sample document(s).

Usage:
    python -m scripts.seed                 # ingest into the default collection
    python -m scripts.seed my_collection   # ingest into a named collection

Safe to run repeatedly — re-ingesting overwrites the same chunk ids.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlmodel import Session, select

from app.config import settings
from app.db import engine, init_db
from app.models import Document
from app.services.ingest import ingest_document

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def seed(collection: str) -> None:
    init_db()
    files = sorted(SAMPLES_DIR.glob("*"))
    if not files:
        print(f"No sample files found in {SAMPLES_DIR}")
        return

    with Session(engine) as session:
        for path in files:
            data = path.read_bytes()
            # Reuse an existing Document row for this filename+collection if present.
            existing = session.exec(
                select(Document)
                .where(Document.collection == collection)
                .where(Document.filename == path.name)
            ).first()
            doc = existing or Document(
                collection=collection,
                filename=path.name,
                content_type="text/markdown",
                size_bytes=len(data),
            )
            doc.status = "processing"
            session.add(doc)
            session.commit()
            session.refresh(doc)

            count = ingest_document(
                document_id=doc.id,
                collection=collection,
                filename=path.name,
                content_type=doc.content_type,
                data=data,
            )
            doc.chunk_count = count
            doc.status = "ready" if count else "error"
            session.add(doc)
            session.commit()
            print(f"  ingested {path.name}: {count} chunks -> '{collection}'")

    print(f"Seed complete. Vector backend: {settings.vector_backend}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else settings.default_collection
    seed(target)
