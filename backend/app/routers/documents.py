"""Document upload, ingestion, listing, and deletion. Manages collections."""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select

from app.config import settings
from app.db import engine, get_session
from app.models import Document
from app.schemas import IngestResponse
from app.services import ingest
from app.services.vectorstore import get_vectorstore

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# The intersection of Chroma's and LanceDB's naming rules: 3-63 chars of
# [A-Za-z0-9._-], starting and ending alphanumeric. Checked up front so a bad
# name is a 422 at upload time rather than an "error" status after background
# ingestion has already accepted the file.
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")


def validate_collection_name(name: str) -> str:
    if not _COLLECTION_RE.match(name) or ".." in name:
        raise HTTPException(
            422,
            "Collection names must be 3-63 characters of letters, digits, '.', '_' or '-', "
            "starting and ending with a letter or digit",
        )
    if name.startswith("chat-"):
        raise HTTPException(422, "The 'chat-' prefix is reserved for per-conversation attachments")
    return name


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload fully into memory, but refuse anything over the configured
    size limit so a large file cannot exhaust server memory (returns HTTP 413).
    The file is read in chunks and aborted as soon as the cap is exceeded.
    """
    max_bytes = settings.max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, f"File too large (limit {settings.max_upload_mb} MB)")
        chunks.append(chunk)
    return b"".join(chunks)


def _process(document_id: str, collection: str, filename: str, content_type: str | None, data: bytes):
    with Session(engine) as session:
        doc = session.get(Document, document_id)
        if not doc:
            return
        doc.status = "processing"
        session.add(doc)
        session.commit()
        try:
            count = ingest.ingest_document(
                document_id=document_id,
                collection=collection,
                filename=filename,
                content_type=content_type,
                data=data,
            )
            doc.chunk_count = count
            doc.status = "ready" if count else "error"
            doc.error = None if count else "No extractable text"
        except Exception as exc:  # pragma: no cover - depends on file content
            logger.warning(
                "ingest failed",
                extra={"document_id": document_id, "collection": collection, "error": str(exc)},
            )
            doc.status = "error"
            doc.error = str(exc)
        session.add(doc)
        session.commit()
        logger.info(
            "ingested",
            extra={"document_id": document_id, "status": doc.status, "chunks": doc.chunk_count},
        )


@router.post("/upload", response_model=IngestResponse)
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form(settings.default_collection),
    session: Session = Depends(get_session),
):
    validate_collection_name(collection)
    data = await _read_capped(file)
    doc = Document(
        collection=collection,
        filename=file.filename or "upload",
        content_type=file.content_type,
        size_bytes=len(data),
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    # Ingest off the request path so the UI stays responsive.
    background.add_task(_process, doc.id, collection, doc.filename, doc.content_type, data)
    return {"document_id": doc.id, "status": doc.status, "chunk_count": 0}


@router.post("/attach")
async def attach(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    session: Session = Depends(get_session),
):
    """Ephemeral "talk to this doc": ingest a file inline (synchronously) into a
    per-conversation collection so the very next chat message can retrieve it.
    """
    data = await _read_capped(file)
    # Chroma requires collection names to start/end alphanumeric, so no "__" prefix.
    collection = f"chat-{conversation_id}"
    doc = Document(
        collection=collection,
        filename=file.filename or "attachment",
        content_type=file.content_type,
        size_bytes=len(data),
        status="processing",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    try:
        count = ingest.ingest_document(
            document_id=doc.id,
            collection=collection,
            filename=doc.filename,
            content_type=doc.content_type,
            data=data,
        )
        doc.chunk_count = count
        doc.status = "ready" if count else "error"
        doc.error = None if count else "No extractable text"
    except Exception as exc:  # pragma: no cover - depends on file content
        logger.warning("attach ingest failed", extra={"document_id": doc.id, "error": str(exc)})
        doc.status = "error"
        doc.error = str(exc)
    session.add(doc)
    session.commit()
    return {
        "document_id": doc.id,
        "collection": collection,
        "filename": doc.filename,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
    }


@router.get("")
def list_documents(collection: str | None = None, session: Session = Depends(get_session)):
    stmt = select(Document).order_by(Document.created_at.desc())
    if collection:
        stmt = stmt.where(Document.collection == collection)
    return session.exec(stmt).all()


@router.get("/stats")
def collection_stats(session: Session = Depends(get_session)):
    """Per-collection knowledge-base totals: documents, ready count, chunks, and
    bytes. Ephemeral chat-attachment collections are excluded."""
    docs = session.exec(select(Document)).all()
    return ingest.aggregate_collection_stats(docs)


@router.get("/collections")
def list_collections(session: Session = Depends(get_session)):
    rows = session.exec(select(Document.collection).distinct()).all()
    # Hide per-conversation ephemeral attachment collections from the KB picker.
    visible = {c for c in rows if not c.startswith("chat-")}
    cols = sorted({*visible, settings.default_collection})
    return cols


@router.delete("/{document_id}")
def delete_document(document_id: str, session: Session = Depends(get_session)):
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Not found")
    get_vectorstore().delete_document(doc.collection, document_id)
    session.delete(doc)
    session.commit()
    return {"ok": True}
