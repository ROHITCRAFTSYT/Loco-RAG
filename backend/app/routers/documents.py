"""Document upload, ingestion, listing, and deletion. Manages collections."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select

from app.config import settings
from app.db import engine, get_session
from app.models import Document
from app.services import ingest
from app.services.vectorstore import get_vectorstore

router = APIRouter(prefix="/api/documents", tags=["documents"])


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
            doc.status = "error"
            doc.error = str(exc)
        session.add(doc)
        session.commit()


@router.post("/upload")
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form(settings.default_collection),
    session: Session = Depends(get_session),
):
    data = await file.read()
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
    data = await file.read()
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
