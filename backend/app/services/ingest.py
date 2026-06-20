"""Document parsing + token-aware chunking, then embed + index."""
from __future__ import annotations

import io
from dataclasses import dataclass

import tiktoken

from app.config import settings
from app.services.embeddings import get_embedder
from app.services.vectorstore import VectorRecord, get_vectorstore

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class ParsedChunk:
    text: str
    page: int | None
    chunk_index: int


def _read_pdf(data: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            out.append((i + 1, text))
    return out


def _read_docx(data: bytes) -> list[tuple[int, str]]:
    from docx import Document as Docx

    doc = Docx(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(None, text)] if text.strip() else []


def _read_text(data: bytes) -> list[tuple[int, str]]:
    text = data.decode("utf-8", errors="replace")
    return [(None, text)] if text.strip() else []


def parse(filename: str, content_type: str | None, data: bytes) -> list[tuple[int | None, str]]:
    name = filename.lower()
    if name.endswith(".pdf") or content_type == "application/pdf":
        return _read_pdf(data)
    if name.endswith(".docx"):
        return _read_docx(data)
    # txt, md, csv, code, etc.
    return _read_text(data)


def chunk_tokens(pages: list[tuple[int | None, str]]) -> list[ParsedChunk]:
    """Token-aware sliding-window chunking across page-tagged text."""
    size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    step = max(1, size - overlap)
    chunks: list[ParsedChunk] = []
    idx = 0
    for page, text in pages:
        tokens = _enc.encode(text)
        for start in range(0, len(tokens), step):
            window = tokens[start : start + size]
            if not window:
                continue
            chunk_text = _enc.decode(window).strip()
            if chunk_text:
                chunks.append(ParsedChunk(text=chunk_text, page=page, chunk_index=idx))
                idx += 1
            if start + size >= len(tokens):
                break
    return chunks


def ingest_document(
    *, document_id: str, collection: str, filename: str, content_type: str | None, data: bytes
) -> int:
    """Full pipeline: parse -> chunk -> embed -> upsert. Returns chunk count."""
    pages = parse(filename, content_type, data)
    chunks = chunk_tokens(pages)
    if not chunks:
        return 0

    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])
    store = get_vectorstore()

    records = [
        VectorRecord(
            id=f"{document_id}:{c.chunk_index}",
            text=c.text,
            embedding=vec,
            metadata={
                "document_id": document_id,
                "document": filename,
                "collection": collection,
                "page": c.page,
                "chunk_index": c.chunk_index,
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    store.upsert(collection, records)
    return len(records)
