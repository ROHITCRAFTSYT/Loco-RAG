"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import init_db
from app.routers import chat, conversations, documents, models, prompts, search, voice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info(
        "startup vector_backend=%s embed_provider=%s default_provider=%s",
        settings.vector_backend,
        settings.embed_provider,
        settings.default_provider,
    )
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    """Attach a request id and log latency for every request."""
    request_id = uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - convert unhandled errors to JSON 500
        logger.exception("unhandled error rid=%s %s %s", request_id, request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "rid=%s %s %s -> %s %.1fms",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.include_router(conversations.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(models.router)
app.include_router(search.router)
app.include_router(prompts.router)
app.include_router(voice.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "vector_backend": settings.vector_backend,
        "embed_provider": settings.embed_provider,
        "default_provider": settings.default_provider,
        "default_model": settings.default_model,
    }
