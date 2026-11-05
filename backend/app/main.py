"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import init_db
from app.logging_config import configure_logging, request_id_var
from app.routers import chat, conversations, documents, models, prompts, search, voice
from app.services.llm import LLMError
from app.services.rag import RetrievalError

configure_logging(settings.log_level, settings.log_format)
logger = logging.getLogger("app")

# Accept a caller-supplied X-Request-ID (e.g. from a reverse proxy) only if it is
# short and boring, so it can't be used to inject junk into the logs.
_RID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info(
        "startup",
        extra={
            "vector_backend": settings.vector_backend,
            "embed_provider": settings.embed_provider,
            "default_provider": settings.default_provider,
            "default_model": settings.default_model,
        },
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
    """Attach a request id (echoed as X-Request-ID and stamped on every log line
    emitted while handling the request) and log method, path, status, latency."""
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _RID_RE.match(incoming) else uuid4().hex[:12]
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:  # convert unhandled errors to a JSON 500 with the id
            logger.exception(
                "unhandled error", extra={"method": request.method, "path": request.url.path}
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.exception_handler(LLMError)
async def _llm_error(request: Request, exc: LLMError):
    # The local model runtime is an upstream dependency: 502, with an actionable message.
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc), "provider": exc.provider, "request_id": request_id_var.get()},
    )


@app.exception_handler(RetrievalError)
async def _retrieval_error(request: Request, exc: RetrievalError):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "stage": exc.stage, "request_id": request_id_var.get()},
    )


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
