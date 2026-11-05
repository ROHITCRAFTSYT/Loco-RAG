"""Model discovery across configured providers."""
from __future__ import annotations

import logging

from fastapi import APIRouter

from app.services.llm.base import all_providers

router = APIRouter(prefix="/api/models", tags=["models"])
logger = logging.getLogger(__name__)


@router.get("")
async def list_models():
    out = []
    for provider in all_providers():
        try:
            out.extend(await provider.list_models())
        except Exception as exc:
            # A provider that isn't running just contributes no models.
            logger.debug("provider unavailable", extra={"provider": provider.name, "error": str(exc)})
            continue
    return out
