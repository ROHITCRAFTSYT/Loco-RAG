"""Model discovery across configured providers."""
from __future__ import annotations

from fastapi import APIRouter

from app.services.llm.base import all_providers

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models():
    out = []
    for provider in all_providers():
        try:
            out.extend(await provider.list_models())
        except Exception:
            # A provider that isn't running just contributes no models.
            continue
    return out
