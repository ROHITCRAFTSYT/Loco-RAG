"""Local speech-to-text via faster-whisper.

The model is lazy-loaded on first use so it never delays app startup and stays
optional — TTS is handled entirely in the browser (speechSynthesis), so no
text-to-speech dependency is needed server-side.
"""
from __future__ import annotations

import io
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings

router = APIRouter(prefix="/api/voice", tags=["voice"])


@lru_cache
def _model():
    from faster_whisper import WhisperModel

    # int8 on CPU keeps it light; users can override via env.
    return WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio")
    try:
        segments, info = _model().transcribe(io.BytesIO(data), beam_size=1)
        text = "".join(seg.text for seg in segments).strip()
    except Exception as exc:  # pragma: no cover - depends on audio/codec
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    return {"text": text, "language": info.language}
