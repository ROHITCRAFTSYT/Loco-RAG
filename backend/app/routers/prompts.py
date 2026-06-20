"""Saved system-prompt library."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import SavedPrompt
from app.schemas import SavedPromptIn

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.get("")
def list_prompts(session: Session = Depends(get_session)):
    return session.exec(select(SavedPrompt).order_by(SavedPrompt.created_at.desc())).all()


@router.post("")
def create_prompt(body: SavedPromptIn, session: Session = Depends(get_session)):
    p = SavedPrompt(name=body.name, content=body.content)
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


@router.delete("/{prompt_id}")
def delete_prompt(prompt_id: str, session: Session = Depends(get_session)):
    p = session.get(SavedPrompt, prompt_id)
    if not p:
        raise HTTPException(404, "Not found")
    session.delete(p)
    session.commit()
    return {"ok": True}
