"""Conversation + message CRUD."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.db import get_session
from app.models import Conversation, Message
from app.schemas import ConversationCreate, ConversationUpdate

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
def list_conversations(session: Session = Depends(get_session)):
    rows = session.exec(
        select(Conversation).order_by(Conversation.pinned.desc(), Conversation.updated_at.desc())
    ).all()
    return rows


@router.post("")
def create_conversation(body: ConversationCreate, session: Session = Depends(get_session)):
    conv = Conversation(**{k: v for k, v in body.model_dump().items() if v is not None})
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, session: Session = Depends(get_session)):
    conv = session.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Not found")
    return conv


@router.patch("/{conversation_id}")
def update_conversation(
    conversation_id: str, body: ConversationUpdate, session: Session = Depends(get_session)
):
    conv = session.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(conv, k, v)
    conv.updated_at = datetime.now(timezone.utc)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, session: Session = Depends(get_session)):
    conv = session.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "Not found")
    for m in session.exec(select(Message).where(Message.conversation_id == conversation_id)).all():
        session.delete(m)
    session.delete(conv)
    session.commit()
    return {"ok": True}


@router.get("/{conversation_id}/messages")
def list_messages(conversation_id: str, session: Session = Depends(get_session)):
    rows = session.exec(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    ).all()
    out = []
    for m in rows:
        out.append(
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": json.loads(m.sources) if m.sources else None,
                "usage": json.loads(m.usage) if m.usage else None,
                "created_at": m.created_at,
            }
        )
    return out


@router.delete("/{conversation_id}/messages/after/{message_id}")
def delete_after(conversation_id: str, message_id: str, session: Session = Depends(get_session)):
    """Used for edit-and-rerun: drop the target message and everything after it."""
    target = session.get(Message, message_id)
    if not target:
        raise HTTPException(404, "Message not found")
    rows = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.created_at >= target.created_at)
    ).all()
    for m in rows:
        session.delete(m)
    session.commit()
    return {"deleted": len(rows)}
