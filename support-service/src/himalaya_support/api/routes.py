from __future__ import annotations

import base64
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from himalaya_support.adapt.to_nepali import unicoder
from himalaya_support.api.schemas import (
    CallStartRequest,
    ChatRequest,
    ChatResponse,
    SpeakRequest,
    TicketCreateRequest,
    TicketUpdateRequest,
    UnicoderRequest,
)
from himalaya_support.config import get_settings
from himalaya_support.inference.client import InferenceError
from himalaya_support.support.engine import SupportEngine

router = APIRouter()


@lru_cache(maxsize=1)
def get_engine() -> SupportEngine:
    return SupportEngine(get_settings())


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.post("/support/unicoder")
def convert_unicoder(payload: UnicoderRequest) -> dict:
    result = unicoder(payload.text)
    return {"nepali": result["nepali"], "mode": result["mode"]}


@router.post("/support/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = get_engine().chat(
            payload.message,
            conversation_id=payload.conversation_id,
            locale=payload.locale,
            channel=payload.channel or "chat",
        )
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail="Support is temporarily unavailable") from exc
    return ChatResponse(
        conversation_id=result["conversation_id"],
        reply=result["reply"],
        language=result["language"],
        transliterated=result.get("transliterated"),
        grounded=bool(result.get("grounded", True)),
        speech_register=result.get("register"),
        tickets=result.get("tickets") or [],
        pending_confirm=result.get("pending_confirm"),
    )


@router.post("/support/calls/start")
def start_call(payload: CallStartRequest) -> dict:
    try:
        return get_engine().start_call(payload.locale)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail="Voice is temporarily unavailable") from exc


@router.get("/support/conversations")
def list_conversations(channel: str | None = None) -> list[dict]:
    return get_engine().store.list_conversations(channel=channel)


@router.get("/support/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    messages = get_engine().store.list_messages(conversation_id)
    if not messages:
        rows = get_engine().store.list_conversations()
        if not any(row["id"] == conversation_id for row in rows):
            raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": conversation_id, "messages": messages}


@router.post("/support/speak")
def speak(payload: SpeakRequest) -> dict:
    try:
        mime, audio = get_engine().speak(payload.text, payload.locale)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail="Voice is temporarily unavailable") from exc
    return {"audio_base64": base64.b64encode(audio).decode("ascii"), "mime": mime}


@router.post("/support/tickets")
def create_ticket(payload: TicketCreateRequest) -> dict:
    return get_engine().store.create_ticket(payload.model_dump())


@router.get("/support/tickets")
def list_tickets(user_id: str | None = None) -> list[dict]:
    return get_engine().store.list_tickets(user_id=user_id)


@router.get("/support/tickets/{ticket_id}")
def get_ticket(ticket_id: str) -> dict:
    ticket = get_engine().store.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.patch("/support/tickets/{ticket_id}")
def update_ticket(ticket_id: str, payload: TicketUpdateRequest) -> dict:
    ticket = get_engine().store.update_ticket(
        ticket_id,
        status=payload.status,
        priority=payload.priority,
        description=payload.note,
    )
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket
