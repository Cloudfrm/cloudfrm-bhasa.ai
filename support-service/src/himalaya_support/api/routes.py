from __future__ import annotations

import base64
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from himalaya_support.adapt.candidates import convert_message
from himalaya_support.api.schemas import (
    CallStartRequest,
    CandidatesRequest,
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
    """Real check with the time of the check (E5)."""
    return get_engine().health()


@router.get("/capabilities")
def capabilities() -> dict:
    """Read at runtime by the desk. Voice controls render only when available === true (E3)."""
    return get_engine().capabilities()


@router.post("/support/unicoder")
def convert_unicoder(payload: UnicoderRequest) -> dict:
    """Word-run conversion (E8). Kept for the member chat page; protected runs are never touched."""
    result = convert_message(payload.text)
    return {"nepali": result["text"], "mode": "word_runs", "runs": result["runs"]}


@router.post("/support/translit/candidates")
def translit_candidates(payload: CandidatesRequest) -> dict:
    """Per-run decisions + ranked candidates for the IME strip (E9)."""
    result = convert_message(payload.text, payload.choices)
    return {"runs": result["runs"], "default_text": result["text"]}


@router.get("/terminology")
def terminology() -> dict:
    """Terminology and format rules (product name, numerals, time, calendar, normalisation)."""
    import json

    path = get_settings().terminology_path
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@router.get("/support/topics")
def topics() -> dict:
    """Sample questions verified answerable by the loaded corpus (E6)."""
    engine = get_engine()
    return {"kind": "sample_questions", "verified": engine.refusals is not None, "chips": engine.chips()}


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
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(**{k: v for k, v in result.items() if k in ChatResponse.model_fields})


@router.post("/support/calls/start")
def start_call(payload: CallStartRequest) -> dict:
    try:
        return get_engine().start_call(payload.locale)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail="voice not_deployed") from exc


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
        raise HTTPException(status_code=503, detail="voice not_deployed") from exc
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
