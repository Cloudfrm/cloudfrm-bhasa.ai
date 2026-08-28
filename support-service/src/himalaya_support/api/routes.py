from __future__ import annotations

import base64
import threading
import time
from collections import OrderedDict
from functools import lru_cache

from fastapi import APIRouter, Header, HTTPException

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


# --------------------------------------------------------------------------
# Idempotency for POST /support/chat.
#
# Two fast clicks on a topic chip produced two conversations with identical
# previews. Caching only completed responses does not fix that: the duplicate
# arrives while the first request is still running. So an entry is claimed
# before the work starts, and a second caller with the same key waits for the
# first result instead of starting its own.
#
# In-memory and per-process, which is enough for the single-worker deployment
# this runs on. A shared store is needed before scaling to multiple workers.
# --------------------------------------------------------------------------
_IDEMPOTENCY_TTL = 300.0
_IDEMPOTENCY_MAX = 512
_idempotency_lock = threading.Lock()
_idempotency: OrderedDict[str, dict] = OrderedDict()


def _purge_expired(now: float) -> None:
    for key in [k for k, e in _idempotency.items() if now - e["created"] > _IDEMPOTENCY_TTL]:
        _idempotency.pop(key, None)
    while len(_idempotency) > _IDEMPOTENCY_MAX:
        _idempotency.popitem(last=False)


def _claim(key: str) -> tuple[dict, bool]:
    """Return (entry, is_owner). The owner does the work; others wait on it."""
    now = time.monotonic()
    with _idempotency_lock:
        _purge_expired(now)
        entry = _idempotency.get(key)
        if entry is not None:
            return entry, False
        entry = {"created": now, "done": threading.Event(), "result": None, "error": None}
        _idempotency[key] = entry
        return entry, True


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.post("/support/unicoder")
def convert_unicoder(payload: UnicoderRequest) -> dict:
    result = unicoder(payload.text)
    return {"nepali": result["nepali"], "mode": result["mode"]}


def _run_chat(payload: ChatRequest) -> ChatResponse:
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


@router.post("/support/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatResponse:
    key = (idempotency_key or payload.client_id or "").strip()
    if not key:
        return _run_chat(payload)

    entry, is_owner = _claim(key)
    if not is_owner:
        # A duplicate of a request already running or finished: wait for the
        # first result and return it, rather than creating a second
        # conversation for what the member sent once.
        entry["done"].wait(timeout=120)
        if entry["error"] is not None:
            raise entry["error"]
        if entry["result"] is not None:
            return entry["result"]
        raise HTTPException(status_code=503, detail="Support is temporarily unavailable")

    try:
        entry["result"] = _run_chat(payload)
    except HTTPException as exc:
        # Do not cache a failure: the officer must be able to retry.
        entry["error"] = exc
        with _idempotency_lock:
            _idempotency.pop(key, None)
        entry["done"].set()
        raise
    except Exception as exc:
        entry["error"] = exc
        with _idempotency_lock:
            _idempotency.pop(key, None)
        entry["done"].set()
        raise
    entry["done"].set()
    return entry["result"]


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
