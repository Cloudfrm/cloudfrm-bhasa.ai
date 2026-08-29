from __future__ import annotations

import base64
import hashlib
import threading
import time
from collections import OrderedDict
from functools import lru_cache

from fastapi import APIRouter, Header, HTTPException, Response

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
# A client key only dedupes a retry that reuses it. It cannot catch the case
# that actually produced duplicates in QA: a send that reached the server but
# whose response never got back, so the client still has no conversation id
# and the next click sends the same text again under a brand-new key. Identical
# text into the same conversation within this window is treated as one message.
_CONTENT_TTL = 12.0
_idempotency_lock = threading.Lock()
_idempotency: OrderedDict[str, dict] = OrderedDict()
_content_keys: OrderedDict[str, tuple[float, str]] = OrderedDict()


def _content_key(payload: ChatRequest) -> str:
    body = " ".join((payload.message or "").split())
    return hashlib.sha1(
        f"{payload.conversation_id or 'new'}|{payload.channel or 'chat'}|{body}".encode("utf-8")
    ).hexdigest()


def _purge_expired(now: float) -> None:
    for key in [k for k, e in _idempotency.items() if now - e["created"] > _IDEMPOTENCY_TTL]:
        _idempotency.pop(key, None)
    while len(_idempotency) > _IDEMPOTENCY_MAX:
        _idempotency.popitem(last=False)


def _claim(key: str, content: str | None = None) -> tuple[dict, bool]:
    """Return (entry, is_owner). The owner does the work; others wait on it.

    `content` lets a duplicate with a different client key still find the
    first request, as long as it arrives inside the content window.
    """
    now = time.monotonic()
    with _idempotency_lock:
        _purge_expired(now)
        for stale in [k for k, (ts, _) in _content_keys.items() if now - ts > _CONTENT_TTL]:
            _content_keys.pop(stale, None)

        entry = _idempotency.get(key)
        if entry is not None:
            return entry, False

        if content:
            seen = _content_keys.get(content)
            if seen is not None:
                existing = _idempotency.get(seen[1])
                if existing is not None:
                    return existing, False

        entry = {"created": now, "done": threading.Event(), "result": None, "error": None}
        _idempotency[key] = entry
        if content:
            _content_keys[content] = (now, key)
            while len(_content_keys) > _IDEMPOTENCY_MAX:
                _content_keys.popitem(last=False)
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
    content = _content_key(payload)
    key = (idempotency_key or payload.client_id or "").strip() or ("auto-" + content)
    entry, is_owner = _claim(key, content)
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
            _content_keys.pop(content, None)
        entry["done"].set()
        raise
    except Exception as exc:
        entry["error"] = exc
        with _idempotency_lock:
            _idempotency.pop(key, None)
            _content_keys.pop(content, None)
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


CONVERSATION_PAGE_MAX = 200


@router.get("/support/conversations")
def list_conversations(
    response: Response,
    channel: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """A page of conversations, newest first.

    The response stays a bare array so existing callers are unaffected; the
    full count travels in X-Total-Count, because the inbox badge has to state
    how many conversations the desk has, not how many fitted in this page.
    """
    store = get_engine().store
    page = max(1, min(int(limit), CONVERSATION_PAGE_MAX))
    start = max(0, int(offset))
    rows = store.list_conversations(channel=channel, limit=page, offset=start)
    response.headers["X-Total-Count"] = str(store.count_conversations(channel=channel))
    response.headers["Access-Control-Expose-Headers"] = "X-Total-Count"
    return rows


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
def list_tickets(
    response: Response,
    user_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """A page of tickets, newest first, optionally filtered by status.

    Same contract as conversations: bare array, count in X-Total-Count, so a
    caller that needs "how many open tickets" gets the real number instead of
    the length of a truncated page.
    """
    store = get_engine().store
    page = max(1, min(int(limit), CONVERSATION_PAGE_MAX))
    rows = store.list_tickets(
        user_id=user_id, status=status, limit=page, offset=max(0, int(offset))
    )
    response.headers["X-Total-Count"] = str(store.count_tickets(user_id=user_id, status=status))
    response.headers["Access-Control-Expose-Headers"] = "X-Total-Count"
    return rows


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
