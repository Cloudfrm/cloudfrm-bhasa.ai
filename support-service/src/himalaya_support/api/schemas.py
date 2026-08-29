from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # What the member keyed, before the composer transliterated it. The desk
    # converts Latin typing into Devanagari on the way out, so by the time a
    # message arrives "I want to die" has become "म चाहन्छु दिए" — which the
    # crisis, courtesy and yes/no layers cannot match, because they are all
    # meant to read what was typed. Optional: an API caller that sends only
    # `message` behaves exactly as before.
    typed: str | None = Field(default=None, max_length=4000)
    # How the member is typing, chosen by them at the composer:
    #   romanized — Latin letters become Devanagari (the desk's original and,
    #               until now, only behaviour)
    #   english   — what is typed is what is sent, stored and shown
    #   unicode   — the member has a Devanagari keyboard; nothing is converted
    # Defaults to romanized so an API caller that omits it behaves as before.
    input_mode: str = Field(default="romanized")
    conversation_id: str | None = None
    locale: str = Field(default="ne")
    channel: str = Field(default="chat")
    # Generated once per message by the client and reused on retry, so a
    # duplicate click or a retried send cannot create a second conversation.
    # The Idempotency-Key header takes precedence over this field.
    client_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    language: str
    transliterated: str | None = None
    grounded: bool = True
    speech_register: str | None = None
    tickets: list[str] = Field(default_factory=list)
    pending_confirm: str | None = None


class UnicoderRequest(BaseModel):
    text: str = Field(default="", max_length=4000)


class CallStartRequest(BaseModel):
    locale: str = "ne"


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    locale: str = "ne"


class TicketCreateRequest(BaseModel):
    subject: str
    description: str
    category: str = "other"
    priority: str = "normal"
    user_id: str | None = None
    conversation_id: str | None = None


class TicketUpdateRequest(BaseModel):
    status: str | None = None
    note: str | None = None
    priority: str | None = None
