from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    # "auto" = reply language follows the question (E10). "ne"/"en" force it.
    locale: str = Field(default="auto")
    channel: str = Field(default="chat")


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    language: str
    # "answer" | "refusal" | "credential_decline"
    kind: str = "answer"
    refusal_type: str | None = None
    passage: dict[str, Any] | None = None
    # What the thread should show for the member turn (redacted when a credential was detected).
    echo: str = ""
    credential_kinds: list[str] = Field(default_factory=list)
    grounded: bool = True
    # True when a generative model wrote `reply` (the passage is then provenance, not the reply).
    generated: bool = False
    model: str | None = None
    backend: str | None = None  # "ollama" | "openai_compat" | … | "extractive" | "extractive_fallback"
    note: str | None = None  # e.g. "llm_unreachable", "ungrounded_quantity: …"
    suggest_ticket: bool = False
    considered: list[dict[str, Any]] = Field(default_factory=list)


class UnicoderRequest(BaseModel):
    text: str = Field(default="", max_length=4000)


class CandidatesRequest(BaseModel):
    text: str = Field(default="", max_length=4000)
    choices: dict[str, str] = Field(default_factory=dict)


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
