from __future__ import annotations

import json
import re
from typing import Any

from himalaya_support.store.db import SupportStore

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        if name:
            calls.append({"name": name, "arguments": arguments})
    return calls


def strip_tool_markup(text: str) -> str:
    cleaned = TOOL_CALL_RE.sub("", text).strip()
    return cleaned


class ToolRunner:
    """Execute Hermes-style tool calls from himalaya-ai/nepali-hermes-function-calling-v1."""

    def __init__(self, store: SupportStore) -> None:
        self.store = store

    def run(self, name: str, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "create_ticket": self._create_ticket,
            "escalate_to_human": self._escalate,
            "update_ticket": self._update_ticket,
            "lookup_knowledge": self._lookup_knowledge,
        }.get(name)
        if not handler:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        return handler(arguments, context)

    def _create_ticket(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        ticket = self.store.create_ticket(
            {
                "conversation_id": context.get("conversation_id"),
                "user_id": context.get("user_id"),
                "subject": arguments.get("subject"),
                "description": arguments.get("description") or context.get("user_message"),
                "category": arguments.get("category") or context.get("intent") or "other",
                "priority": arguments.get("priority") or "normal",
            }
        )
        return {"ok": True, "ticket": ticket}

    def _escalate(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        ticket_id = arguments.get("ticket_id")
        if ticket_id:
            ticket = self.store.update_ticket(ticket_id, status="escalated")
        else:
            ticket = self.store.create_ticket(
                {
                    "conversation_id": context.get("conversation_id"),
                    "user_id": context.get("user_id"),
                    "subject": "Escalated conversation",
                    "description": arguments.get("reason") or context.get("user_message"),
                    "category": context.get("intent") or "other",
                    "priority": "high",
                }
            )
            if ticket:
                ticket = self.store.update_ticket(ticket["id"], status="escalated")
        return {"ok": True, "escalated": True, "reason": arguments.get("reason"), "ticket": ticket}

    def _update_ticket(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        ticket_id = arguments.get("ticket_id")
        if not ticket_id:
            return {"ok": False, "error": "ticket_id is required"}
        note = arguments.get("note")
        ticket = self.store.update_ticket(
            ticket_id,
            status=arguments.get("status"),
            description=note,
        )
        if not ticket:
            return {"ok": False, "error": f"Ticket {ticket_id} not found"}
        return {"ok": True, "ticket": ticket}

    def _lookup_knowledge(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        retriever = context.get("retriever")
        query = arguments.get("query") or context.get("user_message") or ""
        hits = retriever.search(query, k=4) if retriever else []
        return {"ok": True, "hits": hits}
