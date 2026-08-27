"""Confirmed, idempotent side-effect actions."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

YES = {"हो", "हजुर", "हुन्छ", "ल", "ठीक छ"}
NO = {"होइन", "हैन", "पर्दैन"}


def parse_confirmation(text: str) -> bool | None:
    words = set((text or "").replace("।", "").replace(",", "").split())
    if words & NO:
        return False
    if words & YES:
        return True
    return None


def idempotency_key(session_id: str, turn_index: int, action: str, params: dict[str, Any]) -> str:
    raw = json.dumps([session_id, turn_index, action, params], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class PendingAction:
    key: str
    readback: str
    action: str
    params: dict[str, Any]
    confirmed: bool = False


class ActionExecutor:
    def __init__(self) -> None:
        self.pending: dict[str, PendingAction] = {}
        self.executed: set[str] = set()

    def prepare(self, session_id: str, turn_index: int, action: str,
                params: dict[str, Any], readback: str) -> PendingAction:
        key = idempotency_key(session_id, turn_index, action, params)
        pending = PendingAction(key, readback, action, params)
        self.pending[key] = pending
        return pending

    def confirm(self, pending: PendingAction, answer: str, execute: Callable[[str, dict[str, Any]], None]) -> bool:
        if parse_confirmation(answer) is not True or pending.key in self.executed:
            return False
        pending.confirmed = True
        execute(pending.action, pending.params)
        self.executed.add(pending.key)
        return True

    def abandon(self, pending: PendingAction) -> None:
        self.pending.pop(pending.key, None)
