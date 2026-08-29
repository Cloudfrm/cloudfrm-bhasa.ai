"""Confirmed, idempotent side-effect actions."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# The app's own copy says "say yes to open a ticket", in English, and this
# only ever knew Devanagari — so a member who said yes was never confirming
# anything. "ठीक छ" could not match either: it was one entry in a set matched
# against single whitespace-split tokens.
YES = {
    "हो", "हजुर", "हुन्छ", "ल", "ठीक", "ठिक", "हवस्", "अँ", "सही",
    "yes", "yeah", "yep", "yup", "ya", "sure", "alright", "correct",
    "hunchha", "huncha", "thik", "thikcha", "hawas", "hajur", "ho",
}
# "ok" is deliberately not a confirmation. The offer says "say yes", and in
# English "ok" is at least as often an acknowledgement of the answer as an
# assent to the offer — and since almost every substantive reply now ends
# with the offer, reading "ok" as consent would fill an officer's queue with
# tickets nobody asked for. A member who meant it can still say yes. The
# Nepali "हुन्छ" and "ठीक छ" stay: those are assent, not mere acknowledgement.
NO = {
    "होइन", "हैन", "पर्दैन", "छैन", "नचाहिने", "नाइँ", "अहँ",
    "no", "nope", "nah", "not", "dont", "nothing",
    "hoina", "haina", "pardaina", "chaina",
}
TICKET_WORDS = {"ticket", "tickets", "टिकट", "टिकेट"}

# Words that carry no decision of their own.
_FILLER = {"a", "an", "the", "plz", "है", "त", "न", "म", "मलाई", "i", "it", "is", "do"}
_TOKEN = re.compile(r"[\wऀ-ॿ']+", re.UNICODE)

# A confirmation has to be essentially the whole message. Without this rule,
# widening the vocabulary to English would make "ok so my card is blocked"
# open a ticket, and "I have not received my money" refuse one — both of them
# real messages that happen to contain a decision word.
_MAX_DECISION_WORDS = 3


def parse_confirmation(text: str) -> bool | None:
    """True for yes, False for no, None when this is a real message.

    Matched on what the member typed, before transliteration, for the same
    reason crisis and small-talk matching is: transliteration turns "yes"
    into Devanagari and the decision is lost.
    """
    words = [word.lower().replace("'", "") for word in _TOKEN.findall(text or "")]
    content = [word for word in words if word not in _FILLER]
    if not content:
        return None
    # "yes please open a ticket" names the thing being confirmed, so length
    # stops mattering.
    about_ticket = any(word in TICKET_WORDS for word in content)
    decisive = about_ticket or len(content) <= _MAX_DECISION_WORDS
    if not decisive:
        return None
    if any(word in NO for word in content):
        return False
    if any(word in YES for word in content):
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
