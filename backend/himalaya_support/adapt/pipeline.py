"""Glue: his language/safety stack on top of our Gemma replies."""
from __future__ import annotations

import re
from typing import Any

from himalaya_support.adapt.devanagari import normalize
from himalaya_support.adapt.grounding import GROUNDING_FALLBACK, check_numeric_grounding
from himalaya_support.adapt.honorific import check_honorific
from himalaya_support.adapt.register import classify
from himalaya_support.adapt.to_nepali import latin_to_nepali

GENERIC_NE = "कुन विषय हो"
GENERIC_EN = "Which topic can I help with"


def prepare_user_text(message: str) -> dict[str, Any]:
    raw = (message or "").strip()
    normalized = normalize(raw)
    converted = latin_to_nepali(normalized)
    nepali = converted["out"] or normalized
    return {
        "raw": raw,
        "normalized": normalized,
        "script": converted["script"],
        "mode": converted["mode"],
        "transliterated": nepali if nepali != raw else None,
        "search": nepali,
    }


def evidence_text(snippets: list[dict]) -> str:
    parts = []
    for item in snippets or []:
        parts.append(item.get("title") or "")
        parts.append(item.get("text") or "")
    return " ".join(parts)


def compose_from_knowledge(message: str, snippets: list[dict], language: str) -> str:
    bodies = []
    for item in snippets[:1]:
        body = re.sub(r"\s+", " ", (item.get("text") or "").strip())
        if body and body not in bodies:
            bodies.append(body)
    facts = " ".join(bodies).strip()
    asked = re.sub(r"\s+", " ", (message or "").strip())[:180]
    if language == "ne":
        if facts:
            return (
                f"हजुर, यसरी पूरा गर्नुहोस्। {facts} "
                "यति गर्दा पनि नखुले शाखामा परिचयपत्र लिएर जानुहोस्। "
                "पिन, पासवर्ड वा ओटीपी यहाँ नलेख्नुहोस्।"
            )
        return (
            f"तपाईंले लेख्नुभएको कुरा बुझें: “{asked}”। "
            "लगइन, रकम, ऋण, केवाईसी, बचत, कार्ड वा शाखा समयमा कदम-कदममा भन्न सक्छु। "
            "थोरै थप लेख्नुहोस्, वा टिकट खोलौं हो?"
        )
    if facts:
        return (
            f"Here is the full procedure. {facts} "
            "If that still fails, visit the branch with original ID, or say yes to open a ticket. "
            "Do not type PIN, password, or OTP here."
        )
    return (
        f"I understood: “{asked}”. I can walk through login, transfers, loans, KYC, savings, cards, or hours. "
        "Add one more sentence, or say yes and I will open a ticket."
    )


def is_thin_reply(reply: str, language: str, intent: str) -> bool:
    text = (reply or "").strip()
    if not text:
        return True
    if GENERIC_NE in text or GENERIC_EN in text:
        return True
    if intent == "greeting":
        return len(text) < 8
    letters = re.findall(r"[\w\u0900-\u097F]", text)
    return len(letters) < (48 if language == "ne" else 40)


def finish_reply(
    reply: str,
    snippets: list[dict],
    language: str,
    intent: str = "other",
    user_message: str = "",
) -> tuple[str, dict[str, Any]]:
    text = (reply or "").strip()
    if is_thin_reply(text, language, intent):
        text = compose_from_knowledge(user_message, snippets, language)
    evidence = evidence_text(snippets)
    grounded, failures = check_numeric_grounding(text, evidence, [])
    if not grounded:
        if snippets:
            retry = compose_from_knowledge(user_message, snippets, language)
            grounded, failures = check_numeric_grounding(retry, evidence, [])
            if grounded:
                text = retry
            elif language == "ne":
                text = GROUNDING_FALLBACK
            else:
                text = (
                    "I cannot confirm that amount from our records. "
                    "Ask an officer, or say yes to open a ticket."
                )
        elif language == "ne":
            text = GROUNDING_FALLBACK
        else:
            text = (
                "I cannot confirm that amount from our records. "
                "Ask an officer, or say yes to open a ticket."
            )
    honorific = check_honorific(text, "high") if language == "ne" else {"ok": True, "reason": "english"}
    register = classify(text) if language == "ne" else {"register": "english"}
    return text, {
        "grounded": grounded,
        "grounding_failures": failures,
        "honorific_ok": honorific.get("ok", True),
        "honorific_reason": honorific.get("reason"),
        "register": register.get("register"),
    }
