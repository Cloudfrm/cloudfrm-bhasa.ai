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

# Knowledge rows are stored as "प्रश्न: <question>\nजवाफ: <answer>". Only the
# answer may ever reach an officer: splicing the row in whole showed the
# corpus's own question back as if Bhasa had asked it.
_QA_ROW = re.compile(r"प्रश्न\s*:\s*(?P<question>.*?)\s*(?:\n+|\s)जवाफ\s*:\s*(?P<answer>.*)", re.S)
_LABEL = re.compile(r"^\s*(?:प्रश्न|जवाफ)\s*:\s*", re.M)

# Text that must never appear in a reply. These are our own scaffolding, so
# their presence in an outgoing answer means the reply was assembled wrong.
TEMPLATE_MARKERS = ("हजुर, यसरी पूरा गर्नुहोस्", "प्रश्न:", "जवाफ:")

# Some knowledge rows are behavioural rules aimed at Bhasa, not facts for the
# member — the greeting row reads "when the member says नमस्ते ... reply
# respectfully ... offer help with login, transfers". Shown verbatim it hands
# the member our own instructions. These rows carry a tag, but the phrasing is
# checked too so a corpus without tags is still covered: a row that says what
# to do *when the member says something* is a rule, never an answer.
DIRECTIVE_TAGS = {"greeting", "smalltalk", "instruction", "style", "persona"}
_DIRECTIVE_RULE = re.compile(r"सदस्यले.{0,80}?भने", re.S)


def is_directive_row(text: str, tags: list[str] | tuple[str, ...] | None = None) -> bool:
    if tags and DIRECTIVE_TAGS.intersection({str(t).lower() for t in tags}):
        return True
    return bool(_DIRECTIVE_RULE.search(text or ""))

_STOPWORDS_NE = {
    "हो", "हुन्छ", "छ", "छैन", "के", "कति", "कसरी", "मेरो", "मलाई", "गर्ने",
    "गर्न", "भएमा", "पनि", "वा", "र", "यो", "त्यो", "म", "तपाईं",
}


class ReplyGenerationError(RuntimeError):
    """The assembled reply was unusable and must not be shown to an officer."""


def contains_template_markers(text: str) -> list[str]:
    return [marker for marker in TEMPLATE_MARKERS if marker in (text or "")]


def split_knowledge_row(text: str) -> tuple[str, str]:
    """Return (question, answer) for a stored row; question is "" for prose."""
    raw = re.sub(r"\s+", " ", (text or "").strip())
    match = _QA_ROW.search(raw)
    if match:
        return match.group("question").strip(), match.group("answer").strip()
    return "", _LABEL.sub("", raw).strip()


# Every Q&A row states the question it answers, which is a free signal that
# word overlap throws away: "ऋण किस्ता कहिले तिर्ने?" (when) was answered by the
# row for "कर्जाको किस्ता तिर्न ढिलो भएमा के हुन्छ?" (what happens) purely because
# both repeat किस्ता. Asking *when* and asking *what happens* are different
# questions, and the corpus says so.
_QUESTION_KINDS = {
    "कहिले": "when",
    "कति": "quantity",
    "कतिको": "quantity",
    "कसरी": "how",
    "कहाँ": "where",
    "कता": "where",
    "किन": "why",
    "को": "who",
    "कुन": "which",
    "के": "what",
}


def question_kinds(text: str) -> set[str]:
    """Interrogatives present, matched whole-word.

    Token matching matters: के is a substring of केवाईसी and केही, and
    substring matching would call every KYC question a "what" question.
    """
    tokens = re.findall(r"[\wऀ-ॿ]+", (text or "").lower())
    return {_QUESTION_KINDS[tok] for tok in tokens if tok in _QUESTION_KINDS}


def asks_a_different_question(stored_question: str, asked: str) -> bool:
    """True only when both sides name a question type and they disagree.

    Deliberately conservative — a prose row states no question, and a member
    who asks nothing interrogative ("मेरो पिन बिर्सें") is never re-ranked.
    """
    if not stored_question:
        return False
    mine = question_kinds(asked)
    theirs = question_kinds(stored_question)
    if not mine or not theirs:
        return False
    return mine.isdisjoint(theirs)


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[\wऀ-ॿ]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS_NE}


def snippet_answers(question: str, asked: str) -> bool:
    """Does a Q&A row plausibly answer what was asked?

    A row retrieved for a different question ("how many wrong PIN attempts
    block a card?") was being presented as the answer to "I forgot my PIN",
    stated confidently and marked grounded. When the stored question shares
    no content words with the member's, say so instead of guessing.
    """
    if not question:
        return True  # prose rows carry no question to disagree with
    asked_words = _content_words(asked)
    if not asked_words:
        return True
    overlap = asked_words & _content_words(question)
    return bool(overlap)


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
    asked = re.sub(r"\s+", " ", (message or "").strip())[:180]
    facts = ""
    for item in snippets[:1]:
        raw = item.get("text") or ""
        # A rule about how to answer is not an answer.
        if is_directive_row(raw, item.get("tags")):
            continue
        question, answer = split_knowledge_row(raw)
        # Only use the row when it actually addresses what was asked; a
        # confident answer to a different question is worse than none.
        if answer and snippet_answers(question, asked):
            facts = answer
    if language == "ne":
        if facts:
            # No "here is how to do it" lead-in: the row may be a fact, not a
            # procedure, and asserting otherwise misleads the officer.
            return (
                f"{facts} "
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
            f"{facts} "
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
    # Last gate before an officer sees this. Scaffolding in the output means
    # the reply was assembled wrong, so it is a generation failure — never a
    # valid answer, and never grounded.
    leaked = contains_template_markers(text)
    if leaked:
        raise ReplyGenerationError(
            "reply contained template markers: " + ", ".join(leaked)
        )
    if is_directive_row(text):
        raise ReplyGenerationError("reply was a behavioural instruction, not an answer")
    honorific = check_honorific(text, "high") if language == "ne" else {"ok": True, "reason": "english"}
    register = classify(text) if language == "ne" else {"register": "english"}
    return text, {
        "grounded": grounded,
        "grounding_failures": failures,
        "honorific_ok": honorific.get("ok", True),
        "honorific_reason": honorific.get("reason"),
        "register": register.get("register"),
    }
