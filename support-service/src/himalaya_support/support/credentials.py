"""Credential-shaped input guard.

Detects PIN/OTP/card/CVV/password-shaped content BEFORE it is stored or
answered. Detected content is redacted before it touches the store, and the
raw text is never logged. The reply declines and repeats the never-share rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_DEVA_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")

# Units / contexts that make a digit run a quantity, not a secret.
_QUANTITY_AFTER = re.compile(
    r"^\s*(?:%|प्रतिशत|percent|रुपैयाँ|रुपियाँ|रु\.?|rs\.?|rupees?|paisa|पैसा|npr|usd|dollars?|मिनेट|minutes?|mins?|दिन|days?|"
    r"घण्टा|hours?|hrs?|बजे|वर्ष|years?|पटक|times|अङ्क|digits?|अक्षर|characters?|"
    r"महिना|months?|km|kg|साल|गते)",
    re.IGNORECASE,
)
_QUANTITY_BEFORE = re.compile(r"(?:रु\.?|रुपैयाँ|rs\.?|npr|usd|\$|₹|#|no\.?|नं\.?|number|नम्बर)\s*$", re.IGNORECASE)
_DATE_LIKE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")

_CARD_RUN = re.compile(r"(?<![\d\w])(?:\d[ -]?){13,19}(?![\d\w])")
_CVV = re.compile(r"\b(?:cvv|cvc|cvv2|सीभीभी)\s*[:=\-]?\s*\d{3,4}\b", re.IGNORECASE)
_PASSWORD = re.compile(
    r"(?:\b(?:password|passwd|pwd|pass|पासवर्ड)\b|पासवर्ड)\s*(?:is|:|=|-|हो|चाहिँ)?\s*[\"“']?([^\s\"”']{4,})",
    re.IGNORECASE,
)
_PIN_OTP_WORD = re.compile(r"\b(?:pin|otp|mpin|tpin|पिन|ओटीपी|ओटिपी)\b", re.IGNORECASE)
_DIGIT_RUN = re.compile(r"(?<![\d.,/-])\d{4,8}(?![\d.,/-])")

REDACTION = "[•••]"


@dataclass
class CredentialCheck:
    detected: bool
    kinds: list[str] = field(default_factory=list)
    redacted: str = ""


def _ascii_digits(text: str) -> str:
    return text.translate(_DEVA_TO_ASCII)


def _looks_like_quantity(text: str, start: int, end: int) -> bool:
    after = text[end : end + 14]
    before = text[max(0, start - 12) : start]
    return bool(_QUANTITY_AFTER.match(after)) or bool(_QUANTITY_BEFORE.search(before))


def check_credentials(text: str) -> CredentialCheck:
    """Return whether `text` contains credential-shaped content plus a redacted copy."""
    raw = text or ""
    work = _ascii_digits(raw)  # Devanagari digits count too
    kinds: list[str] = []
    spans: list[tuple[int, int]] = []

    for match in _CARD_RUN.finditer(work):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19:
            kinds.append("card")
            spans.append(match.span())

    for match in _CVV.finditer(work):
        kinds.append("cvv")
        spans.append(match.span())

    for match in _PASSWORD.finditer(work):
        kinds.append("password")
        spans.append(match.span(1))

    has_pin_word = bool(_PIN_OTP_WORD.search(work))
    for match in _DIGIT_RUN.finditer(work):
        start, end = match.span()
        if any(s <= start < e for s, e in spans):
            continue
        if _looks_like_quantity(work, start, end):
            continue
        token = work[start:end]
        if _DATE_LIKE.match(token):
            continue
        # A bare 4–8 digit run: only PIN/OTP-shaped if the message talks about
        # a PIN/OTP, or the run is 6 digits (the OTP length used by the desk).
        if has_pin_word or len(token) in (4, 6):
            kinds.append("pin_otp")
            spans.append((start, end))

    if not spans:
        return CredentialCheck(False, [], raw)

    # Redact on the original string; spans are index-identical because the
    # digit translation is 1:1 in code points.
    chars = list(raw)
    for start, end in sorted(set(spans), reverse=True):
        chars[start:end] = list(REDACTION)
    return CredentialCheck(True, sorted(set(kinds)), "".join(chars))


DECLINE = {
    "ne": (
        "यो सन्देशमा पिन, ओटीपी, कार्ड नम्बर वा पासवर्ड जस्तो देखिने विवरण छ, त्यसैले यसलाई "
        "प्रशोधन गरिएन र सुरक्षित गरिएन। पिन, पासवर्ड, ओटीपी वा सीभीभी यहाँ कहिल्यै नलेख्नुहोस् — "
        "bhasa ले ती कहिल्यै माग्दैन। प्रश्न मात्र फेरि लेख्नुहोस्।"
    ),
    "en": (
        "This message contains something shaped like a PIN, OTP, card number, or password, so it "
        "was not processed or stored. Never type a PIN, password, OTP, or CVV here — bhasa never "
        "asks for them. Please resend the question without it."
    ),
}
