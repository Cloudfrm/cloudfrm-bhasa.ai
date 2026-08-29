"""Speech-side text cleanup from himalaya-voice-engine TTS (no extra providers)."""
from __future__ import annotations

import re

from himalaya_support.adapt.devanagari import to_ascii_digits

_MARKDOWN = re.compile(r"(\*\*|__|\*|`{1,3}|^#{1,6}\s+|^\s*[-•]\s+)", re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]")
_ABBREV = {
    "रु.": "रुपैयाँ",
    "रू.": "रुपैयाँ",
    "डा.": "डाक्टर",
    "प्रा.": "प्राध्यापक",
    "नं.": "नम्बर",
    "वि.सं.": "विक्रम सम्बत्",
    "इ.सं.": "ईस्वी सम्बत्",
    "प्र.म.": "प्रधानमन्त्री",
}


def normalize_for_speech(text: str) -> str:
    text = _URL.sub("लिङ्क", text or "")
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    for short, long in _ABBREV.items():
        text = text.replace(short, long)
    text = to_ascii_digits(text)
    return re.sub(r"\s+", " ", text).strip()
