"""Turn Latin-script typing into Nepali Devanagari Unicode.

Two paths:
  * Romanized Nepali  (mero pin birse) → phonetic/lexicon transliteration
  * Actual English    (I forgot my PIN) → phrase/word translation, then Unicode
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from himalaya_support.adapt.devanagari import normalize
from himalaya_support.adapt.translit import classify_input, to_devanagari

_DATA = Path(__file__).resolve().parents[3] / "data" / "knowledge" / "english_nepali.json"
_WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']+")
_LATIN_RE = re.compile(r"[A-Za-z]")


@lru_cache(maxsize=1)
def _tables() -> dict:
    if not _DATA.exists():
        return {"phrases": {}, "words": {}, "drop": []}
    payload = json.loads(_DATA.read_text(encoding="utf-8"))
    phrases = {
        str(key).lower(): str(value)
        for key, value in (payload.get("phrases") or {}).items()
        if str(value).strip()
    }
    words = {
        str(key).lower(): str(value)
        for key, value in (payload.get("words") or {}).items()
    }
    drop = {str(item).lower() for item in (payload.get("drop") or [])}
    ordered = sorted(phrases.items(), key=lambda item: len(item[0]), reverse=True)
    return {"phrases": phrases, "words": words, "drop": drop, "ordered": ordered}


# A token mixing letters and digits is a reference, not romanized Nepali:
# TXN-1001, 24x7, A1B2. Transliterating them produced ट्क्ष्ण-1001 and 24क्ष7,
# which destroys exactly the detail a member pastes when reporting a failed
# transfer. No Nepali word fuses digits into letters, so the rule is safe —
# and it deliberately leaves letter-only words like PIN alone, which should
# still become पिन.
_ID_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-/_][A-Za-z0-9]+)*")
_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")
_PUA_BASE = 0xE000
_MAX_PROTECTED = 256


def _protect_identifiers(text: str) -> tuple[str, list[str]]:
    """Swap identifiers for private-use placeholders during conversion.

    Private-use characters carry no letters or digits, so neither the
    transliteration tables nor Unicode normalisation touch them.
    """
    saved: list[str] = []

    def swap(match: re.Match) -> str:
        token = match.group(0)
        if (
            len(saved) < _MAX_PROTECTED
            and _HAS_LETTER.search(token)
            and _HAS_DIGIT.search(token)
        ):
            saved.append(token)
            return chr(_PUA_BASE + len(saved) - 1)
        return token

    return _ID_TOKEN.sub(swap, text), saved


def _restore_identifiers(text: str, saved: list[str]) -> str:
    for index, token in enumerate(saved):
        text = text.replace(chr(_PUA_BASE + index), token)
    return text


def latin_to_nepali(text: str) -> dict[str, str]:
    """Convert English letters into Nepali Unicode. Already-Devanagari text is left alone."""
    raw = (text or "").strip()
    if not raw:
        return {"out": "", "mode": "empty", "script": "empty"}
    if not _LATIN_RE.search(raw):
        return {"out": normalize(raw), "mode": "devanagari", "script": "devanagari"}

    script = classify_input(raw)
    tables = _tables()
    compact = re.sub(r"[^\w\s']+", "", raw.lower()).strip()
    if compact in tables["phrases"]:
        mapped = tables["phrases"][compact]
        prefix = re.match(r"^[^A-Za-z]+", raw)
        suffix = re.search(r"[^A-Za-z]+$", raw)
        if prefix:
            mapped = prefix.group(0) + mapped
        if suffix:
            mapped = mapped + suffix.group(0)
        return {"out": normalize(mapped), "mode": "translate", "script": "english"}
    protected, saved = _protect_identifiers(raw)
    if script in {"romanized_nepali", "mixed"}:
        converted = to_devanagari(protected, force=True).out
        return {
            "out": _restore_identifiers(normalize(converted), saved),
            "mode": "translit",
            "script": script,
        }

    translated = _translate_english(protected)
    return {
        "out": _restore_identifiers(normalize(translated), saved),
        "mode": "translate",
        "script": "english",
    }


def _translate_english(text: str) -> str:
    tables = _tables()
    work = f" {text.strip()} "
    lowered = work.lower()
    for phrase, nepali in tables["ordered"]:
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])", re.IGNORECASE)
        lowered = pattern.sub(lambda _m, n=nepali: n, lowered)
        work = pattern.sub(lambda _m, n=nepali: n, work)

    parts: list[str] = []
    for token in _WORD_RE.findall(work):
        if not re.fullmatch(r"[A-Za-z']+", token):
            parts.append(token)
            continue
        key = token.lower()
        if key in tables["drop"]:
            continue
        mapped = tables["words"].get(key)
        if mapped:
            parts.append(mapped)
            continue
        phonetic = to_devanagari(token, force=True).out
        parts.append(phonetic if phonetic else token)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def unicoder(text: str) -> dict[str, str]:
    """Turn Latin letters in the composer into Nepali Unicode, including the current word."""
    raw = text or ""
    if not raw:
        return {"nepali": "", "mode": "empty"}
    if not _LATIN_RE.search(raw):
        return {"nepali": normalize(raw), "mode": "devanagari"}
    trail = re.search(r"\s+$", raw)
    trail_ws = trail.group(0) if trail else ""
    converted = latin_to_nepali(raw)
    nepali = converted["out"]
    if _LATIN_RE.search(nepali):
        nepali = normalize(to_devanagari(raw, force=True).out)
        if trail_ws and not nepali.endswith(trail_ws):
            nepali += trail_ws
        return {"nepali": nepali, "mode": "translit"}
    if trail_ws and not nepali.endswith(trail_ws):
        nepali += trail_ws
    return {"nepali": nepali, "mode": converted["mode"]}
