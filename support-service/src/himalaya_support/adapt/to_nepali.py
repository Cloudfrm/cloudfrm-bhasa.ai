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
    if script in {"romanized_nepali", "mixed"}:
        converted = to_devanagari(raw, force=True).out
        return {"out": normalize(converted), "mode": "translit", "script": script}

    translated = _translate_english(raw)
    return {"out": normalize(translated), "mode": "translate", "script": "english"}


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
