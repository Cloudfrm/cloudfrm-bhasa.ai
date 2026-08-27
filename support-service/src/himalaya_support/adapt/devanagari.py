"""
Devanagari normalisation for STT output, RAG queries and user input.

Policy (E11): NFC only. NFKC is never used on user text, passages or refusal
strings. Measured earlier, when NFKC was still in use:

    क़ (U+0958)  --NFKC-->  क + ़ (U+0915 U+093C)      nukta unified   ✓
    क + ZWNJ + ख --NFKC-->  unchanged                   ZWJ survives    ✗
    २०८२        --NFKC-->  unchanged                   digits survive  ✗

ZWJ/ZWNJ are format characters, not compatibility characters, so NFKC never
touches them — and 'नेपाल' vs 'नेपा‌ल' remain different strings that embed
to different vectors. Stripping them is a separate, explicit step.

Digit folding is deliberately NOT applied to indexed text: Devanagari numerals
carry meaning in dates and amounts, and folding them at index time makes the
stored passage differ from the source document. Fold at TTS time instead.
"""
import re
import unicodedata

# ZWSP, ZWNJ, ZWJ, LRM/RLM, directional isolates/embeddings, WORD JOINER, BOM
_ZERO_WIDTH = re.compile("[​-‏‪-‮⁠-⁯﻿]")
_WHITESPACE = re.compile(r"\s+")
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ASCII_TO_DEVA = str.maketrans("0123456789", "०१२३४५६७८९")

# Danda spacing: writers put a space before । about as often as not.
_DANDA_SPACE = re.compile(r"\s+([।॥])")


def normalize(text: str, *, fold_digits: bool = False, strip_zero_width: bool = True) -> str:
    """Canonical form for embedding, retrieval and comparison.

    NFC only — never NFKC (E11). NFC already decomposes the precomposed nukta
    forms (U+0958–U+095F are composition exclusions), so nothing is lost, and
    compatibility folding is never applied to text that must stay byte-exact.
    Then explicit format-character removal, then whitespace and danda tidying.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFC", text)
    if strip_zero_width:
        out = _ZERO_WIDTH.sub("", out)
    if fold_digits:
        out = out.translate(_DEVA_DIGITS)
    out = _DANDA_SPACE.sub(r"\1", out)
    return _WHITESPACE.sub(" ", out).strip()


def normalize_for_index(text: str) -> str:
    """Passages going into Qdrant. Digits preserved — they are content."""
    return normalize(text, fold_digits=False)


def normalize_for_query(text: str) -> str:
    """Query side. Must use the same folding as the index or recall silently drops."""
    return normalize(text, fold_digits=False)


def to_ascii_digits(text: str) -> str:
    return text.translate(_DEVA_DIGITS)


def to_devanagari_digits(text: str) -> str:
    return text.translate(_ASCII_TO_DEVA)


def diff_report(a: str, b: str) -> dict[str, object]:
    """Explain why two visually identical strings are not equal."""
    na, nb = normalize(a), normalize(b)
    return {
        "raw_equal": a == b,
        "normalized_equal": na == nb,
        "a_codepoints": [f"U+{ord(c):04X}" for c in a],
        "b_codepoints": [f"U+{ord(c):04X}" for c in b],
        "cause": (
            "identical" if a == b
            else "zero-width format characters" if _ZERO_WIDTH.sub("", a) == _ZERO_WIDTH.sub("", b)
            else "nukta composition" if unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)
            else "genuinely different text"
        ),
    }
