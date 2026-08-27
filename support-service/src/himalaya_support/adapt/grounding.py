"""Guard quantity-bearing replies against unsupported retrieved facts."""
from __future__ import annotations

import re

DIGIT_WORDS = {
    "शून्य": 0, "एक": 1, "दुई": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "छ": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "एघार": 11, "बाह्र": 12, "तेह्र": 13, "चौध": 14,
    "पन्ध्र": 15, "सोह्र": 16, "सत्र": 17, "अठार": 18, "उन्नाइस": 19,
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "साठी": 60,
    "सत्तरी": 70, "असी": 80, "नब्बे": 90,
}
UNITS = {"सय": 100, "हजार": 1_000, "लाख": 100_000, "करोड": 10_000_000}
_DIGITS = re.compile(r"(?<!\w)[०-९0-9][०-९0-9,]*(?!\w)")
_NUMBER_WORDS = re.compile(
    r"(?:शून्य|एक|दुई|तीन|चार|पाँच|छ|सात|आठ|नौ|दस|एघार|बाह्र|तेह्र|चौध|"
    r"पन्ध्र|सोह्र|सत्र|अठार|उन्नाइस|बीस|तीस|चालीस|पचास|साठी|सत्तरी|असी|नब्बे|"
    r"सय|हजार|लाख|करोड)(?:\s+(?:शून्य|एक|दुई|तीन|चार|पाँच|छ|सात|आठ|नौ|दस|"
    r"एघार|बाह्र|तेह्र|चौध|पन्ध्र|सोह्र|सत्र|अठार|उन्नाइस|बीस|तीस|चालीस|पचास|"
    r"साठी|सत्तरी|असी|नब्बे|सय|हजार|लाख|करोड)){0,3}(?:\s+रुपैयाँ)?")
_ORDINALS = {"पहिलो", "दोस्रो", "तेस्रो", "चौथो", "पाँचौँ", "छैटौँ"}
# The refusal strings are fetched, never retyped here (see support/refusals.py).


def _digit_value(value: str) -> int:
    return int(value.replace(",", "").translate(str.maketrans("०१२३४५६७८९", "0123456789")))


def _words_value(phrase: str) -> int | None:
    words = [word for word in phrase.split() if word != "रुपैयाँ"]
    if not words or all(word in _ORDINALS for word in words) or words == ["छ"]:
        return None
    total = 0
    current = 0
    for word in words:
        if word in DIGIT_WORDS:
            current += DIGIT_WORDS[word]
        elif word in UNITS:
            current = max(current, 1) * UNITS[word]
            total += current
            current = 0
        else:
            return None
    value = total + current
    return value if value > 1 or "रुपैयाँ" in phrase else None


def quantities(text: str) -> list[tuple[int, str]]:
    """Return canonical quantities; bare article `एक` and ordinals are ignored."""
    found: list[tuple[int, str]] = []
    for match in _DIGITS.finditer(text or ""):
        found.append((_digit_value(match.group()), match.group()))
    for match in _NUMBER_WORDS.finditer(text or ""):
        value = _words_value(match.group())
        if value is not None:
            found.append((value, match.group().strip()))
    return found


def check_numeric_grounding(reply: str, context: str,
                            tool_results: list[dict]) -> tuple[bool, list[str]]:
    """Require every quantity in a reply to occur in retrieved evidence."""
    evidence = context + " " + " ".join(str(result) for result in tool_results)
    available = {value for value, _ in quantities(evidence)}
    failures = []
    for value, raw in quantities(reply):
        if value not in available:
            failures.append(f"ungrounded quantity {raw!r} -> {value}")
    account_pattern = re.compile(
        r"(?:खाता|account)\s*(?:नम्बर|number|no\.?|#)?\s*([०-९0-9]+)"
        r"[^।\n]{0,40}?(?:(?:रकम|balance|मौज्दात)[^०-९0-9]{0,12})?"
        r"([०-९0-9,]+)\s*(?:रुपैयाँ|rupees)?", re.IGNORECASE)
    context_pairs = {(_digit_value(account), _digit_value(amount))
                     for account, amount in account_pattern.findall(evidence)}
    for account, amount in account_pattern.findall(reply or ""):
        pair = (_digit_value(account), _digit_value(amount))
        if context_pairs and pair not in context_pairs:
            failures.append(f"quantity attached to ungrounded account {account}")
    return not failures, failures


async def release_grounded_reply(reply: str, context: str, tool_results: list[dict], regenerate, fallback: str) -> str:
    """Allow a reply through after one repair attempt, otherwise use fallback."""
    ok, _ = check_numeric_grounding(reply, context, tool_results)
    if ok:
        return reply
    retry = await regenerate()
    ok, _ = check_numeric_grounding(retry, context, tool_results)
    return retry if ok else fallback
