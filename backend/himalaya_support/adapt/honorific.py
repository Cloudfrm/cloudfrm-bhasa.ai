"""Nepali address-level consistency checks."""
from __future__ import annotations

import re

LEVELS = {"low": 0, "mid": 1, "high": 2}
PRONOUNS = {
    "high": re.compile(r"तपाईं|हजुर"),
    "mid": re.compile(r"तिमी"),
    "low": re.compile(r"\bतँ\b"),
}
VERBS = {
    "high": re.compile(r"(?:गर्नुहोस्|जानुहोस्|भन्नुहोस्|हुनुहुन्छ)"),
    "mid": re.compile(r"(?:गर्छौ|जान्छौ|भन्छौ|छौ)"),
    "low": re.compile(r"(?:गर|जा|भन|छस्)(?:\s|[।?!,]|$)"),
}


def check_honorific(text: str, required: str = "high") -> dict:
    """Return detected levels, mixed status, and whether the reply is allowed."""
    if required not in LEVELS:
        raise ValueError(f"unknown honorific level: {required}")
    observed = {level for level, pattern in PRONOUNS.items() if pattern.search(text or "")}
    observed.update(level for level, pattern in VERBS.items() if pattern.search(text or ""))
    if not observed:
        return {"ok": True, "level": None, "levels": [], "mixed": False, "reason": "no address marker"}
    levels = sorted(observed, key=LEVELS.__getitem__)
    mixed = len(levels) > 1
    below_required = any(LEVELS[level] < LEVELS[required] for level in levels)
    reason = "mixed address levels" if mixed else ("below required level" if below_required else "ok")
    return {"ok": not mixed and not below_required, "level": levels[-1],
            "levels": levels, "mixed": mixed, "reason": reason}
