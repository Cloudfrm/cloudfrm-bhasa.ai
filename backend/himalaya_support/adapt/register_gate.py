"""
Bounded regeneration loop around the register classifier (§0.3).

The blueprint says translationese is "BLOCKED and automatically regenerated".
Regeneration needs a bound, or a model that is reliably producing translationese
loops forever while a caller waits on a voice call. Three attempts, escalating
the instruction each time, then a deterministic spoken-Nepali fallback that is
guaranteed to pass because it is a constant.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from himalaya_support.adapt import register as R

RETRY_HINTS = [
    "",
    ("अघिल्लो जवाफ अंग्रेजीबाट उल्था गरेजस्तो भयो। बोल्ने नेपालीमा फेरि भन्नुहोस् — "
     "'प्रदान गर्नु', 'प्राप्त गर्नु' नलेख्नुहोस्, हरेक वाक्यमा 'म' र 'तपाईं' नराख्नुहोस्।"),
    ("धेरै औपचारिक भयो। साथीसँग कुरा गरेजस्तो, छोटो र सिधा भन्नुहोस्। "
     "वाक्यको अन्त्यमा 'नि' वा 'है' राख्न सकिन्छ।"),
]

DETERMINISTIC_FALLBACK = "माफ गर्नुहोस्, अहिले राम्रोसँग भन्न सकिनँ। फेरि सोध्नुहोस् न।"


@dataclass
class GateAttempt:
    text: str
    register: str
    accepted: bool
    reason: str = ""


@dataclass
class GateResult:
    text: str
    accepted: bool
    attempts: list[GateAttempt] = field(default_factory=list)
    used_fallback: bool = False

    @property
    def tries(self) -> int:
        return len(self.attempts)


def enforce(generate: Callable[[str | None], str], max_attempts: int = 3) -> GateResult:
    """`generate(hint)` produces a candidate reply; hint is None on the first try."""
    attempts: list[GateAttempt] = []
    for i in range(max_attempts):
        hint = RETRY_HINTS[i] if i < len(RETRY_HINTS) and RETRY_HINTS[i] else None
        candidate = generate(hint)
        ok, why = R.check_reply(candidate)
        attempts.append(GateAttempt(candidate, R.classify(candidate)["register"], ok, "" if ok else why))
        if ok:
            return GateResult(text=candidate, accepted=True, attempts=attempts)
    return GateResult(text=DETERMINISTIC_FALLBACK, accepted=False,
                      attempts=attempts, used_fallback=True)
