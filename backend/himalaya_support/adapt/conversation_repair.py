"""Bounded recovery for empty or low-confidence speech recognition."""
from __future__ import annotations

FIRST = "माफ गर्नुहोस्, राम्रोसँग सुनिनँ।"
SECOND = "फेरि भन्नुहोस् न, चाहे keypad बाट नम्बर थिच्नुहोस्।"
ESCALATION = "माफ गर्नुहोस्, अहिले जोडिन सकिनँ। कर्मचारीसँग कुरा गर्नुहोस्।"


def repair_message(failures: int) -> str:
    if failures <= 0:
        return FIRST
    if failures == 1:
        return SECOND
    return ESCALATION


def should_escalate(failures: int) -> bool:
    return failures >= 2
