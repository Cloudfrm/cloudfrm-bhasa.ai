from __future__ import annotations

import json
from pathlib import Path

# Himalaya honorific bench levels: त (intimate) / तिमी (peer) / तपाईं (respect).
# Customer support is a hierarchical public-service register → तपाईं.
SUPPORT_REGISTER = {
    "tapai": (
        "Nepali honorific register (from himalaya-ai/nepali-honorific-bench): "
        "always address the customer as तपाईं. Use respectful verb forms "
        "(-नुहुन्छ, -गर्नुहोस्, -दिनुहोस्, -भएको छ). "
        "Never use त, तँ, तैंले, or casual तिमी with a customer."
    ),
    "timi": (
        "Nepali honorific register: address the customer as तिमी "
        "(peer / informal app tone). Avoid त / तँ."
    ),
    "ta": (
        "Nepali honorific register: intimate त. Do not use this for support "
        "unless the product explicitly wants a close-friend voice."
    ),
}

INFORMAL_MARKERS = (" त ", "तँ", "तैं", "तिमी", "तिम्रो", "तिम्रा")
FORMAL_MARKERS = ("तपाईं", "तपाईँ", "हजुर")


def load_honorific_examples(path: Path, limit: int = 6) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= 80:
                break
    # Prefer hierarchical / professional contexts — those match support.
    ranked = [
        row
        for row in rows
        if "hierarchical" in str(row.get("context_english", "")).lower()
        or "principal" in str(row.get("context_english", "")).lower()
        or "priest" in str(row.get("context_english", "")).lower()
    ]
    chosen = (ranked or rows)[:limit]
    examples = []
    for row in chosen:
        examples.append(
            {
                "source": "himalaya-ai/nepali-honorific-bench",
                "title": "Honorific alignment",
                "text": (
                    f"Context: {row.get('context_english', '')}\n"
                    f"तपाईं form: {row.get('option_a') or row.get('option_c')}"
                ),
            }
        )
    return examples


def uses_informal_register(text: str) -> bool:
    if not any("\u0900" <= ch <= "\u097F" for ch in text):
        return False
    if any(marker in text for marker in FORMAL_MARKERS):
        return False
    return any(marker in text for marker in INFORMAL_MARKERS)
