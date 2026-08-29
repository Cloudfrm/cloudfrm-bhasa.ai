"""Crisis detection for member messages.

A member typed "I want to die." and the desk replied with the banking menu.
Another wrote that they had no reason to live and their loan was too much, and
the desk answered with repayment terms and late-payment penalties.

This runs on the raw message, before transliteration and before retrieval,
because transliteration mangles exactly the phrases that matter: "I am going
to kill myself" became "म हुँ गोइङ किल्ल ंय्सेल्फ". Matching what was typed is
the whole point.

Wording and helpline numbers live in data/knowledge/crisis_resources.json so
they can be corrected without a deploy. Nothing here invents a phone number:
if no resource has been verified, the reply says a person will follow up and
names no number at all. A wrong helpline is worse than none.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Deliberately narrow. Each phrase states intent toward one's own life; none
# of them is ordinary banking language. "dead" and "kill" alone are absent on
# purpose — "my card is dead", "the app killed my session" must not match.
_PATTERNS = (
    # English
    r"\bwant(?:s|ed)?\s+to\s+die\b",
    r"\bwanna\s+die\b",
    r"\bkill\s+(?:my\s?self|myself)\b",
    r"\bkilling\s+my\s?self\b",
    r"\b(?:end|take)\s+(?:my|his|her)\s+(?:own\s+)?life\b",
    r"\bend\s+it\s+all\b",
    r"\bno\s+(?:reason|point)\s+(?:to|in)\s+(?:live|living)\b",
    r"\bnot\s+worth\s+living\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bsuicid(?:e|al)\b",
    r"\b(?:harm|hurt)\s+my\s?self\b",
    r"\bdon'?t\s+want\s+to\s+(?:live|be\s+alive)\b",
    # Devanagari Nepali
    r"मर्न\s*चाहन्छु",
    r"मर्न\s*मन\s*(?:छ|लाग्यो|लाग्छ)",
    r"मर्न\s*पाए",
    r"मरे\s*हुन्थ्यो",
    r"आत्म\s*हत्या",
    r"आत्महत्या",
    r"ज्यान\s*दिन",
    r"ज्यान\s*लिन",
    r"बाँच्न\s*(?:मन\s*छैन|सक्दिनँ|सक्दिन|चाहन्न)",
    r"बाँच्ने\s*इच्छा\s*छैन",
    r"आफूलाई\s*मार्न",
    r"जीवन\s*अन्त्य",
    # Romanized Nepali
    r"\bmarna\s*(?:man|chahanchu|chahanxu|manchu|paye)\b",
    r"\bmare\s*hunthyo\b",
    r"\baatma\s*hatya\b|\batmahatya\b",
    r"\bjyan\s*(?:dina|dine|lina)\b",
    r"\bbachna\s*(?:man\s*chaina|sakdina|sakdinna)\b",
)

_CRISIS_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE | re.UNICODE)

_CONFIG_NAME = "crisis_resources.json"

_DEFAULT_MESSAGE: dict[str, dict[str, str]] = {
    "ne": {
        "acknowledge": (
            "तपाईंले यो कुरा भन्नुभयो — यो सजिलो छैन, र तपाईं एक्लै हुनुहुन्न।"
        ),
        "reach_out": (
            "कृपया अहिले नै आफूले विश्वास गर्ने कोही — परिवार, साथी वा "
            "स्वास्थ्यकर्मीसँग कुरा गर्नुहोस्।"
        ),
        "resources_intro": "तुरुन्तै कुरा गर्न सकिने ठाउँ:",
        "follow_up": "हाम्रो टोलीका एक जना यहाँ तपाईंलाई सम्पर्क गर्नेछन्।",
    },
    "en": {
        "acknowledge": (
            "Thank you for telling me. That sounds very hard, and you are not alone."
        ),
        "reach_out": (
            "Please talk to someone you trust right now — family, a friend, "
            "or a health worker."
        ),
        "resources_intro": "People you can talk to right now:",
        "follow_up": "Someone from our team will get in touch with you.",
    },
}


def looks_like_crisis(text: str) -> bool:
    """True when the member's own words signal risk to their life."""
    return bool(_CRISIS_RE.search(text or ""))


def _config_path(knowledge_path: Path | None) -> Path | None:
    if knowledge_path is None:
        return None
    return Path(knowledge_path).parent / _CONFIG_NAME


def load_crisis_config(knowledge_path: Path | None = None) -> dict[str, Any]:
    """Read the editable config, falling back to safe built-in wording."""
    config: dict[str, Any] = {"resources": [], "message": _DEFAULT_MESSAGE}
    path = _config_path(knowledge_path)
    if path and path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return config
        # Only resources someone has actually confirmed are ever shown.
        if loaded.get("resources_verified") is True:
            config["resources"] = [
                row for row in (loaded.get("resources") or [])
                if row.get("name") and row.get("contact")
            ]
        for lang, fields in (loaded.get("message") or {}).items():
            if lang in config["message"] and isinstance(fields, dict):
                config["message"][lang] = {**config["message"][lang], **fields}
    return config


def crisis_reply(language: str, config: dict[str, Any] | None = None) -> str:
    """A short, warm reply. Never contains financial content."""
    config = config or {"resources": [], "message": _DEFAULT_MESSAGE}
    lang = "ne" if language == "ne" else "en"
    words = (config.get("message") or _DEFAULT_MESSAGE).get(lang, _DEFAULT_MESSAGE[lang])

    parts = [words["acknowledge"], words["reach_out"]]
    resources = config.get("resources") or []
    if resources:
        lines = [words["resources_intro"]]
        for row in resources:
            hours = f" ({row['hours']})" if row.get("hours") else ""
            lines.append(f"• {row['name']}: {row['contact']}{hours}")
        parts.append("\n".join(lines))
    # No verified resource means no number is named. Silence beats a wrong one.
    parts.append(words["follow_up"])
    return "\n\n".join(parts)
