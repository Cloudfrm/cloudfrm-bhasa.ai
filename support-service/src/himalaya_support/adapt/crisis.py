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

The phrase list was written English-first, and it showed: every miss found in
review was a Nepali or an indirect form. Nepali gives a speaker many ways to
inflect one thought, so matching now works two ways — outright phrases, and a
living/dying stem near a marker of not wanting to, which collapses every
inflection of that stem to a single rule. Additions belong in
data/knowledge/crisis_patterns.json, which takes plain phrases rather than
regexes so a native speaker can own the list without a deploy and without
being able to break the matcher. That list has not yet been reviewed by a
native speaker or a clinician; the file records that it has not.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Two kinds of match, because one kind was not enough.
#
# DIRECT phrases state the thing outright. They are deliberately narrow: none
# of them is ordinary banking language. "dead" and "kill" alone are absent on
# purpose — "my card is dead", "the app killed my session" must not match.
#
# PAIRS exist because the first version matched phrases, and Nepali inflects.
# "मलाई बाँच्ने इच्छा छैन" was caught while "मलाई बाँच्नु मन छैन" was not — the
# same sentence, a different verbal noun and a different word for wanting. A
# pair matches a living/dying stem near a marker of not wanting to, so every
# inflection of the stem collapses to one rule.
#
# The asymmetry that governs the tuning: a false positive costs a member one
# gentle, wrong message. A false negative drops the message into retrieval,
# which is well tuned on financial content, so a message about debt and
# despair reliably matches the loan article and comes back with repayment
# penalties. Recall is worth more here than precision.
_DIRECT = (
    # English — outright statements
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
    # English — the indirect register, which is how a great many people
    # actually say it. None of these names dying at all.
    r"\bwish\s+(?:i\s+)?(?:was|were)\s+dead\b",
    r"\bwish\s+i\s+(?:could\s+)?(?:just\s+)?disappear\b",
    r"\bwant\s+(?:it|this|everything)\s+all\s+to\s+end\b",
    r"\bwant\s+(?:it|this|everything)\s+to\s+(?:just\s+)?end\b",
    r"\bcan(?:'?t|not)\s+go\s+on\s+(?:any\s?more|like\s+this)\b",
    r"\bcan(?:'?t|not)\s+(?:do|take)\s+(?:this|it)\s+any\s?more\b",
    r"\bno\s+longer\s+want\s+to\s+(?:live|be\s+here|be\s+alive)\b",
    r"\blife\s+(?:is|feels|has\s+become)\s+(?:meaningless|pointless|empty|worthless)\b",
    r"\blife\s+has\s+no\s+(?:meaning|point|purpose)\b",
    r"\bnothing\s+(?:left\s+)?to\s+live\s+for\b",
    r"\btired\s+of\s+(?:living|life|being\s+alive)\b",
    r"\bi\s+am\s+a\s+burden\b|\bi'?m\s+a\s+burden\b",
    r"\bburden\s+to\s+(?:my|every)\w*\b",
    r"\beveryone\s+(?:would\s+be|is)\s+better\s+off\s+without\s+me\b",
    # Devanagari Nepali — outright
    r"मर्न\s*चाहन्छु",
    r"मर्न\s*मन\s*(?:छ|लाग्यो|लाग्छ)",
    r"मर्न\s*पाए",
    r"मरे\s*हुन्थ्यो",
    r"आत्म\s*हत्या",
    r"आत्महत्या",
    r"ज्यान\s*दिन",
    r"ज्यान\s*लिन",
    r"आफूलाई\s*मार्न",
    r"जीवन\s*अन्त्य",
    # Devanagari Nepali — indirect. जीवन only ever appears here bound to a
    # word about meaning or ending, never as a bare anchor, because जीवन बीमा
    # is life insurance and a cooperative member asks about it constantly.
    r"(?:जीवन|जिन्दगी)\s*(?:नै\s*)?अर्थहीन",
    r"(?:जीवन|जिन्दगी)(?:को)?\s*(?:कुनै\s*)?अर्थ\s*छैन",
    r"बाँच्नु?(?:को)?\s*(?:कुनै\s*)?अर्थ\s*छैन",
    r"(?:सबै|सब)\s*(?:कुरा\s*)?(?:समाप्त|अन्त्य)\s*(?:गर्न|पार्न|गरौं)",
    r"(?:सबै|सब)\s*(?:कुरा\s*)?(?:सकियोस्|सक्कियोस्|सकिदिए)",
    r"(?:जीवन|जिन्दगी)\s*समाप्त",
    r"म\s*(?:सबैको\s*लागि\s*)?बोझ\s*(?:भएँ|भएको|हुँ)",
    # Romanized Nepali
    r"\bmarna\s*(?:man|chahanchu|chahanxu|manchu|paye)\b",
    r"\bmare\s*hunthyo\b",
    r"\baatma\s*hatya\b|\batmahatya\b",
    r"\bjyan\s*(?:dina|dine|lina)\b",
    r"\b(?:jiwan|jeevan|jindagi)\s*(?:nai\s*)?arthahin\b",
    r"\b(?:sabai|sab)\s*(?:kura\s*)?samapta\s*(?:garna|parna)\b",
)

# (stem, marker) — a match needs both inside _WINDOW characters of each other.
# The stems are the living/dying verbs stripped of their endings, so बाँच्न,
# बाँच्नु, बाँच्ने, बाँचिरहन and the common no-chandrabindu spelling बाच्न all
# hit the same rule.
_PAIRS = (
    # Nepali: a verb of living, near a marker of not wanting to
    (
        r"बाँच|बाच्|जिउन|जिउँ|जिउने",
        r"मन\s*छैन|मन\s*लाग्दैन|मन\s*लागेन|इच्छा\s*छैन|इच्छा\s*मर|"
        r"चाहन्न|चाहँदिन|चाहन्नँ|सक्दिन|सक्दिनँ|"
        r"कारण\s*छैन|अर्थ\s*छैन|अर्थहीन|मतलब\s*छैन|"
        r"थाकें|थाकिसकें|थाकें\s*अब|गाह्रो\s*भयो\s*अब",
    ),
    # Nepali: a verb of dying, near a marker of wanting to
    (
        r"मर्न|मर्ने|मर्नु|मरे|मर्दा",
        r"मन|चाहन्छु|चाहन्छ|लाग्यो|लाग्छ|हुन्थ्यो|पाए|इच्छा|खोज्",
    ),
    # English: alive/live/living near a marker of not wanting to. "life" is
    # deliberately absent — "I don't want to lose my life savings" is a
    # sentence a member really writes.
    (
        # "live" excludes the locative sense: "I do not want to live in
        # Kathmandu anymore, closing my account" is a message a member with a
        # migration story really sends, and it is not a crisis. "live with"
        # stays in — "I can't live with this" is the figurative one.
        r"\balive\b|\bliving\b|\blive\b(?!\s+(?:in|at|near|abroad|outside|overseas))",
        r"\bdon'?t\s+want\b|\bdo\s+not\s+want\b|\brather\s+not\b|"
        r"\bno\s+(?:reason|point|meaning|will)\b|\bmeaningless\b|\bpointless\b|"
        r"\bnot\s+worth\b|\btired\s+of\b|\bsick\s+of\b|\bno\s+longer\s+want\b|"
        r"\bgive\s+up\b|\bstop\s+wanting\b",
    ),
    # Romanized Nepali
    (
        r"\bbach(?:na|nu|ne)\b|\bbaach(?:na|nu|ne)\b|\bjiu(?:na|nu|ne)\b",
        r"man\s*chaina|mann\s*chaina|ichha\s*chaina|iccha\s*chaina|"
        r"karan\s*chaina|artha\s*chaina|sakdina|sakdinna|chahanna|chahandina",
    ),
)

# Characters, not words: Devanagari has no reliable word boundary in a regex
# and the phrases that matter are short. Wide enough for "बाँच्ने कुनै कारण
# छैन", narrow enough that two unrelated sentences do not pair up.
_WINDOW = 42

_PATTERNS_NAME = "crisis_patterns.json"
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


def _phrase_to_regex(phrase: str) -> str:
    """Compile a plain phrase from the config into a tolerant pattern.

    Entries in the config file are phrases, not regexes, so the person
    maintaining them does not need to be a programmer and cannot break
    detection with a stray bracket. Whitespace between words is made flexible
    and word boundaries are added only where the phrase starts or ends in a
    Latin letter; Devanagari has no \\b to speak of.
    """
    words = [re.escape(word) for word in str(phrase).split() if word]
    if not words:
        return ""
    body = r"\s*".join(words)
    if re.match(r"^[A-Za-z]", phrase.strip()):
        body = r"\b" + body
    if re.search(r"[A-Za-z]$", phrase.strip()):
        body = body + r"\b"
    return body


@dataclass(frozen=True)
class Lexicon:
    direct: re.Pattern[str]
    pairs: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...]


def _compile(extra_phrases: list[str], extra_pairs: list[dict]) -> Lexicon:
    phrases = list(_DIRECT) + [p for p in (_phrase_to_regex(x) for x in extra_phrases) if p]
    pairs = [
        (re.compile(anchor, re.IGNORECASE | re.UNICODE),
         re.compile(marker, re.IGNORECASE | re.UNICODE))
        for anchor, marker in _PAIRS
    ]
    for row in extra_pairs:
        anchor = _phrase_to_regex(row.get("stem") or row.get("anchor") or "")
        markers = [p for p in (_phrase_to_regex(m) for m in (row.get("markers") or [])) if p]
        if anchor and markers:
            pairs.append((
                re.compile(anchor, re.IGNORECASE | re.UNICODE),
                re.compile("|".join(markers), re.IGNORECASE | re.UNICODE),
            ))
    return Lexicon(
        direct=re.compile("|".join(phrases), re.IGNORECASE | re.UNICODE),
        pairs=tuple(pairs),
    )


_BUILTIN = _compile([], [])
_loaded: dict[str, Lexicon] = {}


def load_lexicon(knowledge_path: Path | None = None) -> Lexicon:
    """Built-in patterns, plus any a reviewer has added in the config file.

    The file can only widen the net, never narrow it. Widening is the safe
    direction for someone editing without a deploy: the cost of one wrong
    match is a gentle message a member did not need, and the cost of one
    missed match is the desk answering despair with penalty terms. Removing a
    pattern stays a code change, with a test.
    """
    path = _patterns_path(knowledge_path)
    if path is None or not path.exists():
        return _BUILTIN
    key = str(path)
    cached = _loaded.get(key)
    if cached is not None:
        return cached
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _BUILTIN
    try:
        lexicon = _compile(
            [str(x) for x in (loaded.get("phrases") or []) if str(x).strip()],
            [row for row in (loaded.get("pairs") or []) if isinstance(row, dict)],
        )
    except re.error:
        return _BUILTIN
    _loaded[key] = lexicon
    return lexicon


def looks_like_crisis(text: str, knowledge_path: Path | None = None) -> bool:
    """True when the member's own words signal risk to their life."""
    raw = text or ""
    if not raw.strip():
        return False
    lexicon = load_lexicon(knowledge_path)
    if lexicon.direct.search(raw):
        return True
    for anchor, marker in lexicon.pairs:
        for hit in anchor.finditer(raw):
            near = raw[max(0, hit.start() - _WINDOW): hit.end() + _WINDOW]
            if marker.search(near):
                return True
    return False


def _patterns_path(knowledge_path: Path | None) -> Path | None:
    if knowledge_path is None:
        return None
    return Path(knowledge_path).parent / _PATTERNS_NAME


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
