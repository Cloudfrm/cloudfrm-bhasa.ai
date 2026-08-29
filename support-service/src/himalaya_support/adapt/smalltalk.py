"""Courtesy and gesture words, answered like a person would.

Twenty common courtesies all fell through to retrieval and came back as
"I understood: 'X'. I can walk through login, transfers…". Two were worse:
"ok" returned OTP validity rules and "yes" returned a loan interest rate —
and the fallback the member was answering had just said "or say yes and I
will open a ticket".

Matching runs on the member's original text, before transliteration, and only
when the courtesy is essentially the whole message. "Hello, when is my loan
due?" is a loan question with a greeting attached, and must stay one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

GREETING = "greeting"
THANKS = "thanks"
APOLOGY = "apology"
ATTENTION = "attention"
FAREWELL = "farewell"
PRAISE = "praise"
WELL_WISH = "well_wish"
AFFIRM = "affirm"
DECLINE = "decline"

# Each category has CORE words that carry its meaning, and ALLOWED words that
# may accompany them. A message matches when every content word is in either
# set AND at least one is core.
#
# Membership alone is not enough: "thanks" appears in "no thanks", so a bare
# "Thanks" was being read as a decline; and "you" belongs to several
# categories, so "Thank you" matched none of them.
_CORE: dict[str, set[str]] = {
    GREETING: {
        "hello", "hi", "hey", "namaste", "namaskar", "greetings",
        "morning", "afternoon", "evening",
        "नमस्ते", "नमस्कार", "अभिवादन", "प्रभात", "सञ्चै",
        "sanchai", "sancho",
    },
    THANKS: {
        "thanks", "thank", "thankyou", "thx", "ty", "grateful",
        "appreciated", "appreciate",
        "धन्यवाद", "धन्यबाद", "आभारी", "dhanyabad", "dhanyabaad",
    },
    APOLOGY: {
        "sorry", "apologies", "apologize", "apologise", "pardon",
        "माफ", "क्षमा", "माफी", "maaf", "kshama",
    },
    ATTENTION: {
        "excuse", "listening", "सुन्नुहोस्", "हेर्नुहोस्", "एक्सक्यूज", "sunnus",
    },
    FAREWELL: {
        "bye", "goodbye", "byebye", "farewell", "care", "cheers",
        "बिदा", "अलविदा", "भेटौंला", "शुभरात्री",
        "bhetaula", "bida",
    },
    PRAISE: {
        "done", "great", "nice", "excellent", "perfect", "awesome",
        "brilliant", "thumbs", "wow", "helpful", "bravo", "superb",
        "fantastic", "amazing",
        "राम्रो", "उत्तम", "बढिया", "सुन्दर", "ramro", "badhiya", "साबास",
    },
    WELL_WISH: {
        "luck", "wishes", "congratulations", "congrats",
        "शुभकामना", "बधाई", "शुभेच्छा", "subhakamana", "badhai",
    },
    AFFIRM: {
        "yes", "yeah", "yep", "yup", "ok", "okay", "okey", "sure",
        "alright", "correct",
        "हो", "हुन्छ", "ठिक", "ठीक", "हवस्", "अँ",
        "hunchha", "huncha", "thik", "thikcha", "hawas",
    },
    DECLINE: {
        "no", "nope", "nah", "not", "nothing",
        "होइन", "छैन", "पर्दैन",
        "hoina", "chaina", "pardaina",
    },
}

_ALLOWED: dict[str, set[str]] = {
    GREETING: {"good", "day", "there", "शुभ", "बिहान", "साँझ", "छ", "कस्तो", "ke", "cha", "kasto"},
    THANKS: {"you", "so", "much", "very", "lot", "many", "dherai", "धेरै", "u"},
    APOLOGY: {"my", "bad", "so", "very", "really", "गर्नुहोस्", "गर्नुहोला"},
    ATTENTION: {"me", "you", "there", "are", "hello", "हजुर"},
    FAREWELL: {"take", "see", "you", "later", "good", "night", "फेरि", "जान्छु", "pheri"},
    PRAISE: {"well", "good", "job", "up", "so", "very", "super", "that", "was", "work"},
    WELL_WISH: {"good", "best", "all", "wish", "of", "you", "to"},
    AFFIRM: {"right", "then", "fine", "sounds", "that"},
    DECLINE: {"thanks", "thank", "you", "now", "later", "for", "अहिले"},
}

# Words that carry no meaning of their own and should not block a match.
_FILLER = {"a", "the", "is", "it", "u", "and", "please", "plz", "hai", "ho", "na", "ni", "है"}

MEETING = "meeting"

# Some courtesies are phrases, not bags of words. "Have a nice day" and "Nice
# to meet you" both fell through: adding their words to a category would have
# made "nice" plus any stray word match it, and "good day" — a greeting —
# would have been answered as a well-wish. The whole phrase is the unit, so
# it is matched as one. Spacing and case are flexible; a trailing "!" or "."
# is stripped before matching.
_PHRASES: dict[str, str] = {
    "nice to meet you": MEETING,
    "nice to meet you too": MEETING,
    "pleased to meet you": MEETING,
    "pleasure to meet you": MEETING,
    "good to meet you": MEETING,
    "great to meet you": MEETING,
    "nice meeting you": MEETING,
    "भेटेर खुसी लाग्यो": MEETING,
    "तपाईंलाई भेटेर खुसी लाग्यो": MEETING,
    "chinera khusi lagyo": MEETING,
    "have a nice day": WELL_WISH,
    "have a good day": WELL_WISH,
    "have a great day": WELL_WISH,
    "have a lovely day": WELL_WISH,
    "have a nice evening": WELL_WISH,
    "have a good evening": WELL_WISH,
    "have a good night": WELL_WISH,
    "have a nice weekend": WELL_WISH,
    "have a good weekend": WELL_WISH,
    "you too have a nice day": WELL_WISH,
    # Reported after the phrase fix shipped. "Good day" is the pointed one:
    # the reasoning for matching phrases used it as the example of what
    # bag-of-words would get wrong, and then left it matching nothing at all.
    # A bare "good day" to a support desk is an opening, not a send-off.
    "good day": GREETING,
    "g day": GREETING,
    "lovely day": GREETING,
    "beautiful day": GREETING,
    "how do you do": GREETING,
    "how do you do?": GREETING,
    "how are you": GREETING,
    "how are you doing": GREETING,
    "hope you are well": GREETING,
    "bless you": WELL_WISH,
    "god bless you": WELL_WISH,
    "all the best": WELL_WISH,
    "best of luck": WELL_WISH,
    "safe travels": WELL_WISH,
    "safe journey": WELL_WISH,
    "get well soon": WELL_WISH,
    "शुभ यात्रा": WELL_WISH,
    "सन्चै हुनुहुन्छ": GREETING,
    "कस्तो छ": GREETING,
    "कस्तो हुनुहुन्छ": GREETING,
    "शुभ दिन": WELL_WISH,
    "शुभ दिनको कामना": WELL_WISH,
    "तपाईंको दिन राम्रो होस्": WELL_WISH,
    "दिन राम्रो होस्": WELL_WISH,
    "subha din": WELL_WISH,
    "ramro din": WELL_WISH,
}

_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_TRIM = re.compile(r"[\s।!?.,;:]+")
_MAX_WORDS = 6  # a courtesy is short; anything longer is a real message


def _phrase_key(text: str) -> str:
    """Lowercase, punctuation-stripped, single-spaced form for phrase lookup."""
    return " ".join(_TRIM.split((text or "").lower())).strip()


_CONFIG_NAME = "courtesy_phrases.json"
_ORDER = (DECLINE, AFFIRM, THANKS, WELL_WISH, MEETING, GREETING, APOLOGY,
          ATTENTION, FAREWELL, PRAISE)
_CATEGORIES = frozenset(_ORDER)
_loaded: dict[str, tuple[dict[str, str], dict[str, set[str]]]] = {}


def _load_extra(knowledge_path: Path | None) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Phrases and words a desk owner has added, on top of the built-ins.

    The courtesy list started as twenty items and the desk's own table is
    longer; reporting the gap one string at a time is nobody's good use of an
    afternoon. Entries here are plain text and a category name, so the list
    can be completed without a deploy. Unknown categories are ignored rather
    than crashing the desk.
    """
    if knowledge_path is None:
        return {}, {}
    path = Path(knowledge_path).parent / _CONFIG_NAME
    if not path.exists():
        return {}, {}
    key = str(path)
    cached = _loaded.get(key)
    if cached is not None:
        return cached
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, {}
    phrases = {
        _phrase_key(text): category
        for text, category in (loaded.get("phrases") or {}).items()
        if category in _CATEGORIES and str(text).strip()
    }
    words: dict[str, set[str]] = {}
    for word, category in (loaded.get("words") or {}).items():
        if category in _CATEGORIES and str(word).strip():
            words.setdefault(category, set()).add(str(word).strip().lower())
    _loaded[key] = (phrases, words)
    return phrases, words


def classify(text: str, knowledge_path: Path | None = None) -> str | None:
    """Return a small-talk category, or None when this is a real message."""
    raw = (text or "").strip()
    if not raw:
        return None

    extra_phrases, extra_words = _load_extra(knowledge_path)
    key = _phrase_key(raw)
    phrase = _PHRASES.get(key) or extra_phrases.get(key)
    if phrase:
        return phrase

    words = [w.lower() for w in _WORD.findall(raw)]
    content = [w for w in words if w not in _FILLER]
    if not content or len(content) > _MAX_WORDS:
        return None

    # Order resolves overlap: "no thanks" is a decline, a bare "thanks" is
    # thanks, and "good luck" is a well-wish rather than a greeting.
    for category in _ORDER:
        core = _CORE.get(category, set()) | extra_words.get(category, set())
        allowed = core | _ALLOWED.get(category, set())
        if all(word in allowed for word in content) and any(word in core for word in content):
            return category
    return None


_REPLIES: dict[str, dict[str, str]] = {
    GREETING: {
        "ne": "नमस्ते! म कसरी सहयोग गर्न सक्छु?",
        "en": "Namaste! How can I help you today?",
    },
    THANKS: {
        "ne": "स्वागत छ! अरू केही चाहिएमा भन्नुहोस्।",
        "en": "You're welcome. Tell me if you need anything else.",
    },
    APOLOGY: {
        "ne": "केही छैन, चिन्ता नलिनुहोस्।",
        "en": "No need to apologise at all.",
    },
    ATTENTION: {
        "ne": "म यहीँ छु, भन्नुहोस्।",
        "en": "I'm here — go ahead.",
    },
    FAREWELL: {
        "ne": "धन्यवाद! फेरि भेटौंला।",
        "en": "Thank you — take care.",
    },
    PRAISE: {
        "ne": "धन्यवाद! यस्तै सहयोग गर्न पाउँदा खुसी लाग्यो।",
        "en": "Thank you — glad that helped.",
    },
    WELL_WISH: {
        "ne": "धन्यवाद! तपाईंलाई पनि शुभकामना।",
        "en": "Thank you — the same to you.",
    },
    MEETING: {
        "ne": "भेटेर मलाई पनि खुसी लाग्यो! म कसरी सहयोग गर्न सक्छु?",
        "en": "Good to meet you too. How can I help?",
    },
    # Reached only when nothing was pending; a pending ticket offer is
    # answered before this layer runs.
    AFFIRM: {
        "ne": "हुन्छ। म कसरी सहयोग गरूँ भन्नुहोस्।",
        "en": "Of course — tell me what you need help with.",
    },
    DECLINE: {
        "ne": "हुन्छ। अरू केही चाहिएमा भन्नुहोस्।",
        "en": "No problem. I'm here if you need anything.",
    },
}


def reply_for(category: str, language: str) -> str:
    lang = "ne" if language == "ne" else "en"
    return _REPLIES[category][lang]
