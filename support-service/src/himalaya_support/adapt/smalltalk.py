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

import re

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
        "brilliant", "thumbs", "wow", "helpful",
        "राम्रो", "उत्तम", "बढिया", "सुन्दर", "ramro", "badhiya",
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

_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_MAX_WORDS = 6  # a courtesy is short; anything longer is a real message


def classify(text: str) -> str | None:
    """Return a small-talk category, or None when this is a real message."""
    raw = (text or "").strip()
    if not raw:
        return None
    words = [w.lower() for w in _WORD.findall(raw)]
    content = [w for w in words if w not in _FILLER]
    if not content or len(content) > _MAX_WORDS:
        return None

    # Order resolves overlap: "no thanks" is a decline, a bare "thanks" is
    # thanks, and "good luck" is a well-wish rather than a greeting.
    for category in (DECLINE, AFFIRM, THANKS, WELL_WISH, GREETING, APOLOGY,
                     ATTENTION, FAREWELL, PRAISE):
        core = _CORE[category]
        allowed = core | _ALLOWED[category]
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
