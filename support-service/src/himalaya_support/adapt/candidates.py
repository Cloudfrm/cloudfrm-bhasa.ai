"""Word-run transliteration with an IME-style candidate list.

Rules (E8/E9):
  * Decisions are made per Latin word-run, never per message.
  * Protected runs are never converted: known product/banking terms, any
    ALL-CAPS run, anything with digits, URLs, e-mails, Devanagari text.
  * A run is converted only when it parses as romanized Nepali (lexicon hit,
    stem+case-suffix hit, or strong romanized-Nepali markers). Plain English
    is left exactly as typed.
  * Candidates are ranked: exact lexicon match, then approved banking
    terminology, then phonetic variants by edit distance. Raw Latin is always
    the last option (the client appends it).
"""
from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass, field

from himalaya_support.adapt.translit import (
    ENGLISH_PASSTHROUGH,
    LEXICON,
    LOANWORDS,
    NEPALI_MARKERS,
    SUFFIX_DEVANAGARI,
    _assemble,
    _tokenize_word,
)

# Protected in ANY case: proper nouns, currency codes and acronyms with no
# everyday Nepali spelling. ATM / KYC / PIN / OTP are protected as typed in
# caps (the ALL-CAPS rule) — their lowercase forms are ordinary loanwords in
# romanized Nepali ("mero pin birse") and get पिन / ओटीपी / एटीएम / केवाईसी
# offered as candidates, with the raw Latin always available as the last option.
PROTECTED_TERMS = {
    "nimb", "scb", "usd", "npr", "swift", "emi", "ipo", "cib", "qr", "sms",
    "cvv", "ips", "connectips", "esewa", "khalti", "imepay", "fonepay", "nrb",
    "nepse", "vat", "nchl", "rtgs", "neft", "inr", "eur", "gbp", "aud",
}

# Approved banking terminology (romanized -> Devanagari). Ranked after exact
# lexicon hits and before phonetic guesses.
BANKING_TERMS: dict[str, str] = {
    "khata": "खाता", "byaj": "ब्याज", "byajdar": "ब्याजदर", "bhyajdar": "ब्याजदर",
    "rin": "ऋण", "karja": "कर्जा", "kista": "किस्ता", "rakam": "रकम",
    "bachat": "बचत", "chalti": "चल्ती", "maujdat": "मौज्दात", "sulka": "शुल्क",
    "shulka": "शुल्क", "shulk": "शुल्क", "sima": "सीमा", "seema": "सीमा",
    "chek": "चेक", "chekbook": "चेकबुक", "sakha": "शाखा", "shakha": "शाखा",
    "kagajat": "कागजात", "nagarikta": "नागरिकता", "rahadani": "राहदानी",
    "remittance": "रेमिट्यान्स", "remitance": "रेमिट्यान्स", "jarivana": "जरिवाना",
    "jariwana": "जरिवाना", "bhuktani": "भुक्तानी", "bhuktaani": "भुक्तानी",
    "sthanantaran": "स्थानान्तरण", "pasbook": "पासबुक", "passbook": "पासबुक",
    "sadasya": "सदस्य", "sahakari": "सहकारी", "bank": "बैंक", "banking": "बैंकिङ",
    "mobile": "मोबाइल", "login": "लगइन", "kewaisi": "केवाईसी", "kyc": "केवाईसी",
    "card": "कार्ड", "kard": "कार्ड", "debit": "डेबिट", "credit": "क्रेडिट",
    "statement": "स्टेटमेन्ट", "otp": "ओटीपी", "pin": "पिन", "paisa": "पैसा",
    "rupaiya": "रुपैयाँ", "rupees": "रुपैयाँ", "ruppee": "रुपैयाँ", "byaaj": "ब्याज",
    "nikasa": "निकासा", "jamma": "जम्मा", "jama": "जमा",
}

# Two dependent vowel signs in a row, or a vowel sign with no consonant before
# it: a broken glyph sequence (e.g. the web-lexicon form पर्याे).
_MALFORMED = re.compile("[ा-ौॉॊ][ा-ौॉॊ]|(?:^|[^क-हक़-य़़])[ा-ौॉॊ]")
_URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DEVA_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z']*")
# Split a message into runs: URLs/e-mails first so they stay whole.
_SEGMENT_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|[^\s@]+@[^\s@]+\.[^\s@]+)|[A-Za-z][A-Za-z']*|[०-९0-9]+|\s+|[^\sA-Za-z०-९0-9]+"
)

_ENGLISH_SHAPES = re.compile(
    r"(?:tion|sion|ing|ness|ment|ould|ight|ough|ance|ence|ible|able|ful|less|"
    r"^(?:the|and|for|with|that|this|from|have|has|are|was|were|will|what|when|where|which|"
    r"why|who|did|does|do|can|could|not|but|about|your|you|our|my|me|is|it|in|on|to|of|at|by|"
    r"if|or|so|no|yes|please|help|money|send|sent|arrive|arrived|did|not|forgot|lost|new|old|"
    r"open|close|closed|account|balance|transfer|loan|card|branch|time|day|days|week|month|"
    r"interest|rate|fee|limit|daily|apply|payment|pay|paid|due|late|cheque|check|book|deposit|"
    r"withdraw|savings|current|minimum|maximum|number|mobile|banking|online|app|sign|login)$)"
)
_NEPALI_SHAPES = re.compile(
    r"(?:chh|xa|x$|aa|ee|oo|au|ai|nu$|nus$|hos$|hosh$|cha$|chu$|chhu$|yo$|ko$|ma$|le$|lai$|"
    r"haru$|bata$|sanga$|dekhi$|samma$|tira$|sita$|nch|ncha|ndai|dai$|daina$|era$|eko$|eka$|"
    r"eki$|bha|dha|tha|gha|kha|jha|pha|nna|lla|tta|kka|ppa|mma|^(?:ke|ko|ka|ki|ku|ma|mero|timro|"
    r"hamro|tapai|hajur|kati|kasari|kaha|kahile|kina|kun|yo|tyo|ho|hola|huncha|hunchha|thiyo|"
    r"garnu|garne|garna|birse|birsey|birsiye|pathaye|pathaunu|khulcha|khulyo|milcha|sakincha|"
    r"aaudaina|aayena|aayo|pugena|pugyo|harayo|haraye|bhayo|bhaena|chaina|chhaina|xaina)$)"
)


# Reasons that are Nepali evidence on their own. Loanwords / banking terms are
# spelled like English ("mobile", "banking", "card") and are NOT evidence.
STRONG_NEPALI_REASONS = {"lexicon", "lexicon_stem_suffix", "nepali_marker", "nepali_shape"}


@dataclass
class RunDecision:
    text: str
    kind: str  # "latin" | "protected" | "devanagari" | "digits" | "space" | "punct" | "url" | "email"
    reason: str = ""
    parses: bool = False
    candidates: list[dict] = field(default_factory=list)
    default: str = ""  # what is sent if the user does not pick
    auto: bool = False  # convert without an explicit pick (Space/Enter)
    start: int = 0
    end: int = 0

    def public(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "reason": self.reason,
            "parses": self.parses,
            "auto": self.auto,
            "candidates": self.candidates,
            "default": self.default,
            "start": self.start,
            "end": self.end,
        }


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def protected_reason(run: str) -> str | None:
    """Why a run must never be converted, or None."""
    if not run:
        return "empty"
    if _DEVA_RE.search(run):
        return "devanagari"
    if _URL_RE.match(run):
        return "url"
    if _EMAIL_RE.match(run):
        return "email"
    if any(ch.isdigit() for ch in run):
        return "digits"
    if run.lower() in PROTECTED_TERMS:
        return "protected_term"
    letters = [c for c in run if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return "all_caps"
    return None


def _stem_suffix(lower: str) -> tuple[str, str] | None:
    for suffix in sorted(SUFFIX_DEVANAGARI, key=len, reverse=True):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            stem = lower[: -len(suffix)]
            if stem in LEXICON or stem in BANKING_TERMS:
                return stem, suffix
    return None


def parses_as_nepali(run: str) -> tuple[bool, str]:
    """Does this Latin run read as romanized Nepali?"""
    lower = run.lower().strip("'")
    if not lower:
        return False, "empty"
    if lower in BANKING_TERMS and lower not in ENGLISH_PASSTHROUGH:
        return True, "banking_term"
    if lower in LOANWORDS:
        return True, "loanword"
    if lower in ENGLISH_PASSTHROUGH:
        return False, "english_passthrough"
    if lower in NEPALI_MARKERS:
        return True, "nepali_marker"
    if lower in LEXICON:
        return True, "lexicon"
    if _stem_suffix(lower):
        return True, "lexicon_stem_suffix"
    if _ENGLISH_SHAPES.search(lower):
        return False, "english_shape"
    if _NEPALI_SHAPES.search(lower):
        return True, "nepali_shape"
    return False, "unknown_kept_raw"


# --- phonetic variants -----------------------------------------------------

_SWAPS: list[tuple[str, str]] = [
    ("aa", "a"), ("a", "aa"), ("ee", "i"), ("i", "ee"), ("oo", "u"), ("u", "oo"),
    ("t", "T"), ("d", "D"), ("n", "N"), ("s", "sh"), ("sh", "s"), ("b", "v"), ("v", "b"),
    ("ch", "chh"), ("chh", "ch"), ("ri", "ree"), ("w", "v"), ("ph", "f"),
]


def _roman_variants(word: str, limit: int = 24) -> list[str]:
    """Spellings one or two ambiguity-swaps away from `word`, nearest first."""
    seen = {word}
    out: list[str] = []
    level = [word]
    for _depth in range(2):
        nxt: list[str] = []
        for base in level:
            for old, new in _SWAPS:
                start = 0
                while True:
                    idx = base.find(old, start)
                    if idx < 0:
                        break
                    cand = base[:idx] + new + base[idx + len(old):]
                    start = idx + 1
                    if cand in seen:
                        continue
                    seen.add(cand)
                    nxt.append(cand)
                    out.append(cand)
                    if len(out) >= limit:
                        return out
        level = nxt
    return out


def _rules(word: str) -> str:
    return _nfc(_assemble(_tokenize_word(word)))


def _lexicon_form(lower: str) -> str | None:
    if lower in LEXICON:
        return LEXICON[lower]
    split = _stem_suffix(lower)
    if split:
        stem, suffix = split
        head = LEXICON.get(stem) or BANKING_TERMS.get(stem)
        if head:
            return head + SUFFIX_DEVANAGARI[suffix]
    return None


def word_candidates(run: str, limit: int = 5) -> list[dict]:
    """Ranked Devanagari candidates for one Latin run (raw Latin not included)."""
    lower = run.lower().strip("'")
    ranked: list[dict] = []
    seen: set[str] = set()

    def push(text: str, source: str, distance: int) -> None:
        text = _nfc(text)
        if not text or text in seen or _LATIN_RUN.fullmatch(text):
            return
        if _MALFORMED.search(text):  # matra on matra / matra on nothing: never offer it
            return
        seen.add(text)
        ranked.append({"text": text, "source": source, "distance": distance})

    lex = _lexicon_form(lower)
    if lex:
        push(lex, "lexicon", 0)
    if lower in BANKING_TERMS:
        push(BANKING_TERMS[lower], "banking", 0)
    split = _stem_suffix(lower)
    if split and split[0] in BANKING_TERMS:
        push(BANKING_TERMS[split[0]] + SUFFIX_DEVANAGARI[split[1]], "banking", 0)
    # Banking terms that are a near spelling of the run.
    for roman, deva in BANKING_TERMS.items():
        if roman != lower and abs(len(roman) - len(lower)) <= 1 and _edit_distance(roman, lower) == 1:
            push(deva, "banking", 1)
    push(_rules(run), "phonetic", 0)
    for depth, variant in enumerate(_roman_variants(lower)):
        if len(ranked) >= limit + 6:
            break
        v_lex = _lexicon_form(variant)
        if v_lex:
            push(v_lex, "lexicon", 1 + depth // 8)
        push(_rules(variant), "phonetic", 1 + depth // 8)
    order = {"lexicon": 0, "banking": 1, "phonetic": 2}
    ranked.sort(key=lambda item: (item["distance"] if item["source"] != "lexicon" else 0, order[item["source"]]))
    return ranked[:limit]


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def decide_run(run: str) -> RunDecision:
    reason = protected_reason(run)
    if reason:
        kind = {"url": "url", "email": "email", "devanagari": "devanagari", "digits": "digits"}.get(reason, "protected")
        return RunDecision(run, kind, reason, False, [], run)
    parses, why = parses_as_nepali(run)
    if not parses:
        return RunDecision(run, "latin", why, False, [], run)
    cands = word_candidates(run)
    default = cands[0]["text"] if cands else run
    return RunDecision(run, "latin", why, True, cands, default)


def segment(text: str) -> list[RunDecision]:
    """Split a message into runs and decide each one.

    `auto` (convert on Space/Enter without an explicit pick) is true for runs
    with strong Nepali evidence, and for English-spelled loanwords ONLY when
    the message already carries strong Nepali evidence or Devanagari — so
    "mero pin birse" converts `pin`, while "my mobile banking app" leaves
    `mobile` and `banking` exactly as typed (candidates are still offered).
    """
    out: list[RunDecision] = []
    for match in _SEGMENT_RE.finditer(_nfc(text or "")):
        piece = match.group()
        start, end = match.span()
        if piece.isspace():
            run = RunDecision(piece, "space", "", False, [], piece)
        elif _LATIN_RUN.fullmatch(piece):
            run = decide_run(piece)
        elif re.fullmatch(r"[०-९0-9]+", piece):
            run = RunDecision(piece, "digits", "digits", False, [], piece)
        elif _URL_RE.match(piece) or _EMAIL_RE.match(piece):
            run = RunDecision(piece, "url" if _URL_RE.match(piece) else "email", "protected", False, [], piece)
        elif _DEVA_RE.search(piece):
            run = RunDecision(piece, "devanagari", "devanagari", False, [], piece)
        else:
            run = RunDecision(piece, "punct", "", False, [], piece)
        run.start, run.end = start, end
        out.append(run)
    nepali_context = any(
        (r.kind == "latin" and r.parses and r.reason in STRONG_NEPALI_REASONS) or r.kind == "devanagari"
        for r in out
    )
    seen_word = False
    for r in out:
        if r.kind == "latin" and r.parses:
            r.auto = r.reason in STRONG_NEPALI_REASONS or nepali_context
            # Position-sensitive "ma": sentence-initial it is the pronoun म,
            # after a content word it is the locative मा (stage-4 rule).
            if r.text.lower() == "ma" and seen_word and r.candidates:
                r.candidates.sort(key=lambda c: 0 if c["text"] == "मा" else 1)
                r.default = r.candidates[0]["text"]
        if r.kind in {"latin", "devanagari", "protected", "digits"}:
            seen_word = True
    return out


def convert_message(text: str, choices: dict[str, str] | None = None) -> dict:
    """Default conversion of a whole message, run by run (used by /unicoder and tests)."""
    runs = segment(text)
    pieces: list[str] = []
    for run in runs:
        if run.kind == "latin" and run.parses:
            chosen = (choices or {}).get(run.text.lower())
            pieces.append(chosen or (run.default if run.auto else run.text))
        else:
            pieces.append(run.text)
    return {"text": _nfc("".join(pieces)), "runs": [r.public() for r in runs]}


def detect_question_language(text: str) -> str:
    """'ne' | 'en' decided after protected tokens are set aside.

    Only strong evidence counts: Devanagari, and Latin runs that read as
    romanized Nepali for a reason other than being an English-spelled loanword.
    """
    runs = segment(text)
    deva = 0
    latin_en = 0
    latin_ne = 0
    for run in runs:
        if run.kind == "devanagari":
            deva += len(_DEVA_RE.findall(run.text))
        elif run.kind == "latin":
            if run.parses and run.reason in STRONG_NEPALI_REASONS:
                latin_ne += len(run.text)
            elif not run.parses:
                latin_en += len(run.text)
            # loanwords are neutral
        # protected/digits/url/email/punct/space are set aside
    if deva == 0 and latin_en == 0 and latin_ne == 0:
        return "ne"
    if deva + latin_ne >= latin_en:
        return "ne"
    return "en"
