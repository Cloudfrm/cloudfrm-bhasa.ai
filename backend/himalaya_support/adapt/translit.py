"""Romanized Nepali -> Devanagari transliteration engine (TRANSLIT-1).

Four stages, run in this order, per word/message:

    1. classify_input   -- is this message Devanagari, English, romanized
                            Nepali, or a mix? English is never converted.
    2. lexicon           -- word lookup. Beats stage 3 because Nepali
                            romanization is unstandardised: banking
                            loanwords (`balance` -> ब्यालेन्स), position-
                            sensitive grammar words (`ma` = म or मा), and
                            Sanskrit-derived spellings (`namaskar` ->
                            नमस्कार, long vowel a literal reading would
                            miss) all need a table, not a rule.
    3. syllable rules     -- longest-match tokenizer + a two-pass
                            consonant/vowel assembler for words the
                            lexicon does not cover.
    4. post-processing    -- suffix agglutination across word-boundaries,
                            nasal assimilation, sentence-final danda.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

# ============================================================== stage 1

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")

# Function words / verb endings that are strong, near-unambiguous signals
# of romanized Nepali. Deliberately excludes anything that also reads as
# a plain English word (see LOANWORDS below for those).
NEPALI_MARKERS = {
    "chha", "cha", "xa", "chhaina", "chaina", "xaina",
    "garnu", "garna", "garne", "garcha", "garchu", "garnuhos", "garnuhous",
    "garnuhosh", "garnus", "huncha", "hunchha", "hunxa", "bhayo", "vayo",
    "hola", "parcha", "parxa", "ho", "haina", "hoina", "mero", "timro",
    "hamro", "kati", "kasari", "kaha", "kahile", "kina", "malai", "yo",
    "tyo", "hajur", "tapai", "tapaai", "sanga", "sakiyo", "sakchhu",
    "sakchu", "janchhu", "thiyo", "chu", "chhu", "hu", "hoon", "khai",
    "aayo", "aayena", "bhulyo", "galat", "baki", "baaki", "pheri", "jama",
    "chahiyo", "chahincha", "chahinchha", "aaudaina", "aaunchu", "aauchu",
    "sanchai", "sanchhai", "sanjai", "sancho", "namaste", "namaskar",
}

# English-spelled loanwords that are conventional Nepali vocabulary in a
# banking/telecom register (§ "loanwords do not transliterate phonetically").
# They live in the lookup table like any other word, but their surface
# form is indistinguishable from plain English, so they must NOT count as
# evidence that a *whole message* is romanized Nepali -- otherwise a pure
# English sentence containing "balance" or "account" would trip stage 1.
LOANWORDS: dict[str, str] = {
    "balance": "ब्यालेन्स", "account": "एकाउन्ट", "card": "कार्ड",
    "block": "ब्लक", "sim": "सिम", "mobile": "मोबाइल", "data": "डाटा",
    "pack": "प्याक", "statement": "स्टेटमेन्ट", "pin": "पिन", "otp": "ओटिपी",
    "branch": "ब्रान्च", "cheque": "चेक", "check": "चेक", "loan": "लोन",
    "number": "नम्बर", "app": "एप", "table": "टेबल", "email": "इमेल",
    "atm": "एटीएम", "phone": "फोन", "bank": "बैंक",
}

_CASE_SUFFIXES = ("bata", "sanga", "haru", "samma", "dekhi", "tira", "sita",
                  "ko", "lai", "ma", "le")
_SUFFIX_RE = re.compile(r"(?:" + "|".join(_CASE_SUFFIXES) + r")$", re.IGNORECASE)


def classify_input(text: str) -> str:
    """"devanagari" | "english" | "romanized_nepali" | "mixed"."""
    text = text or ""
    has_deva = bool(_DEVANAGARI_RE.search(text))
    latin_words = _LATIN_WORD_RE.findall(text)

    if has_deva:
        return "mixed" if latin_words else "devanagari"
    if not latin_words:
        return "english"

    marker_hits = sum(1 for w in latin_words if w.lower() in NEPALI_MARKERS)
    suffix_hits = sum(1 for w in latin_words
                       if len(w) > 4 and _SUFFIX_RE.search(w.lower()))
    lex_hits = sum(1 for w in latin_words
                    if w.lower() in LEXICON and w.lower() not in LOANWORDS)
    score = (marker_hits * 3.0 + suffix_hits * 1.5 + lex_hits * 2.0) / len(latin_words)
    return "romanized_nepali" if score >= 0.35 else "english"


# ============================================================== stage 2

# Hand-curated core lexicon. Values here are ground truth: where a literal
# phonetic reading of the roman spelling would differ (Sanskrit-derived
# long vowels, retroflex loanwords, idiomatic postpositions), the table
# wins and stage 3 never runs for that word.
LEXICON: dict[str, str] = {
    # pronouns / grammar words (note: bare "ma" is handled positionally,
    # see _attach_suffixes -- this entry is the sentence-initial "I" sense)
    "ma": "म", "timi": "तिमी", "hami": "हामी", "u": "ऊ", "yo": "यो", "tyo": "त्यो",
    "mero": "मेरो", "merai": "मेरै", "timro": "तिम्रो", "hamro": "हाम्रो",
    "tapai": "तपाईं", "tapaai": "तपाईं", "hajur": "हजुर", "malai": "मलाई",
    "timilai": "तिमीलाई", "hamilai": "हामीलाई",
    # copula / existential
    "cha": "छ", "chha": "छ", "xa": "छ", "chaina": "छैन", "chhaina": "छैन",
    "xaina": "छैन", "ho": "हो", "haina": "हैन", "hoina": "होइन",
    "thiyo": "थियो", "thiena": "थिएन", "hunchha": "हुन्छ", "huncha": "हुन्छ",
    "hunxa": "हुन्छ", "bhayo": "भयो", "vayo": "भयो", "hola": "होला",
    "parcha": "पर्छ", "parxa": "पर्छ", "chu": "छु", "chhu": "छु",
    "hu": "हुँ", "hoon": "हुँ",
    # question words
    "ke": "के", "ko": "को", "kina": "किन", "kahile": "कहिले", "kaha": "कहाँ",
    "kati": "कति", "kasari": "कसरी", "kasto": "कस्तो", "kun": "कुन",
    # verbs (common conjugations used in banking chat)
    "garnu": "गर्नु", "garna": "गर्न", "garne": "गर्ने", "garchu": "गर्छु",
    "garcha": "गर्छ", "garnuhos": "गर्नुहोस्", "garnuhous": "गर्नुहोस्",
    "garnuhosh": "गर्नुहोस्", "garnus": "गर्नुहोस्",
    "janchhu": "जान्छु", "jane": "जाने", "janu": "जानु",
    "aauchu": "आउँछु", "aaunchu": "आउँछु", "aayo": "आयो", "aayena": "आयेन",
    "aaudaina": "आउँदैन",
    "sakchhu": "सक्छु", "sakchu": "सक्छु", "sakcha": "सक्छ", "sakchha": "सक्छ",
    "sakiyo": "सकियो",
    "dinu": "दिनु", "dinus": "दिनुस्", "hernu": "हेर्नु", "herna": "हेर्न",
    "sodhnu": "सोध्नु", "bhannu": "भन्नु", "bhannus": "भन्नुस्", "bhanne": "भन्ने",
    "bhanidinu": "भनिदिनु", "pathaidinuhos": "पठाइदिनुहोस्",
    "pathaune": "पठाउने", "kholnu": "खोल्नु", "kholna": "खोल्न",
    "khulcha": "खुल्छ", "kinnu": "किन्नु", "nikalna": "निकाल्न",
    "milcha": "मिल्छ", "bhulyo": "भुल्यो", "basxu": "बस्छु", "baschu": "बस्छु",
    "chahincha": "चाहिन्छ", "chahinchha": "चाहिन्छ", "chahiyo": "चाहियो",
    "sodhchu": "सोध्छु",
    # common nouns / adjectives / adverbs
    "khata": "खाता", "khatama": "खातामा", "paisa": "पैसा", "rupees": "रुपैयाँ",
    "rupaiya": "रुपैयाँ", "sahayog": "सहयोग", "samasya": "समस्या",
    "kripaya": "कृपया", "jankari": "जानकारी", "seva": "सेवा", "grahak": "ग्राहक",
    "ramro": "राम्रो", "dherai": "धेरै", "thorai": "थोरै", "ali": "अलि",
    "sathi": "साथी", "manchhe": "मान्छे", "manche": "मान्छे", "ghar": "घर",
    "aaja": "आज", "aja": "आज", "bholi": "भोलि", "hijo": "हिजो",
    "samaya": "समय", "bihana": "बिहान", "banda": "बन्द", "band": "बन्द",
    "galat": "गलत", "baki": "बाँकी", "baaki": "बाँकी", "pheri": "फेरि",
    "jama": "जमा", "baje": "बजे", "rin": "ऋण", "sanga": "सँग",
    "sanchai": "सञ्चै", "sanchhai": "सञ्चै", "sanjai": "सञ्चै",
    "sancho": "सन्चो", "ani": "अनि",
    # greetings / discourse (Sanskrit-derived spellings, kept explicit
    # because the literal short-vowel reading of the roman string is wrong)
    "namaste": "नमस्ते", "namaskar": "नमस्कार", "namaskaar": "नमस्कार",
    "dhanyabad": "धन्यवाद", "dhanyawad": "धन्यवाद", "nepal": "नेपाल",
    "nepali": "नेपाली", "kathmandu": "काठमाडौं", "butwal": "बुटवल",
    "pokhara": "पोखरा", "swasthya": "स्वास्थ्य", "aama": "आमा",
    "ishwar": "ईश्वर", "ankit": "अंकित", "sambhav": "सम्भव",
    # conjuncts needing an irregular (long-vowel) reading -- the general
    # word "prashna"/"mishra" style clusters are handled fine by stage 3
    "gyan": "ज्ञान", "gyaan": "ज्ञान", "agyan": "अज्ञान",
    "kshama": "क्षमा", "kshamaa": "क्षमा",
    "pratham": "प्रथम", "prashna": "प्रश्न", "mishra": "मिश्र", "chakra": "चक्र",
    "shabda": "शब्द", "dherai": "धेरै", "thorai": "थोरै", "bhanne": "भन्ने",
    # standalone forms of the case markers, used when a suffix word is not
    # merged onto a preceding word (see SUFFIX_DEVANAGARI / _attach_suffixes)
    "lai": "लाई", "le": "ले", "bata": "बाट", "haru": "हरू", "samma": "सम्म",
    "dekhi": "देखि", "tira": "तिर", "sita": "सित",
}
LEXICON.update(LOANWORDS)

# Case markers, both as a standalone token following a content word
# ("ghar ma" -> घरमा) and as an in-word suffix stage 3 falls back on.
# "ma" is deliberately absent here: it is position-sensitive (§3.5 of the
# design note) and handled explicitly in _attach_suffixes.
SUFFIX_DEVANAGARI: dict[str, str] = {
    "ko": "को", "lai": "लाई", "le": "ले", "bata": "बाट", "sanga": "सँग",
    "haru": "हरू", "samma": "सम्म", "dekhi": "देखि", "tira": "तिर",
    "sita": "सित", "ma": "मा",
}

# Pure English words that legitimately appear inside an otherwise-Nepali
# sentence and must be left alone rather than mangled by stage 3.
ENGLISH_PASSTHROUGH = {
    "hello", "bye", "help", "login", "send", "receive", "error", "please",
    "thank", "thanks", "you", "how", "are", "is", "my", "the", "a", "an",
    "to", "of", "for", "and", "or", "not", "yes", "no", "ok", "exit",
    "menu", "back", "next", "cancel", "submit", "save", "active",
    "inactive", "enabled", "disabled", "inquiry", "entry", "fund",
    "verification", "plan", "transfer", "withdraw", "deposit", "payment",
    "amount", "date", "time", "code", "verify", "confirm", "password",
    "address", "customer", "support", "service", "team", "call",
    "message", "now", "today", "change", "activate", "limit", "pdf",
    "match", "working", "properly", "request", "received", "office",
    "opens", "closed", "morning", "hold", "moment", "try", "again",
    "good", "can", "i", "need", "with", "your", "has", "been", "our",
    "what", "want", "would", "like", "welcome",
}

RETROFLEX_KEYS = {"T", "D", "N", "Th", "Dh"}
LABIALS = {"b", "bh", "p", "ph", "m"}


# ============================================================== stage 3

CONSONANTS: dict[str, str] = {
    "ksh": "क्ष", "chh": "छ",
    "kh": "ख", "gh": "घ", "ng": "ङ", "ch": "च", "jh": "झ", "ny": "ञ",
    "Th": "ठ", "Dh": "ढ", "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ",
    "sh": "श", "Sh": "ष", "gy": "ज्ञ", "gn": "ज्ञ", "jn": "ज्ञ",
    "k": "क", "g": "ग", "c": "च", "j": "ज",
    "T": "ट", "D": "ड", "N": "ण", "t": "त", "d": "द", "n": "न",
    "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल",
    "v": "व", "w": "व", "s": "स", "h": "ह", "z": "ज़", "f": "फ",
    "x": "क्ष", "q": "क", "L": "ळ",
}
CONSONANT_KEYS = sorted(CONSONANTS, key=len, reverse=True)

VOWELS_INDEPENDENT: dict[str, str] = {
    "aa": "आ", "ai": "ऐ", "au": "औ", "ee": "ई", "ii": "ई", "oo": "ऊ", "uu": "ऊ",
    "a": "अ", "i": "इ", "u": "उ", "e": "ए", "o": "ओ",
}
VOWELS_MATRA: dict[str, str] = {
    "aa": "ा", "ai": "ै", "au": "ौ", "ee": "ी", "ii": "ी", "oo": "ू", "uu": "ू",
    "a": "", "i": "ि", "u": "ु", "e": "े", "o": "ो",
}
VOWEL_KEYS = sorted(VOWELS_MATRA, key=len, reverse=True)

_DIGITS = {str(d): c for d, c in enumerate("०१२३४५६७८९")}


def _match_at(word: str, i: int, keys: list[str]) -> str | None:
    """Longest-match at position i, exact case first then case-folded."""
    for key in keys:
        if word.startswith(key, i):
            return key
    lower = word.lower()
    for key in keys:
        if lower.startswith(key, i):
            return key
    return None


def _tokenize_word(word: str) -> list[tuple]:
    """Longest-match scan into ('C', deva, key) / ('V', key) / ('O', ch)."""
    tokens: list[tuple] = []
    i, n = 0, len(word)
    while i < n:
        ck = _match_at(word, i, CONSONANT_KEYS)
        vk = _match_at(word, i, VOWEL_KEYS)
        # Prefer whichever match is longer; consonants win length ties
        # (a vowel key never overlaps a consonant key's first letter).
        if ck and (not vk or len(ck) >= len(vk)):
            tokens.append(("C", CONSONANTS[ck], ck))
            i += len(ck)
        elif vk:
            tokens.append(("V", vk))
            i += len(vk)
        elif word[i].isdigit():
            tokens.append(("O", _DIGITS[word[i]]))
            i += 1
        else:
            tokens.append(("O", word[i]))
            i += 1
    return tokens


def _assemble(tokens: list[tuple]) -> str:
    """Two-pass: resolve each pending consonant once its follower is known."""
    out: list[str] = []
    pending: str | None = None
    for idx, tok in enumerate(tokens):
        if pending is not None:
            if tok[0] == "V":
                out.append(pending + VOWELS_MATRA[tok[1]])
            else:
                out.append(pending + "्")  # consonant cluster: halant
            pending = None
            if tok[0] != "V":
                pass  # fall through to process this (non-vowel) token fresh
            else:
                continue

        if tok[0] == "V":
            out.append(VOWELS_INDEPENDENT[tok[1]])
        elif tok[0] == "C":
            deva, key = tok[1], tok[2]
            if key in ("n", "m"):
                nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
                if nxt and nxt[0] == "C" and nxt[2] not in LABIALS and nxt[2] != key:
                    out.append("ं")
                    continue
            pending = deva
        else:
            out.append(tok[1])
    if pending is not None:
        out.append(pending)
    return "".join(out)


def _has_ambiguous_letters(word: str) -> bool:
    lower = word.lower()
    return bool(re.search(r"(?<![A-Za-z])[tdn](?![A-Za-z])|t|d|n|i|u", lower)) and (
        any(c in lower for c in "tdn") or any(v in lower for v in ("i", "u"))
    )


def _rule_alternative(word: str) -> str | None:
    """One plausible alternate reading: dental<->retroflex on t/d/n."""
    if not re.search(r"[tdnTDN]", word):
        return None
    swapped = []
    for ch in word:
        if ch == "t":
            swapped.append("T")
        elif ch == "d":
            swapped.append("D")
        elif ch == "n":
            swapped.append("N")
        elif ch in "TDN":
            swapped.append(ch.lower())
        else:
            swapped.append(ch)
    alt_word = "".join(swapped)
    if alt_word == word:
        return None
    return _assemble(_tokenize_word(alt_word))


def _rules_word(word: str) -> tuple[str, float, list[str]]:
    deva = _assemble(_tokenize_word(word))
    confidence = 0.65
    if re.search(r"[tdnTDN]", word):
        confidence -= 0.15
    if len(word) <= 3:
        confidence += 0.1
    confidence = max(0.3, min(confidence, 0.75))
    alt = _rule_alternative(word)
    alternatives = [alt] if alt and alt != deva else []
    return deva, round(confidence, 3), alternatives


# ============================================================== stage 4

_SENTENCE_END_RE = re.compile(r"[।.!?]$")


def _strip_known_suffix(lower: str) -> tuple[str, str] | None:
    """Split an in-word suffix ('gharma' -> 'ghar', suffix 'ma') if the
    stem is itself a lexicon word. Longest suffix first."""
    for suffix in sorted(_CASE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            stem = lower[: -len(suffix)]
            if stem in LEXICON:
                return stem, suffix
    return None


@dataclass
class WordResult:
    roman: str
    devanagari: str
    source: str
    confidence: float
    alternatives: list[str] = field(default_factory=list)


def _resolve_word(word: str, *, sentence_initial: bool) -> WordResult:
    lower = word.lower()

    if lower in LEXICON:
        return WordResult(word, LEXICON[lower], "lexicon", 0.95)

    split = _strip_known_suffix(lower)
    if split:
        stem, suffix = split
        return WordResult(word, LEXICON[stem] + SUFFIX_DEVANAGARI[suffix],
                          "lexicon", 0.85)

    if lower in ENGLISH_PASSTHROUGH:
        return WordResult(word, word, "passthrough", 1.0)

    deva, confidence, alternatives = _rules_word(word)
    return WordResult(word, deva, "rules", confidence, alternatives)


_TOKEN_RE = re.compile(r"[A-Za-z]+|\s+|.", re.DOTALL)


def _attach_suffixes(word_tokens: list[dict]) -> None:
    """Merge a standalone suffix token onto the immediately preceding word
    (stage 4 agglutination): 'ghar ma' and 'gharma' both read घरमा.

    'ma' is position-sensitive: sentence-initial it is the pronoun म (I),
    everywhere else that follows a content word it is the locative मा.
    """
    for i, entry in enumerate(word_tokens):
        lower = entry["word"].lower()
        if lower not in SUFFIX_DEVANAGARI:
            continue
        if lower == "ma" and entry["sentence_initial"]:
            continue  # pronoun "I", not a suffix
        if i == 0:
            continue  # no host word to attach to (e.g. lone leading "lai")
        prev = word_tokens[i - 1]
        if prev.get("consumed"):
            continue
        entry["consumed"] = True
        entry["merge_into"] = i - 1


def _apply_danda(rendered: list[str]) -> None:
    for i, piece in enumerate(rendered):
        if piece == "." and i > 0 and re.search(r"[ऀ-ॿ]$", rendered[i - 1]):
            rendered[i] = "।"


@dataclass
class TranslitResult:
    out: str
    confidence: float
    words: list[WordResult]


def to_devanagari(text: str, force: bool = False) -> TranslitResult:
    category = classify_input(text)
    if not text:
        return TranslitResult(out=text, confidence=0.0, words=[])
    if category == "devanagari":
        return TranslitResult(out=text, confidence=1.0, words=[])
    if category == "english" and not force:
        return TranslitResult(out=text, confidence=0.0, words=[])

    tokens = _TOKEN_RE.findall(text)

    # Build the list of word-token descriptors (index into `tokens`,
    # whether it starts a sentence/clause) for the suffix-merge pass.
    word_entries: list[dict] = []
    at_sentence_start = True
    for ti, tok in enumerate(tokens):
        if re.fullmatch(r"[A-Za-z]+", tok):
            word_entries.append({
                "token_index": ti, "word": tok,
                "sentence_initial": at_sentence_start,
                "consumed": False, "merge_into": None,
            })
            at_sentence_start = False
        elif _SENTENCE_END_RE.match(tok.strip()) if tok.strip() else False:
            at_sentence_start = True
        elif tok.strip():
            pass  # other punctuation does not reset sentence-initial state

    if category == "mixed":
        # Devanagari spans pass through untouched; only latin words convert.
        pass

    _attach_suffixes(word_entries)

    resolved: dict[int, WordResult] = {}
    for entry in word_entries:
        if entry["consumed"]:
            continue
        resolved[entry["token_index"]] = _resolve_word(
            entry["word"], sentence_initial=entry["sentence_initial"])

    # Fold merged suffixes into their host word's devanagari.
    for entry in word_entries:
        if not entry["consumed"]:
            continue
        host = word_entries[entry["merge_into"]]
        host_result = resolved[host["token_index"]]
        suffix_deva = SUFFIX_DEVANAGARI[entry["word"].lower()]
        resolved[host["token_index"]] = WordResult(
            roman=host_result.roman + " " + entry["word"],
            devanagari=host_result.devanagari + suffix_deva,
            source=host_result.source, confidence=min(host_result.confidence, 0.85),
            alternatives=host_result.alternatives,
        )

    rendered = list(tokens)
    consumed_indices = {e["token_index"] for e in word_entries if e["consumed"]}
    words_out: list[WordResult] = []
    for entry in word_entries:
        if entry["consumed"]:
            rendered[entry["token_index"]] = ""
            # drop one adjacent whitespace token so 'ghar ma' collapses to 'gharमा'
            gap = entry["token_index"] - 1
            if gap >= 0 and rendered[gap].strip() == "" and rendered[gap]:
                rendered[gap] = ""
            continue
        result = resolved[entry["token_index"]]
        rendered[entry["token_index"]] = result.devanagari
        words_out.append(result)

    _apply_danda(rendered)
    out = "".join(rendered)
    out = unicodedata.normalize("NFC", out)

    if words_out:
        confidence = round(sum(w.confidence for w in words_out) / len(words_out), 3)
    else:
        confidence = 0.0
    return TranslitResult(out=out, confidence=confidence, words=words_out)


# --------------------------------------------------------- bootstrap util

_REVERSE_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "c", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "ny", "ट": "T", "ठ": "Th", "ड": "D", "ढ": "Dh",
    "ण": "N", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "व": "v", "श": "sh", "ष": "Sh", "स": "s", "ह": "h", "ळ": "L", "ज़": "z",
}
_REVERSE_MATRA = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ृ": "ri",
}
_REVERSE_INDEPENDENT = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
}
_REVERSE_DIGITS = {c: str(d) for d, c in enumerate("०१२३४५६७८९")}


def devanagari_to_roman(text: str) -> str:
    """Deterministic Devanagari -> roman reverse mapping, used to bootstrap
    the lexicon from a corpus (generate the plausible romanizations of a
    known-correct Devanagari word, then invert). Not used at request time."""
    out: list[str] = []
    chars = list(text or "")
    i, n = 0, len(chars)
    while i < n:
        ch = chars[i]
        if ch in _REVERSE_CONSONANTS:
            out.append(_REVERSE_CONSONANTS[ch])
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == "्":
                i += 2
                continue
            if nxt in _REVERSE_MATRA:
                out.append(_REVERSE_MATRA[nxt])
                i += 2
                continue
            out.append("a")
            i += 1
            continue
        if ch in _REVERSE_INDEPENDENT:
            out.append(_REVERSE_INDEPENDENT[ch])
            i += 1
            continue
        if ch == "ं":
            out.append("n")
            i += 1
            continue
        if ch == "ँ":
            out.append("n")
            i += 1
            continue
        if ch == "ः":
            out.append("h")
            i += 1
            continue
        if ch in _REVERSE_DIGITS:
            out.append(_REVERSE_DIGITS[ch])
            i += 1
            continue
        if ch == "।":
            out.append(".")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ----------------------------------------------------- legacy-shim aliases
# `main.py` historically imported a (text, confidence) tuple. Keep call
# sites simple by exposing that shape too.

def to_devanagari_tuple(text: str) -> tuple[str, float]:
    result = to_devanagari(text)
    return result.out, result.confidence


def looks_romanized_nepali(text: str) -> float:
    category = classify_input(text)
    if category == "devanagari":
        return 0.0
    if category == "english":
        return 0.0
    result = to_devanagari(text)
    return result.confidence


def load_web_lexicon() -> int:
    """Merge roman→Devanagari pairs pulled from public datasets. Does not override core keys."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "roman_lexicon.json"
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    added = 0
    if not isinstance(payload, dict):
        return 0
    for roman, de in payload.items():
        key = str(roman).strip().lower()
        value = str(de).strip()
        if not key or not value or key in LEXICON:
            continue
        LEXICON[key] = value
        added += 1
    return added


load_web_lexicon()
