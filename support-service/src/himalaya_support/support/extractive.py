"""Extractive answer path — the same architecture as the deployed product.

A reply is EITHER a passage copied verbatim from a retrieved document OR one
of the two fetched refusal strings. No generative model runs here. Quoted
evidence is never translated: if the corpus cannot answer in the asked
language, the reply is a refusal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from himalaya_support.adapt.devanagari import normalize
from himalaya_support.rag.retriever import Retriever, tokenize
from himalaya_support.support.refusals import RefusalStrings, nfc

_STOP_NE = {
    "कति", "हो", "के", "छ", "छैन", "कसरी", "गर्ने", "गर्न", "गर्नुपर्छ", "हुन्छ", "कि", "र", "वा",
    "को", "का", "की", "मा", "ले", "बाट", "लाई", "पनि", "त", "नि", "म", "मेरो", "मलाई", "तपाईं",
    "हजुर", "यो", "त्यो", "कुन", "कहिले", "कहाँ", "किन", "सकिन्छ", "पाइन्छ", "लाग्छ", "भएमा",
    "प्रश्न", "जवाफ", "हुन", "गरेपछि", "पछि", "अघि", "सम्म", "देखि", "भन्दा", "जस्तो",
}
_STOP_EN = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "how", "what", "when", "where",
    "which", "why", "who", "can", "could", "i", "my", "me", "you", "your", "to", "of", "in", "on",
    "for", "and", "or", "it", "this", "that", "please", "there", "be", "has", "have", "not", "with",
    "much", "many", "long", "at", "by", "from", "if", "will", "should", "would", "about",
}
_NE_SUFFIXES = ("हरूको", "हरूमा", "हरूले", "हरू", "बाट", "सम्म", "देखि", "लाई", "को", "का", "की", "मा", "ले", "ल")
# A question "asks about a quantity" when it interrogates a figure (कति / how
# much) or names something that IS a figure (a rate, fee, limit, fine, term).
# Bare रकम ("money") is NOT a quantity question — that was the refusal-bait
# class behind E6.
_QUANTITY_Q = re.compile(
    r"कति|कतिको|ब्याजदर|ब्याज दर|शुल्क|सीमा|जरिवाना|म्याद|"
    r"how (?:much|many|long)|\brate\b|\bfees?\b|\bcharges?\b|\blimit\b|percent|minimum|maximum",
    re.IGNORECASE,
)
_QUANTITY_IN = re.compile(
    r"[०-९0-9]|प्रतिशत|रुपैयाँ|हजार|लाख|करोड|सय|एक|दुई|तीन|चार|पाँच|छ\b|सात|आठ|नौ|दस|बीस|तीस|पचास|"
    r"percent|%|rs\.?|npr|\bone\b|\btwo\b|\bthree\b|\bfive\b|\bten\b|hundred|thousand"
)
_ANSWER_SPLIT = re.compile(r"(?:^|\n)\s*जवाफ\s*[:：]\s*")


@dataclass
class Passage:
    doc_id: str
    title: str
    source: str
    language: str
    text: str  # verbatim substring of the document
    document: str
    score: float
    coverage: float

    def public(self) -> dict[str, Any]:
        return {
            "id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "language": self.language,
            "text": self.text,
            "score": round(self.score, 3),
            "coverage": round(self.coverage, 3),
        }


@dataclass
class Answer:
    kind: str  # "answer" | "refusal"
    reply: str
    question_language: str
    refusal_type: str | None = None
    passage: Passage | None = None
    considered: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reply": self.reply,
            "question_language": self.question_language,
            "refusal_type": self.refusal_type,
            "passage": self.passage.public() if self.passage else None,
            "considered": self.considered,
        }


def _stem(token: str) -> str:
    if re.search(r"[ऀ-ॿ]", token):
        for suffix in _NE_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                return token[: -len(suffix)]
        return token
    return token[:-1] if token.endswith("s") and len(token) > 4 else token


def _same_word(a: str, b: str) -> bool:
    """Exact stem match, or two Devanagari forms sharing a ≥4-char stem
    (बिर्सें / बिर्सनुभयो). Latin words must match exactly."""
    if a == b:
        return True
    if not (re.search(r"[ऀ-ॿ]", a) and re.search(r"[ऀ-ॿ]", b)):
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long_.startswith(short)


def _matched(q_tokens: set[str], d_tokens: set[str]) -> int:
    return sum(1 for q in q_tokens if any(_same_word(q, d) for d in d_tokens))


def content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in tokenize(text):
        if tok in _STOP_NE or tok in _STOP_EN or len(tok) < 2:
            continue
        out.add(_stem(tok))
    return out


def is_quantity_question(text: str) -> bool:
    return bool(_QUANTITY_Q.search(text or ""))


def has_quantity(text: str) -> bool:
    return bool(_QUANTITY_IN.search(text or ""))


def verbatim_passage(document: str) -> str:
    """The quotable part of a document: the answer segment of a Q/A row, else the whole body."""
    parts = _ANSWER_SPLIT.split(document, maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    return document.strip()


class ExtractiveAnswerer:
    def __init__(self, retriever: Retriever, refusals: RefusalStrings | None) -> None:
        self.retriever = retriever
        self.refusals = refusals
        self.min_coverage = 0.5
        self.min_quantity_coverage = 0.6

    @property
    def ready(self) -> bool:
        return self.refusals is not None

    def refusal(self, question_language: str, quantity: bool, considered: list[dict[str, Any]] | None = None) -> Answer:
        assert self.refusals is not None
        text = self.refusals.quantity if quantity else self.refusals.general
        return Answer("refusal", nfc(text), question_language, "quantity" if quantity else "general", None, considered or [])

    def answer(self, question: str, question_language: str) -> Answer:
        if self.refusals is None:
            raise RuntimeError("refusal strings unavailable")
        query = normalize(question)
        wants_quantity = is_quantity_question(query)
        q_tokens = content_tokens(query)
        considered: list[dict[str, Any]] = []
        if not q_tokens:
            return self.refusal(question_language, wants_quantity)

        hits = self.retriever.search(query, k=8)
        best: Passage | None = None
        for hit in hits:
            doc = self.retriever.document(hit["id"])
            if doc is None:
                continue
            language = doc.language
            d_tokens = content_tokens(f"{doc.title}\n{doc.text}")
            matched = _matched(q_tokens, d_tokens)
            coverage = matched / len(q_tokens)
            if len(q_tokens) >= 2 and matched < 2:
                coverage = 0.0  # one shared word is never enough to quote a passage
            passage_text = verbatim_passage(doc.text)
            row = {
                "id": doc.doc_id,
                "language": language,
                "coverage": round(coverage, 3),
                "score": hit["score"],
                "reason": "",
            }
            if language != question_language:
                row["reason"] = "wrong_language"
                considered.append(row)
                continue
            if coverage < self.min_coverage:
                row["reason"] = "low_coverage"
                considered.append(row)
                continue
            if wants_quantity and (coverage < self.min_quantity_coverage or not has_quantity(passage_text)):
                row["reason"] = "quantity_not_supported"
                considered.append(row)
                continue
            row["reason"] = "selected"
            considered.append(row)
            best = Passage(doc.doc_id, doc.title, doc.source, language, nfc(passage_text), doc.text, hit["score"], coverage)
            break
        if best is None:
            return self.refusal(question_language, wants_quantity, considered)
        assert best.text in nfc(best.document)  # verbatim guarantee
        return Answer("answer", best.text, question_language, None, best, considered)
