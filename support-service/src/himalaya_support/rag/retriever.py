from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from himalaya_support.config import Settings
from himalaya_support.rag.knowledge import load_product_articles
from himalaya_support.support.honorific import load_honorific_examples

# The old class was [\w\u0900-\u097F], and \u0900-\u097F spans U+0900\u2013U+097F \u2014 which includes the
# danda "\u0964" and double danda "\u0965". Every sentence-final word was therefore
# indexed with its punctuation attached ("\u092C\u093F\u0930\u094D\u0938\u0928\u0941\u092D\u092F\u094B\u0964"), so it never matched
# the same word written mid-sentence. Devanagari digits (U+0966\u2013U+096F) stay.
TOKEN_RE = re.compile(
    r"[\w\u0900-\u0963\u0966-\u096F\u0971-\u097F]+",
    re.UNICODE,
)

# Nepali inflects heavily, and exact-token BM25 misses the match that matters:
# a member types "\u092A\u093F\u0928 \u092C\u093F\u0930\u094D\u0938\u0947\u0902" while the reset article says "\u092A\u093F\u0928 \u092C\u093F\u0930\u094D\u0938\u0928\u0941\u092D\u092F\u094B",
# so the right article loses to a shorter one that merely repeats "\u092A\u093F\u0928".
# Stripping a small set of common endings puts both on the same stem.
# Ordered longest-first; only applied when a real stem remains.
_NE_SUFFIXES = (
    "\u0928\u0941\u0939\u0941\u0928\u094D\u091B", "\u0928\u0941\u092A\u0930\u094D\u091B", "\u0928\u0941\u092D\u092F\u094B", "\u0928\u0941\u0939\u094B\u0938\u094D", "\u0928\u0941\u092A\u0930\u094D\u0928\u0947",
    "\u0939\u0930\u0942\u0932\u093E\u0908", "\u0939\u0930\u0942\u0915\u094B", "\u0939\u0930\u0942\u092E\u093E", "\u0939\u0930\u0942\u0932\u0947", "\u0939\u0930\u0942",
    "\u093F\u090F\u0915\u094B", "\u090F\u0915\u094B", "\u0947\u0915\u094B", "\u0947\u0915\u093E", "\u0947\u0915\u0940", "\u093F\u0928\u094D\u091B", "\u0928\u094D\u091B",
    "\u0932\u093E\u0908", "\u092C\u093E\u091F", "\u0938\u0901\u0917", "\u092E\u093E", "\u0932\u0947", "\u0915\u094B", "\u0915\u093E", "\u0915\u0940",
    "\u0948\u0902", "\u0947\u0902", "\u094B", "\u0947", "\u093E", "\u0940", "\u0942", "\u0941", "\u0901",
)
_MIN_STEM = 3


def _stem(token: str) -> str:
    if not any("\u0900" <= ch <= "\u097F" for ch in token):
        return token
    for suffix in _NE_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(tok.lower()) for tok in TOKEN_RE.findall(text or "")]


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    tokens: list[str]


class Retriever:
    """Lexical BM25 over product knowledge + Himalaya dataset slices.

    Retrieval is only for grounding. The chat model must generate a new answer.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.documents: list[Document] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 1.0
        self.reload()

    def reload(self) -> None:
        docs: list[Document] = []
        for article in load_product_articles(self.settings.knowledge_path):
            docs.append(self._to_doc(article["id"], article["title"], article["text"], article["source"]))
        corpus_dir = self.settings.corpus_dir
        if corpus_dir.exists():
            for path in sorted(corpus_dir.glob("*.jsonl")):
                docs.extend(self._load_jsonl(path))
        raw_honorific = self.settings.raw_dir / "nepali_honorific_alignment_devanagari.jsonl"
        for item in load_honorific_examples(raw_honorific):
            docs.append(self._to_doc(item["title"], item["title"], item["text"], item["source"]))
        self.documents = [doc for doc in docs if doc.tokens]
        self._index()

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query.strip() or not self.documents:
            return []
        q_tokens = tokenize(query)
        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            score = self._bm25(q_tokens, doc)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, doc in scored[:k]:
            results.append(
                {
                    "id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text[:1200],
                    "source": doc.source,
                    "score": round(score, 4),
                }
            )
        return results

    def _index(self) -> None:
        self._df = Counter()
        total_len = 0
        for doc in self.documents:
            total_len += len(doc.tokens)
            for term in set(doc.tokens):
                self._df[term] += 1
        self._avgdl = (total_len / len(self.documents)) if self.documents else 1.0

    def _bm25(self, query_tokens: list[str], doc: Document, k1: float = 1.5, b: float = 0.75) -> float:
        tf = Counter(doc.tokens)
        n = len(self.documents)
        score = 0.0
        dl = len(doc.tokens) or 1
        for term in query_tokens:
            if term not in tf:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / self._avgdl)
            score += idf * (tf[term] * (k1 + 1)) / denom
        return score

    def _load_jsonl(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text") or row.get("body") or row.get("clean") or ""
                if not text and row.get("instruction"):
                    text = f"{row.get('instruction', '')}\n{row.get('output', row.get('response', ''))}"
                if not text:
                    continue
                docs.append(
                    self._to_doc(
                        str(row.get("id", f"{path.stem}-{index}")),
                        str(row.get("title", path.stem)),
                        str(text),
                        str(row.get("source", f"himalaya-ai/{path.stem}")),
                    )
                )
        return docs

    @staticmethod
    def _to_doc(doc_id: str, title: str, text: str, source: str) -> Document:
        # The title says what an article is about, so a query matching it is a
        # stronger signal than the same words buried in the body. Without this
        # a "cannot log in" question matched the daily-limit article, which
        # merely repeats "मोबाइल बैंकिङ", ahead of the login article whose
        # title actually carries "लगइन".
        blob = f"{title}\n{title}\n{text}"
        return Document(doc_id=doc_id, title=title, text=text, source=source, tokens=tokenize(blob))
