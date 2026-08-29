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

TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


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
        blob = f"{title}\n{text}"
        return Document(doc_id=doc_id, title=title, text=text, source=source, tokens=tokenize(blob))
