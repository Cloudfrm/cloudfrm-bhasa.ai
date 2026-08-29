from __future__ import annotations

import json
from pathlib import Path


def load_product_articles(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    articles = payload.get("articles") if isinstance(payload, dict) else payload
    docs: list[dict] = []
    for article in articles or []:
        docs.append(
            {
                "id": article.get("id", article.get("title", "")),
                "title": article.get("title", ""),
                "text": article.get("body", article.get("text", "")),
                "source": "product_knowledge",
                "tags": article.get("tags", []),
                "language": article.get("language", "en"),
            }
        )
    jsonl = path.parent / "banking_government_ne.jsonl"
    if jsonl.exists():
        docs.extend(load_banking_jsonl(jsonl))
    return docs


def load_banking_jsonl(path: Path) -> list[dict]:
    docs: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(
                {
                    "id": row.get("id", ""),
                    "title": row.get("title", ""),
                    "text": row.get("text") or row.get("body") or "",
                    "source": row.get("source", "banking_kb"),
                    "tags": [row.get("domain", "banking")],
                    "language": "ne",
                }
            )
    return docs
