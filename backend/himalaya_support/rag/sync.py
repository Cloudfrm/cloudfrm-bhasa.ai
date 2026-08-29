from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from himalaya_support.config import get_settings

# Pull compact, support-relevant slices — not the full 1.67M SFT dump.
SOURCES = [
    {
        "repo": "himalaya-ai/nepali-honorific-bench",
        "repo_type": "dataset",
        "filename": "nepali_honorific_alignment_devanagari.jsonl",
        "dest": "honorific.jsonl",
        "transform": "honorific",
    },
    {
        "repo": "himalaya-ai/nepali-hermes-function-calling-v1",
        "repo_type": "dataset",
        "filename": "data/train_0.jsonl",
        "dest": "function_calling_examples.jsonl",
        "transform": "tools",
        "limit": 40,
    },
    {
        "repo": "himalaya-ai/nepali-json-mode-singleturn",
        "repo_type": "dataset",
        "filename": "data/train_0.jsonl",
        "dest": "json_mode_examples.jsonl",
        "transform": "json_mode",
        "limit": 40,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Himalaya AI dataset slices for support RAG.")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    settings = get_settings()
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    token = settings.hf_token or None
    for source in SOURCES:
        print(f"Fetching {source['repo']} / {source['filename']}")
        local = hf_hub_download(
            repo_id=source["repo"],
            filename=source["filename"],
            repo_type=source["repo_type"],
            token=token,
        )
        dest = settings.corpus_dir / source["dest"]
        _write_slice(Path(local), dest, source, args.limit)
        print(f"  wrote {dest}")


def _write_slice(src: Path, dest: Path, source: dict, default_limit: int) -> None:
    limit = int(source.get("limit", default_limit))
    kind = source["transform"]
    written = 0
    with src.open(encoding="utf-8") as incoming, dest.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            if written >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = _normalize(row, kind, source["repo"])
            if not doc:
                continue
            outgoing.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1


def _normalize(row: dict, kind: str, repo: str) -> dict | None:
    if kind == "honorific":
        return {
            "id": row.get("id"),
            "source": repo,
            "title": "Nepali honorific alignment",
            "text": (
                f"{row.get('context_english', '')}\n"
                f"A: {row.get('option_a')} (rating {row.get('rating_a')})\n"
                f"B: {row.get('option_b')} (rating {row.get('rating_b')})\n"
                f"C: {row.get('option_c')} (rating {row.get('rating_c')})"
            ),
        }
    if kind == "tools":
        completion = row.get("enhanced_completion") or row.get("completion") or ""
        if "<tool_call>" not in completion:
            return None
        return {
            "id": row.get("id"),
            "source": repo,
            "title": row.get("task") or row.get("category") or "function calling",
            "text": f"Tools: {row.get('tools', '')}\nExample: {completion[:800]}",
        }
    if kind == "json_mode":
        schema = row.get("schema") or ""
        completion = row.get("enhanced_completion") or ""
        if not schema:
            return None
        return {
            "id": row.get("id"),
            "source": repo,
            "title": row.get("category") or "json mode",
            "text": f"Schema: {schema[:600]}\nExample JSON: {completion[:600]}",
        }
    return None


if __name__ == "__main__":
    main()
