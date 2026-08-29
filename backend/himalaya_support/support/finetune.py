from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from himalaya_support.config import Settings
from himalaya_support.support.gold_seed import SYS_EN, SYS_NE, write_gold


class SFTRecorder:
    """Save teacher pairs so Himalaya Gemma can be LoRA-tuned later.

    This machine has no GPU, so we do not train here. We write ShareGPT-style
    messages that Unsloth / TRL can consume later.
    """

    def __init__(self, settings: Settings) -> None:
        self.dir = settings.finetune_dir
        self.path = self.dir / "sft_pairs.jsonl"
        self.gold_path = self.dir / "gold.jsonl"
        self.train_path = self.dir / "train.jsonl"
        self.eval_path = self.dir / "eval.jsonl"

    def record(
        self,
        user: str,
        assistant: str,
        *,
        language: str,
        intent: str,
        sources: list[str],
        teacher: str,
        gemma_draft: str | None = None,
    ) -> str:
        pair_id = uuid.uuid4().hex
        row: dict[str, Any] = {
            "id": pair_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "teacher": teacher,
            "language": language,
            "intent": intent,
            "sources": sources,
            "keep": None,
            "gemma_draft": gemma_draft,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return pair_id

    def rate(self, sft_id: str, keep: bool) -> bool:
        rows = self._read_log()
        found = False
        for row in rows:
            if row.get("id") == sft_id:
                row["keep"] = keep
                found = True
                break
        if not found:
            return False
        self._write_log(rows)
        return True

    def stats(self) -> dict[str, int]:
        gold = self._read_jsonl(self.gold_path) if self.gold_path.exists() else []
        log = self._read_log()
        kept = sum(1 for row in log if row.get("keep") is True)
        skipped = sum(1 for row in log if row.get("keep") is False)
        unlabeled = sum(1 for row in log if row.get("keep") is None)
        return {
            "gold": len(gold),
            "logged": len(log),
            "kept": kept,
            "skipped": skipped,
            "unlabeled": unlabeled,
            "train": self._count_lines(self.train_path),
            "eval": self._count_lines(self.eval_path),
        }

    def export(self) -> dict[str, int]:
        write_gold(self.gold_path)
        gold = self._read_jsonl(self.gold_path)
        kept = [self._as_train_row(row) for row in self._read_log() if row.get("keep") is True]
        train: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        for row in gold:
            target = eval_rows if row.get("split") == "eval" else train
            target.append(self._as_train_row(row))
        for index, row in enumerate(kept):
            if index % 8 == 0:
                eval_rows.append(row)
            else:
                train.append(row)
        self._write_jsonl(self.train_path, train)
        self._write_jsonl(self.eval_path, eval_rows)
        return {"gold": len(gold), "kept": len(kept), "train": len(train), "eval": len(eval_rows)}

    def _as_train_row(self, row: dict[str, Any]) -> dict[str, Any]:
        language = row.get("language") or "ne"
        messages = list(row.get("messages") or [])
        if not messages or messages[0].get("role") != "system":
            system = SYS_EN if language == "en" else SYS_NE
            messages = [{"role": "system", "content": system}, *messages]
        return {
            "id": row.get("id"),
            "language": language,
            "intent": row.get("intent"),
            "source": row.get("source") or row.get("teacher") or "chat",
            "messages": [
                {"role": item["role"], "content": item["content"]}
                for item in messages
                if item.get("role") in {"system", "user", "assistant"} and item.get("content")
            ],
        }

    def _read_log(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = self._read_jsonl(self.path)
        changed = False
        for row in rows:
            if not row.get("id"):
                row["id"] = uuid.uuid4().hex
                changed = True
            if "keep" not in row:
                row["keep"] = None
                changed = True
        if changed:
            self._write_log(rows)
        return rows

    def _write_log(self, rows: list[dict[str, Any]]) -> None:
        self._write_jsonl(self.path, rows)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
