from __future__ import annotations

from himalaya_support.inference.client import HimalayaChatClient
from himalaya_support.inference.prompts import proofread_prompt


DEVANAGARI_RANGE = ("\u0900", "\u097F")


def looks_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


class Proofreader:
    """Repair messy Devanagari using HimalayaGPT, shaped by himalaya-ai/nepali-proofreader."""

    def __init__(self, client: HimalayaChatClient) -> None:
        self.client = client

    def repair(self, text: str) -> str:
        if not looks_devanagari(text) or len(text) < 8:
            return text
        result = self.client.chat(
            [
                {"role": "user", "content": proofread_prompt(text)},
            ],
            model=self.client.settings.proofreader_model,
            temperature=0.2,
            max_tokens=min(256, max(64, len(text) + 32)),
        )
        cleaned = result.text.strip()
        return cleaned or text
