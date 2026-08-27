"""The two byte-frozen refusal strings.

They are NEVER typed into this codebase. They are fetched from the deployed
product's public proof document, NFC-normalised, cached on disk, and compared
byte-for-byte. A near-miss retyped here would make refusal detection fail
silently and a refusal would render as a normal answer.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Key paths inside grounding_public.json. The strings themselves live only there.
_GENERAL_PATH = ("case_c_question_outside_corpus", "reply")
_QUANTITY_PATH = ("case_b_quantity_not_in_corpus", "reply")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


@dataclass(frozen=True)
class RefusalStrings:
    general: str
    quantity: str
    source: str
    fetched_at: str
    from_cache: bool = False

    def classify(self, text: str) -> str | None:
        """Return 'general' | 'quantity' | None by NFC byte comparison."""
        probe = nfc(text).encode("utf-8")
        if probe == nfc(self.general).encode("utf-8"):
            return "general"
        if probe == nfc(self.quantity).encode("utf-8"):
            return "quantity"
        return None

    def public(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fetched_at": self.fetched_at,
            "from_cache": self.from_cache,
            "general_bytes": len(nfc(self.general).encode("utf-8")),
            "quantity_bytes": len(nfc(self.quantity).encode("utf-8")),
        }


def _dig(payload: dict[str, Any], path: tuple[str, ...]) -> str | None:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, str) and node.strip() else None


def _parse(payload: dict[str, Any], source: str, fetched_at: str, from_cache: bool) -> RefusalStrings | None:
    general = _dig(payload, _GENERAL_PATH)
    quantity = _dig(payload, _QUANTITY_PATH)
    if not general or not quantity or general == quantity:
        return None
    return RefusalStrings(nfc(general), nfc(quantity), source, fetched_at, from_cache)


def fetch_refusals(url: str, cache_path: Path, timeout: float = 8.0) -> RefusalStrings | None:
    """Fetch from the proof URL; fall back to the last good cache; else None."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            parsed = _parse(response.json(), url, now, from_cache=False)
            if parsed:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps({"proof": response.json(), "fetched_at": now, "url": url}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except OSError as exc:
                    logger.warning("Could not cache refusal strings: %s", exc)
                return parsed
            logger.warning("Proof document at %s did not contain both refusal strings", url)
        else:
            logger.warning("Proof fetch returned %s", response.status_code)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Proof fetch failed: %s", exc)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = _parse(cached.get("proof") or {}, cached.get("url") or url, cached.get("fetched_at") or "", True)
            if parsed:
                return parsed
        except (OSError, ValueError) as exc:
            logger.warning("Refusal cache unreadable: %s", exc)
    return None


def refusals_dict(refusals: RefusalStrings | None) -> dict[str, Any]:
    if refusals is None:
        return {"available": False, "reason": "refusal_strings_unavailable"}
    return {"available": True, **asdict(refusals) | refusals.public()}
