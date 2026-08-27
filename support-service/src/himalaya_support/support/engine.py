"""Support engine — extractive answer path.

A reply is either a passage quoted verbatim from a loaded bank/product
document, one of the two fetched refusal strings, or a credential decline.
No generative model runs in `chat()`.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from himalaya_support.adapt.candidates import detect_question_language
from himalaya_support.adapt.devanagari import normalize
from himalaya_support.config import Settings
from himalaya_support.inference.client import InferenceError
from himalaya_support.inference.microsoft_tts import synthesize as edge_speak
from himalaya_support.rag.retriever import Retriever
from himalaya_support.store.db import SupportStore
from himalaya_support.support.credentials import DECLINE, check_credentials
from himalaya_support.support.extractive import Answer, ExtractiveAnswerer
from himalaya_support.support.refusals import RefusalStrings, fetch_refusals, refusals_dict

logger = logging.getLogger(__name__)

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def detect_language(text: str) -> str:
    """Question language after protected tokens are set aside (E10)."""
    return detect_question_language(text)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SupportEngine:
    def __init__(self, settings: Settings, *, refusals: RefusalStrings | None | bool = False) -> None:
        self.settings = settings
        self.retriever = Retriever(settings, knowledge_only=True)
        self.store = SupportStore(settings.db_path)
        if refusals is False:
            refusals = fetch_refusals(settings.proof_url, settings.refusal_cache_path)
        self.refusals: RefusalStrings | None = refusals  # type: ignore[assignment]
        self.answerer = ExtractiveAnswerer(self.retriever, self.refusals)
        self.started_at = _now()
        self._chips: dict[str, list[str]] | None = None

    # ------------------------------------------------------------ capabilities

    def capabilities(self) -> dict[str, Any]:
        voice = self.settings.voice_enabled
        return {
            "product": "bhasa",
            "answer_path": "extractive",
            "answer_path_note": (
                "a reply is either a passage copied verbatim from a retrieved document, one of the two "
                "fetched refusal strings, or a credential decline; no generative model runs in the answer path"
            ),
            "languages": [
                {"code": "ne", "label": "नेपाली", "status": "live"},
                {"code": "en", "label": "English", "status": "live"},
            ],
            "domains": sorted({tag for doc in self.retriever.documents for tag in [doc.source]}),
            "documents": len(self.retriever.documents),
            "stt": {"available": bool(voice), "model": None, "reason": None if voice else "not_deployed"},
            "tts": {"available": bool(voice), "model": None, "reason": None if voice else "not_deployed"},
            "grounding": {
                "available": self.refusals is not None,
                "gate": "extractive",
                "refusal_strings": refusals_dict(self.refusals) if self.refusals else {"available": False, "reason": "refusal_strings_unavailable"},
            },
            "rate_limit": {"enabled": False, "retry_after_header": False},
            "transport": {"websocket": False, "http": True},
            "dev_harness": bool(self.settings.dev_harness),
            "checked_at": _now(),
        }

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "checked_at": _now(),
            "started_at": self.started_at,
            "grounding_ready": self.refusals is not None,
            "documents": len(self.retriever.documents),
        }

    # ------------------------------------------------------------------- chips

    def chips(self) -> dict[str, list[str]]:
        """Sample questions verified answerable by the loaded corpus (E6)."""
        if self._chips is not None:
            return self._chips
        out: dict[str, list[str]] = {"ne": [], "en": []}
        path = self.settings.chips_path
        if path.exists() and self.refusals is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                payload = {}
            for lang in ("ne", "en"):
                for question in payload.get(lang) or []:
                    result = self.answerer.answer(question, lang)
                    if result.kind == "answer":
                        out[lang].append(question)
        self._chips = out
        return out

    # -------------------------------------------------------------------- chat

    def chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
        locale: str = "auto",
        channel: str = "chat",
    ) -> dict[str, Any]:
        text = normalize(message)  # NFC only
        language = self._reply_language(locale, text)

        guard = check_credentials(text)
        conversation_id = self.store.get_or_create_conversation(conversation_id, user_id, language, channel=channel)

        if guard.detected:
            # Never store or echo the raw secret; never log it.
            reply = DECLINE["ne"] if language == "ne" else DECLINE["en"]
            self.store.add_message(conversation_id, "user", guard.redacted, {"language": language, "redacted": guard.kinds})
            self.store.add_message(conversation_id, "assistant", reply, {"kind": "credential_decline"})
            return {
                "conversation_id": conversation_id,
                "reply": reply,
                "language": language,
                "kind": "credential_decline",
                "refusal_type": None,
                "passage": None,
                "credential_kinds": guard.kinds,
                "echo": guard.redacted,
                "grounded": True,
                "suggest_ticket": False,
            }

        if self.refusals is None:
            raise InferenceError("refusal strings unavailable; cannot answer safely")

        answer: Answer = self.answerer.answer(text, language)
        self.store.add_message(conversation_id, "user", text, {"language": language})
        self.store.add_message(
            conversation_id,
            "assistant",
            answer.reply,
            {
                "kind": answer.kind,
                "refusal_type": answer.refusal_type,
                "passage": answer.passage.public() if answer.passage else None,
                "language": language,
            },
        )
        return {
            "conversation_id": conversation_id,
            "reply": answer.reply,
            "language": language,
            "kind": answer.kind,
            "refusal_type": answer.refusal_type,
            "passage": answer.passage.public() if answer.passage else None,
            "credential_kinds": [],
            "echo": text,
            "grounded": True,
            "suggest_ticket": answer.kind == "refusal",
            "considered": answer.considered,
        }

    def _reply_language(self, locale: str, message: str) -> str:
        choice = (locale or "auto").strip().lower()
        if choice in {"ne", "nepali", "np"}:
            return "ne"
        if choice in {"en", "english"}:
            return "en"
        return detect_language(message)

    # ------------------------------------------------------------------- voice
    # Kept only behind SUPPORT_VOICE_ENABLED. /v1/capabilities reports
    # available:false, reason:"not_deployed" until that flag is true.

    def _require_voice(self) -> None:
        if not self.settings.voice_enabled:
            raise InferenceError("voice not_deployed")

    def start_call(self, locale: str = "ne") -> dict[str, Any]:
        self._require_voice()
        language = self._reply_language(locale, "")
        greeting = (
            "नमस्ते, म bhasa हुँ। सदस्यको प्रश्न भन्नुहोस्।"
            if language == "ne"
            else "Namaste, this is bhasa. Please state the member's question."
        )
        conversation_id = self.store.get_or_create_conversation(None, None, language, channel="call")
        self.store.add_message(conversation_id, "assistant", greeting, {"language": language, "channel": "call", "kind": "greeting"})
        audio_b64 = None
        mime = None
        try:
            mime, audio = self.speak(greeting, language)
            audio_b64 = base64.b64encode(audio).decode("ascii")
        except InferenceError as exc:
            logger.info("Call greeting TTS skipped: %s", exc)
        return {"conversation_id": conversation_id, "reply": greeting, "language": language, "audio_base64": audio_b64, "mime": mime}

    def speak(self, text: str, locale: str = "ne") -> tuple[str, bytes]:
        self._require_voice()
        return "audio/mpeg", edge_speak((text or "").strip(), locale)
