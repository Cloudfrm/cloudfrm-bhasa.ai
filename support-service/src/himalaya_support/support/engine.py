"""Support engine — grounded generative answer path (Rupesh's decision, 2026-08-28).

    retrieve → Gemma writes the reply from the retrieved passage
            → numeric grounding check against that passage
            → fetched refusal strings when nothing relevant was retrieved or a
              figure is unsupported
            → the verbatim passage travels with every answer as provenance

This is a DIFFERENT architecture from the deployed product (bhasa-api), whose
published claim is "no generative model runs in the answer path". With
SUPPORT_ANSWER_PATH=extractive the engine behaves like the deployed product.
When the model is unreachable the verbatim passage is returned, labelled.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from himalaya_support.adapt.candidates import detect_question_language
from himalaya_support.adapt.devanagari import normalize
from himalaya_support.adapt.grounding import check_numeric_grounding
from himalaya_support.config import Settings
from himalaya_support.inference.client import ChatResult, HimalayaChatClient, InferenceError
from himalaya_support.inference.microsoft_tts import synthesize as edge_speak
from himalaya_support.inference.prompts import context_block, system_prompt
from himalaya_support.rag.retriever import Retriever
from himalaya_support.store.db import SupportStore
from himalaya_support.support.credentials import DECLINE, check_credentials
from himalaya_support.support.extractive import Answer, ExtractiveAnswerer
from himalaya_support.support.refusals import RefusalStrings, fetch_refusals, refusals_dict

logger = logging.getLogger(__name__)

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_THIN_MIN = {"ne": 24, "en": 20}


def detect_language(text: str) -> str:
    """Question language after protected tokens are set aside (E10)."""
    return detect_question_language(text)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_thin(text: str, language: str) -> bool:
    letters = re.findall(r"[\wऀ-ॿ]", text or "")
    return len(letters) < _THIN_MIN.get(language, 20)


class SupportEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        refusals: RefusalStrings | None | bool = False,
        client: HimalayaChatClient | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = Retriever(settings, knowledge_only=True)
        self.store = SupportStore(settings.db_path)
        self.client = client or HimalayaChatClient(settings)
        if refusals is False:
            refusals = fetch_refusals(settings.proof_url, settings.refusal_cache_path)
        self.refusals: RefusalStrings | None = refusals  # type: ignore[assignment]
        self.answerer = ExtractiveAnswerer(self.retriever, self.refusals)
        self.started_at = _now()
        self._chips: dict[str, list[str]] | None = None
        self._llm_probe: dict[str, Any] | None = None
        self._llm_probe_at = 0.0
        self.last_backend: dict[str, Any] | None = None

    # ------------------------------------------------------------ capabilities

    @property
    def generative(self) -> bool:
        return (self.settings.answer_path or "").strip().lower() != "extractive"

    def probe_llm(self, max_age: float = 60.0) -> dict[str, Any]:
        if self._llm_probe is None or time.time() - self._llm_probe_at > max_age:
            try:
                self._llm_probe = self.client.probe()
            except Exception as exc:  # noqa: BLE001
                self._llm_probe = {"backend": None, "model": None, "reachable": False, "detail": str(exc)[:160]}
            self._llm_probe["checked_at"] = _now()
            self._llm_probe_at = time.time()
        return self._llm_probe

    def capabilities(self) -> dict[str, Any]:
        voice = self.settings.voice_enabled
        probe = self.probe_llm() if self.generative else None
        return {
            "product": "bhasa",
            "answer_path": "generative_grounded" if self.generative else "extractive",
            "answer_path_note": (
                "retrieve → a generative model (Gemma) writes the reply from the retrieved passage → numeric "
                "grounding check → fetched refusal strings when nothing relevant is retrieved or a figure is "
                "unsupported. The verbatim passage is shown as provenance. This differs from the deployed "
                "product, which is extractive (no generative model in the answer path)."
                if self.generative
                else "a reply is either a passage copied verbatim from a retrieved document, one of the two "
                "fetched refusal strings, or a credential decline; no generative model runs in the answer path"
            ),
            "llm": (
                {**probe, "fallback": "verbatim passage, labelled, when the model is unreachable"}
                if probe is not None
                else {"backend": None, "model": None, "reachable": False, "detail": "not in the answer path"}
            ),
            "languages": [
                {"code": "ne", "label": "नेपाली", "status": "live"},
                {"code": "en", "label": "English", "status": "live"},
            ],
            "domains": sorted({doc.source for doc in self.retriever.documents}),
            "documents": len(self.retriever.documents),
            "stt": {"available": bool(voice), "model": None, "reason": None if voice else "not_deployed"},
            "tts": {"available": bool(voice), "model": None, "reason": None if voice else "not_deployed"},
            "grounding": {
                "available": self.refusals is not None,
                "gate": "numeric_grounding+refusal_strings" if self.generative else "extractive",
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
        """Sample questions verified to retrieve a passage from the loaded corpus (E6)."""
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
                    if self.answerer.answer(question, lang).kind == "answer":
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
            # Never store or echo the raw secret; never log it; never send it to a model.
            reply = DECLINE["ne"] if language == "ne" else DECLINE["en"]
            self.store.add_message(conversation_id, "user", guard.redacted, {"language": language, "redacted": guard.kinds})
            self.store.add_message(conversation_id, "assistant", reply, {"kind": "credential_decline"})
            return self._result(conversation_id, reply, language, "credential_decline", None, None, guard.redacted,
                                credential_kinds=guard.kinds, generated=False, model="policy", backend="guard")

        if self.refusals is None:
            raise InferenceError("refusal strings unavailable; cannot answer safely")

        # 1. Retrieval decides whether there is evidence at all (and refuses honestly if not).
        ext: Answer = self.answerer.answer(text, language)
        passage = ext.passage.public() if ext.passage else None
        history = self.store.recent_messages(conversation_id, limit=4)
        self.store.add_message(conversation_id, "user", text, {"language": language})

        reply, kind, refusal_type = ext.reply, ext.kind, ext.refusal_type
        generated, model, backend, note = False, "retrieval", "extractive", None
        if self.generative and ext.kind == "answer" and ext.passage is not None:
            # 2. The model writes the reply from the passage.
            try:
                first: ChatResult = self.client.chat(
                    self._build_messages(text, ext, history, language),
                    max_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                )
                draft = (first.text or "").strip()
                if is_thin(draft, language):
                    note = "model_reply_thin"
                else:
                    # 3. Every figure in the generated reply must occur in the passage.
                    grounded, failures = check_numeric_grounding(draft, ext.passage.document, [])
                    if grounded:
                        reply, kind, refusal_type = draft, "answer", None
                        generated, model, backend = True, first.model, first.backend
                    else:
                        reply, kind, refusal_type = self.refusals.quantity, "refusal", "quantity"
                        generated, model, backend = False, first.model, first.backend
                        note = "ungrounded_quantity: " + "; ".join(failures)
            except InferenceError as exc:
                note = "llm_unreachable"
                logger.info("Generative path unavailable, returning verbatim passage: %s", str(exc)[:200])
            if not generated and note in {"llm_unreachable", "model_reply_thin"}:
                backend = "extractive_fallback"
        self.last_backend = {"backend": backend, "model": model, "at": _now()}

        self.store.add_message(
            conversation_id,
            "assistant",
            reply,
            {"kind": kind, "refusal_type": refusal_type, "passage": passage, "language": language,
             "generated": generated, "model": model, "backend": backend, "note": note},
        )
        return self._result(conversation_id, reply, language, kind, refusal_type, passage, text,
                            generated=generated, model=model, backend=backend, note=note,
                            suggest_ticket=kind == "refusal", considered=ext.considered)

    @staticmethod
    def _result(conversation_id, reply, language, kind, refusal_type, passage, echo, *, credential_kinds=None,
                generated=False, model=None, backend=None, note=None, suggest_ticket=False, considered=None) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "reply": reply,
            "language": language,
            "kind": kind,
            "refusal_type": refusal_type,
            "passage": passage,
            "credential_kinds": credential_kinds or [],
            "echo": echo,
            "grounded": True,
            "generated": generated,
            "model": model,
            "backend": backend,
            "note": note,
            "suggest_ticket": suggest_ticket,
            "considered": considered or [],
        }

    def _build_messages(self, question: str, ext: Answer, history: list[dict[str, str]], language: str) -> list[dict[str, str]]:
        assert ext.passage is not None
        snippet = {"title": ext.passage.title, "text": ext.passage.document}
        rule = (
            "\nRULES: Use ONLY the NOTES. Do not add any number, rate, fee, date or limit that is not in the NOTES. "
            "If the NOTES do not answer the question, say you cannot confirm it. Reply in "
            + ("Nepali (Devanagari)." if language == "ne" else "English.")
        )
        messages = [{"role": "system", "content": system_prompt(self.settings.assistant_name, self.settings.honorific, language) + "\n" + context_block([snippet]) + rule}]
        for item in history[-2:]:
            if item["role"] in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": question})
        return messages

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
