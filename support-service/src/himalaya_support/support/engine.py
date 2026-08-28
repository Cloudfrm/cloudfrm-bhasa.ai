from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

from himalaya_support.config import Settings
from himalaya_support.inference.microsoft_tts import synthesize as edge_speak
from himalaya_support.inference.gemini import GeminiClient
from himalaya_support.inference.client import ChatResult, HimalayaChatClient, InferenceError
from himalaya_support.inference.prompts import (
    context_block,
    hybrid_system_prompt,
    intent_prompt,
    system_prompt,
)
from himalaya_support.rag.retriever import Retriever
from himalaya_support.store.db import SupportStore
from himalaya_support.support.finetune import SFTRecorder
from himalaya_support.adapt.actions import parse_confirmation
from himalaya_support.adapt.conversation_repair import repair_message
from himalaya_support.adapt.pipeline import (
    ReplyGenerationError,
    finish_reply,
    prepare_user_text,
)
from himalaya_support.adapt.speech import normalize_for_speech
from himalaya_support.support.honorific import uses_informal_register
from himalaya_support.support.ocr import DevanagariOCR
from himalaya_support.support.proofread import Proofreader, looks_devanagari
from himalaya_support.support.tools import ToolRunner, extract_tool_calls, strip_tool_markup

logger = logging.getLogger(__name__)

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ROMANIZED_NE_RE = re.compile(
    r"\b(sanchai|sancho|namaste|hajur|tapai|timi|kasto|chha|hunxa|hunuhunchha|"
    r"hunuhunxa|dhanyabad|ramro|malai|mero|saman|thyo|garne|birse|phirta|"
    r"k cha|k cha\?|ke cha)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    dev = len(DEVANAGARI_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    romanized = bool(ROMANIZED_NE_RE.search(text))
    if romanized and not dev:
        return "ne"
    if dev and latin and min(dev, latin) / max(dev, latin) > 0.25:
        return "mixed"
    if dev > latin:
        return "ne"
    return "en"


class SupportEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = HimalayaChatClient(settings)
        self.retriever = Retriever(settings)
        self.store = SupportStore(settings.db_path)
        self.tools = ToolRunner(self.store)
        self.proofreader = Proofreader(self.client)
        self.ocr = DevanagariOCR(settings)
        self.gemini = GeminiClient(settings)
        self.sft = SFTRecorder(settings)
        self._stt_failures: dict[str, int] = {}
        self._pending_ticket: dict[str, dict[str, Any]] = {}

    def chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
        locale: str = "auto",
        proofread: bool = True,
        channel: str = "chat",
    ) -> dict[str, Any]:
        prepared = prepare_user_text(message)
        search_text = prepared["search"] or prepared["normalized"] or message
        language = self._reply_language(locale, search_text)
        conversation_id = self.store.get_or_create_conversation(
            conversation_id, user_id, language, channel=channel
        )

        confirmed = parse_confirmation(search_text)
        if conversation_id in self._pending_ticket:
            pending = self._pending_ticket[conversation_id]
            if confirmed is True:
                ticket = self.store.create_ticket(pending)
                del self._pending_ticket[conversation_id]
                reply_text = (
                    f"टिकट खुल्यो — {ticket['id'][:8]}। अधिकृतले हेर्छन्।"
                    if language == "ne"
                    else f"Ticket opened — {ticket['id'][:8]}. An officer will follow up."
                )
                self.store.add_message(conversation_id, "user", message, {"language": language})
                self.store.add_message(conversation_id, "assistant", reply_text, {"tickets": [ticket["id"]]})
                return {
                    "conversation_id": conversation_id,
                    "reply": reply_text,
                    "language": language,
                    "intent": {"intent": "ticket", "needs_ticket": True},
                    "model": "policy",
                    "backend": "actions",
                    "pipeline": "confirm",
                    "transliterated": prepared.get("transliterated"),
                    "grounded": True,
                    "register": "spoken",
                    "tickets": [ticket["id"]],
                    "pending_confirm": None,
                    "retrieved": [],
                    "tools": [{"name": "create_ticket", "ticket": ticket}],
                    "generated": False,
                    "dataset_grounded_only": True,
                }
            if confirmed is False:
                del self._pending_ticket[conversation_id]
                reply_text = "हुन्छ, टिकट खोलिनँ। अरू के सहयोग?" if language == "ne" else "Okay, no ticket. What else can I help with?"
                self.store.add_message(conversation_id, "user", message, {"language": language})
                self.store.add_message(conversation_id, "assistant", reply_text, {})
                return {
                    "conversation_id": conversation_id,
                    "reply": reply_text,
                    "language": language,
                    "intent": {"intent": "other", "needs_ticket": False},
                    "model": "policy",
                    "backend": "actions",
                    "pipeline": "confirm",
                    "transliterated": prepared.get("transliterated"),
                    "grounded": True,
                    "register": "spoken",
                    "tickets": [],
                    "pending_confirm": None,
                    "retrieved": [],
                    "tools": [],
                    "generated": False,
                    "dataset_grounded_only": True,
                }

        cleaned = search_text
        local = self._is_local()
        if proofread and looks_devanagari(search_text) and not local:
            try:
                cleaned = self.proofreader.repair(search_text)
            except InferenceError as exc:
                logger.info("Proofread skipped: %s", exc)

        intent = self._classify_heuristic(cleaned)
        snippets = self._pick_snippet(cleaned, language)
        history = self.store.recent_messages(conversation_id, limit=4)
        self.store.add_message(
            conversation_id,
            "user",
            search_text,
            {
                "language": language,
                "raw": message,
                "proofread": cleaned,
                "transliterated": prepared.get("transliterated"),
            },
        )

        messages = self._build_messages(cleaned, snippets, history, language)
        pipeline = self._pipeline()
        gemma_draft = None
        first = None
        if pipeline in {"hybrid", "himalaya"}:
            try:
                first = self.client.chat(
                    messages,
                    max_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                )
                gemma_draft = first.text
            except InferenceError as exc:
                logger.info("Himalaya Gemma draft skipped: %s", exc)
        if pipeline in {"hybrid", "gemini"} and self.gemini.enabled:
            first = self._gemini_finalize(cleaned, snippets, history, gemma_draft, language)
        if first is None:
            first = ChatResult(self._offline_reply(cleaned, language), "offline", "fallback", {})
        tool_results: list[dict[str, Any]] = []
        reply_text = strip_tool_markup(first.text)
        try:
            reply_text, safety = finish_reply(
                reply_text,
                snippets,
                language,
                intent=str(intent.get("intent") or "other"),
                user_message=cleaned,
            )
        except ReplyGenerationError as exc:
            # Retry once without the knowledge rows, so a malformed corpus row
            # cannot be spliced in a second time.
            logger.error("Reply rejected: %s", exc)
            try:
                reply_text, safety = finish_reply(
                    "",
                    [],
                    language,
                    intent=str(intent.get("intent") or "other"),
                    user_message=cleaned,
                )
            except ReplyGenerationError as retry_exc:
                logger.error("Reply rejected on retry: %s", retry_exc)
                raise InferenceError(
                    "Support could not compose a safe answer"
                ) from retry_exc
        pending_confirm = None
        if intent.get("needs_ticket"):
            self._pending_ticket[conversation_id] = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "subject": intent.get("summary") or cleaned[:80],
                "description": cleaned,
                "category": intent.get("intent") or "other",
                "priority": intent.get("priority") or "normal",
            }
            pending_confirm = "create_ticket"
            extra = " टिकट खोलौं हो?" if language == "ne" else " Open a ticket? Say yes or no."
            if extra.strip() not in reply_text:
                reply_text = reply_text.rstrip() + extra
        used_model = first.model
        backend = first.backend

        ticket_ids: list[str] = []
        self.store.add_message(
            conversation_id,
            "assistant",
            reply_text,
            {
                "model": used_model,
                "backend": backend,
                "intent": intent,
                "tickets": ticket_ids,
                "pipeline": pipeline,
                "safety": safety,
            },
        )
        sft_id = self.sft.record(
            cleaned,
            reply_text,
            language=language,
            intent=str(intent.get("intent", "other")),
            sources=[hit["source"] for hit in snippets],
            teacher=backend,
            gemma_draft=gemma_draft,
        )
        return {
            "conversation_id": conversation_id,
            "reply": reply_text,
            "language": language,
            "intent": intent,
            "model": used_model,
            "backend": backend,
            "pipeline": pipeline,
            "gemma_draft": gemma_draft,
            "sft_id": sft_id,
            "sft_recorded": True,
            "transliterated": prepared.get("transliterated"),
            "grounded": safety.get("grounded", True),
            "register": safety.get("register"),
            "pending_confirm": pending_confirm,
            "retrieved": [
                {"source": hit["source"], "title": hit["title"], "score": hit["score"]}
                for hit in snippets
            ],
            "tools": tool_results,
            "tickets": ticket_ids,
            "generated": True,
            "dataset_grounded_only": False,
        }

    def _reply_language(self, locale: str, message: str) -> str:
        choice = (locale or "auto").strip().lower()
        if choice in {"ne", "nepali", "np"}:
            return "ne"
        if choice in {"en", "english"}:
            return "en"
        return detect_language(message)

    def _pick_snippet(self, query: str, language: str) -> list[dict]:
        hits = self.retriever.search(query, k=8)
        usable = []
        for hit in hits:
            source = str(hit.get("source") or "")
            if "honorific" in source.lower():
                continue
            usable.append(hit)
        preferred: list[dict] = []
        rest: list[dict] = []
        for hit in usable:
            doc_id = str(hit.get("id", ""))
            source = str(hit.get("source") or "")
            is_ne = (
                doc_id.endswith("-ne")
                or "banking" in source
                or "नेपाली" in hit.get("title", "")
            )
            if language == "ne" and is_ne:
                preferred.append(hit)
            elif language != "ne" and not is_ne:
                preferred.append(hit)
            else:
                rest.append(hit)
        return (preferred or rest or usable)[:3]

    def _offline_reply(self, message: str, language: str) -> str:
        if language == "ne":
            return (
                "सञ्चै, मैले तपाईंको कुरा सुनें। "
                "लगइन, रकम स्थानान्तरण, ऋण किस्ता, केवाईसी, वा कार्ड — कुन विषय हो?"
            )
        return (
            "I heard you. Which topic can I help with — "
            "mobile banking login, a transfer, a loan instalment, KYC, or a card?"
        )

    def start_call(self, locale: str = "ne") -> dict[str, Any]:
        language = self._reply_language(locale, "")
        greeting = (
            "नमस्ते, म भाषा हुँ। कल उठ्यो। लगइन, रकम, ऋण, केवाईसी वा कार्ड — भन्नुहोस्, के सहयोग चाहियो?"
            if language == "ne"
            else "Namaste, I am Bhasa. The call is connected. Login, transfer, loan, KYC, or a card — how can I help?"
        )
        conversation_id = self.store.get_or_create_conversation(None, None, language, channel="call")
        self.store.add_message(
            conversation_id,
            "assistant",
            greeting,
            {"language": language, "channel": "call", "kind": "greeting"},
        )
        audio_b64 = None
        mime = None
        try:
            mime, audio = self.speak(greeting, language)
            audio_b64 = base64.b64encode(audio).decode("ascii")
        except InferenceError as exc:
            logger.info("Call greeting TTS skipped: %s", exc)
        return {
            "conversation_id": conversation_id,
            "reply": greeting,
            "language": language,
            "audio_base64": audio_b64,
            "mime": mime,
        }

    def voice_chat(
        self,
        audio: bytes,
        mime: str,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
        locale: str = "ne",
    ) -> dict[str, Any]:
        transcript = (self.gemini.transcribe(audio, mime, locale) or "").strip()
        key = conversation_id or "anon"
        if not transcript:
            self._stt_failures[key] = self._stt_failures.get(key, 0) + 1
            reply = repair_message(self._stt_failures[key] - 1)
            return {
                "conversation_id": conversation_id or "",
                "reply": reply,
                "language": locale,
                "transcript": "",
                "audio_base64": None,
                "audio_mime": None,
            }
        self._stt_failures[key] = 0
        result = self.chat(
            transcript,
            conversation_id=conversation_id,
            user_id=user_id,
            locale=locale,
            proofread=False,
            channel="call",
        )
        result["transcript"] = transcript
        result["audio_base64"] = None
        result["audio_mime"] = None
        try:
            mime, audio = self.speak(result["reply"], result.get("language") or locale)
            result["audio_base64"] = base64.b64encode(audio).decode("ascii")
            result["audio_mime"] = mime
        except InferenceError as exc:
            logger.info("TTS skipped: %s", exc)
        return result

    def speak(self, text: str, locale: str = "ne") -> tuple[str, bytes]:
        spoken = normalize_for_speech(text) if locale == "ne" else (text or "").strip()
        if self.gemini.enabled:
            try:
                return "audio/wav", self.gemini.speak(spoken, locale)
            except InferenceError as exc:
                logger.info("Gemini TTS skipped: %s", exc)
        return "audio/mpeg", edge_speak(spoken, locale)

    def _pipeline(self) -> str:
        chosen = (self.settings.support_pipeline or "hybrid").strip().lower()
        if chosen == "claude":
            chosen = "gemini"
        if chosen == "hybrid" and not self.gemini.enabled:
            return "himalaya"
        if chosen == "gemini" and not self.gemini.enabled:
            return "himalaya"
        return chosen

    def _gemini_finalize(
        self,
        message: str,
        snippets: list[dict],
        history: list[dict[str, str]],
        gemma_draft: str | None,
        reply_language: str = "ne",
    ):
        draft = gemma_draft or "(Himalaya Gemma did not return a draft)"
        messages = [
            {
                "role": "system",
                "content": hybrid_system_prompt(
                    self.settings.assistant_name,
                    self.settings.honorific,
                    reply_language,
                ),
            },
            {"role": "system", "content": context_block(snippets)},
            {
                "role": "system",
                "content": f"Himalaya Gemma draft (may be wrong; do not copy if off-topic):\n{draft}",
            },
        ]
        for item in history[-8:]:
            if item["role"] in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": message})
        return self.gemini.chat(messages)

    def _is_local(self) -> bool:
        backend = (self.settings.inference_backend or "auto").lower()
        return backend in {"llamacpp", "ollama"} or (
            backend == "auto" and self.settings.resolve_gguf() is not None
        )

    def _classify_heuristic(self, message: str) -> dict[str, Any]:
        text = message.lower()
        mapping = (
            ("refund", ("refund", "फिर्ता", "chargeback", "नपुगे", "दोहोरो")),
            ("billing", ("bill", "payment", "subscription", "भुक्तानी", "ब्याज", "शुल्क", "plan")),
            ("account", ("password", "login", "otp", "pin", "पिन", "पासवर्ड", "लगइन", "खाता", "ओटीपी")),
            ("loan", ("loan", "emi", "ऋण", "किस्ता", "कर्जा")),
            ("technical", ("crash", "bug", "error", "sync", "क्र्यास", "त्रुटि")),
            ("onboarding", ("signup", "sign up", "verify", "केवाईसी", "kyc", "साइन अप")),
            ("policy", ("privacy", "delete", "गोपनीयता", "मेटाउ")),
        )
        intent = "other"
        for name, keys in mapping:
            if any(key in text for key in keys):
                intent = name
                break
        if intent == "other" and (ROMANIZED_NE_RE.search(message) or "नमस्ते" in message or "सञ्चै" in message):
            if len(message.split()) <= 8:
                intent = "greeting"
        return {
            "intent": intent,
            "priority": "high" if any(w in text for w in ("urgent", "angry", "lawsuit", "हतार")) else "normal",
            "language": detect_language(message),
            "needs_ticket": intent in {"refund", "billing", "loan"},
            "needs_human": False,
            "summary": message[:160],
        }

    def _classify(self, message: str) -> dict[str, Any]:
        fallback = {
            "intent": "other",
            "priority": "normal",
            "language": detect_language(message),
            "needs_ticket": False,
            "needs_human": False,
            "summary": message[:160],
        }
        try:
            result = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You extract structured JSON for support routing. "
                            "Follow himalaya-ai/nepali-json-mode-singleturn: valid JSON only."
                        ),
                    },
                    {"role": "user", "content": intent_prompt(message)},
                ],
                json_mode=True,
                temperature=0.2,
                max_tokens=256,
            )
            parsed = json.loads(self._extract_json(result.text))
            fallback.update({key: parsed[key] for key in fallback if key in parsed})
        except Exception as exc:  # noqa: BLE001
            logger.info("Intent JSON fallback: %s", exc)
        return fallback

    def _build_messages(
        self,
        message: str,
        snippets: list[dict],
        history: list[dict[str, str]],
        reply_language: str = "ne",
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": system_prompt(
                    self.settings.assistant_name,
                    self.settings.honorific,
                    reply_language,
                )
                + "\n"
                + context_block(snippets),
            },
        ]
        for item in history[-2:]:
            if item["role"] in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": message})
        return messages

    def _run_tools(
        self,
        model_text: str,
        conversation_id: str,
        user_id: str | None,
        user_message: str,
        intent: dict[str, Any],
    ) -> list[dict[str, Any]]:
        calls = extract_tool_calls(model_text)
        if intent.get("needs_human") and not any(call["name"] == "escalate_to_human" for call in calls):
            calls.append({"name": "escalate_to_human", "arguments": {"reason": intent.get("summary")}})
        results = []
        context = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_message": user_message,
            "intent": intent.get("intent"),
            "retriever": self.retriever,
        }
        for call in calls:
            results.append(
                {
                    "name": call["name"],
                    **self.tools.run(call["name"], call.get("arguments") or {}, context),
                }
            )
        return results

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text
