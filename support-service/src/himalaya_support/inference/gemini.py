from __future__ import annotations

import base64
import io
import re
import wave
from typing import Any

import httpx

from himalaya_support.config import Settings
from himalaya_support.inference.client import ChatResult, InferenceError

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._timeout = httpx.Timeout(90.0, connect=15.0)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.gemini_api_key.strip())

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.4,
    ) -> ChatResult:
        if not self.enabled:
            raise InferenceError("GEMINI_API_KEY is not set")
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents: list[dict[str, Any]] = []
        for item in messages:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            contents.append(
                {
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": item["content"]}],
                }
            )
        payload: dict[str, Any] = {
            "contents": contents or [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens or self.settings.max_new_tokens,
                "temperature": temperature,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        data = self._generate(self.settings.gemini_model, payload)
        text = _extract_text(data).strip()
        if not text:
            raise InferenceError("Gemini returned an empty message")
        return ChatResult(text, data.get("model", self.settings.gemini_model), "gemini", data)

    def transcribe(self, audio: bytes, mime: str, locale: str = "ne") -> str:
        if not self.enabled:
            raise InferenceError("GEMINI_API_KEY is not set")
        script = (
            "Transcribe the speech. If it is Nepali, write Devanagari (नेपाली), not romanized. "
            "Return only the transcript."
            if locale == "ne"
            else "Transcribe the speech. Return only the transcript."
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": script},
                        {
                            "inline_data": {
                                "mime_type": mime or "audio/webm",
                                "data": base64.b64encode(audio).decode("ascii"),
                            }
                        },
                    ],
                }
            ]
        }
        data = self._generate(self.settings.gemini_model, payload)
        text = _extract_text(data).strip()
        if not text:
            raise InferenceError("Gemini returned an empty transcript")
        return text

    def speak(self, text: str, locale: str = "ne") -> bytes:
        if not self.enabled:
            raise InferenceError("GEMINI_API_KEY is not set")
        spoken = text.strip()
        if not spoken:
            raise InferenceError("Nothing to speak")
        lead = "Speak in natural Nepali:" if locale == "ne" else "Speak in clear English:"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": f"{lead} {spoken}"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "languageCode": "ne-NP" if locale == "ne" else "en-US",
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self.settings.gemini_voice}
                    },
                },
            },
        }
        last_error = "Gemini TTS failed"
        for model in _tts_models(self.settings.gemini_tts_model):
            try:
                data = self._generate(model, payload)
                return _extract_wav(data)
            except InferenceError as exc:
                last_error = str(exc)
        raise InferenceError(last_error)

    def _generate(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = GEMINI_URL.format(model=model)
        headers = {
            "x-goog-api-key": self.settings.gemini_api_key.strip(),
            "content-type": "application/json",
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise InferenceError(f"Gemini API {response.status_code}: {response.text[:400]}")
        return response.json()


def _tts_models(preferred: str) -> list[str]:
    models = [
        preferred,
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-pro-preview-tts",
        "gemini-3.1-flash-tts-preview",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def _extract_text(data: dict[str, Any]) -> str:
    parts = []
    for candidate in data.get("candidates") or []:
        for part in ((candidate.get("content") or {}).get("parts") or []):
            if part.get("text"):
                parts.append(part["text"])
    return "".join(parts)


def _extract_wav(data: dict[str, Any]) -> bytes:
    for candidate in data.get("candidates") or []:
        for part in ((candidate.get("content") or {}).get("parts") or []):
            blob = part.get("inlineData") or part.get("inline_data") or {}
            raw = blob.get("data")
            if not raw:
                continue
            audio = base64.b64decode(raw)
            mime = str(blob.get("mimeType") or blob.get("mime_type") or "")
            if "wav" in mime or audio[:4] == b"RIFF":
                return audio
            if "mpeg" in mime or "mp3" in mime:
                return audio
            rate = _rate_from_mime(mime)
            return pcm_to_wav(audio, rate=rate)
    raise InferenceError("Gemini TTS returned no audio")


def _rate_from_mime(mime: str) -> int:
    match = re.search(r"rate=(\d+)", mime or "")
    return int(match.group(1)) if match else 24000


def pcm_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()
