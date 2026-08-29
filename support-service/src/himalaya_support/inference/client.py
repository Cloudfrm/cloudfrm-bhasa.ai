from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from himalaya_support.config import Settings

logger = logging.getLogger(__name__)

HF_ROUTER = "https://router.huggingface.co/v1"
HF_INFERENCE = "https://api-inference.huggingface.co/v1"


class InferenceError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    backend: str
    raw: dict[str, Any]


class HimalayaChatClient:
    """Talks to Himalaya chat models through HF Inference or a local OpenAI-compatible server.

    Himalaya AI Labs publishes weights on Hugging Face; they do not run a public
    hosted chat API. This client is the integration layer:
    - hf_router / Hugging Face Inference (needs HF_TOKEN)
    - openai_compat (TGI, vLLM, llama.cpp --inet)
    - ollama after importing the GGUF
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._timeout = httpx.Timeout(120.0, connect=15.0)
        self._llm = None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        model = model or self.settings.chat_model
        temperature = self.settings.temperature if temperature is None else temperature
        max_tokens = self.settings.max_new_tokens if max_tokens is None else max_tokens
        backends = self._backend_order()
        errors: list[str] = []
        for backend in backends:
            try:
                return self._chat_via(
                    backend,
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except Exception as exc:  # noqa: BLE001 — try next backend
                logger.warning("Inference backend %s failed: %s", backend, exc)
                errors.append(f"{backend}: {exc}")
        raise InferenceError(self._help_text(errors))

    def _backend_order(self) -> list[str]:
        chosen = (self.settings.inference_backend or "auto").strip().lower()
        if chosen != "auto":
            return [chosen]
        order: list[str] = []
        if self.settings.inference_base_url:
            order.append("openai_compat")
        if self.settings.resolve_gguf() is not None:
            order.append("llamacpp")
        order.extend(["hf_router", "hf_inference", "ollama"])
        return order

    def _chat_via(
        self,
        backend: str,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResult:
        if backend == "openai_compat":
            if not self.settings.inference_base_url:
                raise InferenceError("INFERENCE_BASE_URL is empty")
            payload = self._openai_payload(model, messages, temperature, max_tokens, json_mode)
            data = self._post_openai(
                self.settings.inference_base_url.rstrip("/"),
                payload,
                api_key=self.settings.inference_api_key or None,
            )
            return ChatResult(self._choice_text(data), model, backend, data)
        if backend == "hf_router":
            payload = self._openai_payload(model, messages, temperature, max_tokens, json_mode)
            data = self._post_openai(HF_ROUTER, payload, require_token=True)
            return ChatResult(self._choice_text(data), model, backend, data)
        if backend == "hf_inference":
            payload = self._openai_payload(model, messages, temperature, max_tokens, json_mode)
            try:
                data = self._post_openai(HF_INFERENCE, payload, require_token=True)
                return ChatResult(self._choice_text(data), model, backend, data)
            except InferenceError:
                text = self._hf_text_generation(model, messages, max_tokens, temperature)
                return ChatResult(text, model, backend, {"generated_text": text})
        if backend == "ollama":
            return self._ollama_chat(messages, temperature, max_tokens)
        if backend == "llamacpp":
            return self._llamacpp_chat(messages, temperature, max_tokens, json_mode)
        raise InferenceError(f"Unknown inference backend: {backend}")

    def _openai_payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _headers(self, require_token: bool = False, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = (api_key if api_key is not None else self.settings.hf_token).strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif require_token:
            raise InferenceError("HF_TOKEN is required for Hugging Face Inference")
        return headers

    def _post_openai(
        self,
        base: str,
        payload: dict[str, Any],
        require_token: bool = False,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        url = f"{base}/chat/completions" if not base.endswith("/chat/completions") else base
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=self._headers(require_token, api_key), json=payload)
        if response.status_code >= 400:
            raise InferenceError(f"{url} returned {response.status_code}: {response.text[:500]}")
        return response.json()

    def _hf_text_generation(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        prompt = self._flatten_messages(messages)
        url = f"https://api-inference.huggingface.co/models/{model}"
        body = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=self._headers(require_token=True), json=body)
        if response.status_code >= 400:
            raise InferenceError(f"{url} returned {response.status_code}: {response.text[:500]}")
        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return str(data[0]["generated_text"]).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        raise InferenceError(f"Unexpected HF text-generation payload: {str(data)[:300]}")

    def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        url = f"{self.settings.ollama_host.rstrip('/')}/api/chat"
        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)
        if response.status_code >= 400:
            raise InferenceError(f"Ollama {url} returned {response.status_code}: {response.text[:400]}")
        data = response.json()
        text = (data.get("message") or {}).get("content") or ""
        if not text:
            raise InferenceError("Ollama returned an empty message")
        return ChatResult(text.strip(), self.settings.ollama_model, "ollama", data)

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        path = self.settings.resolve_gguf()
        if path is None:
            raise InferenceError("No local Himalaya GGUF found under data/models")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise InferenceError("llama-cpp-python is not installed") from exc
        threads = self.settings.llama_n_threads or None
        logger.info("Loading local GGUF %s", path)
        self._llm = Llama(
            model_path=str(path),
            n_ctx=self.settings.llama_n_ctx,
            n_threads=threads,
            n_gpu_layers=0,
            verbose=False,
        )
        return self._llm

    def _llamacpp_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> ChatResult:
        llm = self._get_llm()
        payload = list(messages)
        if json_mode:
            payload = payload + [{"role": "user", "content": "Reply with a single JSON object only."}]
        data = llm.create_chat_completion(
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = self._choice_text(data)
        return ChatResult(text, str(self.settings.resolve_gguf().name), "llamacpp", data)

    @staticmethod
    def _choice_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise InferenceError(f"No choices in completion: {json.dumps(data)[:300]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if not content:
            content = message.get("reasoning_content") or message.get("reasoning") or ""
            content = HimalayaChatClient._strip_thinking(str(content))
        if not content:
            raise InferenceError("Empty model content")
        return str(content).strip()

    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered.startswith("thinking process") or lowered.startswith("the user"):
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            for part in reversed(parts):
                if not part.lower().startswith(("thinking", "1.", "analyze", "the user")):
                    return part
            return ""
        return text

    @staticmethod
    def _flatten_messages(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for item in messages:
            role = item.get("role", "user")
            parts.append(f"<|im_start|>{role}\n{item.get('content', '')}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _help_text(errors: list[str]) -> str:
        detail = " | ".join(errors) if errors else "no backends attempted"
        return (
            "Could not reach a Himalaya chat model. These weights are not hosted as a "
            "public Himalaya API. Set HF_TOKEN and try Hugging Face Inference, or run "
            "himalaya-ai/himalaya-gemma-4-e2b-it-gguf locally (Ollama / llama.cpp / TGI) "
            f"and set INFERENCE_BASE_URL. Attempts: {detail}"
        )
