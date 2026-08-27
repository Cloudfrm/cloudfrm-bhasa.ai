from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SERVICE_ROOT.parent
# Vercel (and most serverless runtimes) ship a read-only deployment bundle;
# only /tmp is writable at runtime.
_WRITABLE_DATA_ROOT = Path("/tmp/himalaya-support") if os.environ.get("VERCEL") else SERVICE_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", SERVICE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_token: str = Field(default="", alias="HF_TOKEN")
    chat_model: str = Field(
        default="himalaya-ai/himalaya-gemma-4-e2b-it",
        alias="HIMALAYA_CHAT_MODEL",
    )
    fallback_model: str = Field(
        default="himalaya-ai/himalayagpt-0.5b-it",
        alias="HIMALAYA_FALLBACK_MODEL",
    )
    ocr_model: str = Field(
        default="himalaya-ai/glm-ocr-devanagari-finetuned",
        alias="HIMALAYA_OCR_MODEL",
    )
    proofreader_model: str = Field(
        default="himalaya-ai/himalayagpt-0.5b-it",
        alias="HIMALAYA_PROOFREADER_MODEL",
    )
    inference_backend: str = Field(default="auto", alias="INFERENCE_BACKEND")
    inference_base_url: str = Field(default="", alias="INFERENCE_BASE_URL")
    ollama_host: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="himalaya-gemma", alias="OLLAMA_MODEL")
    local_gguf: str = Field(default="", alias="HIMALAYA_GGUF_PATH")
    llama_n_ctx: int = Field(default=2048, alias="LLAMA_N_CTX")
    llama_n_threads: int = Field(default=0, alias="LLAMA_N_THREADS")
    assistant_name: str = Field(default="Bhasa", alias="SUPPORT_ASSISTANT_NAME")
    honorific: str = Field(default="tapai", alias="SUPPORT_HONORIFIC")
    host: str = Field(default="127.0.0.1", alias="SUPPORT_HOST")
    port: int = Field(default=8088, alias="SUPPORT_PORT")
    max_new_tokens: int = Field(default=256, alias="SUPPORT_MAX_NEW_TOKENS")
    temperature: float = Field(default=0.4, alias="SUPPORT_TEMPERATURE")
    support_pipeline: str = Field(default="himalaya", alias="SUPPORT_PIPELINE")
    manage_llama: bool = Field(default=True, alias="SUPPORT_MANAGE_LLAMA")
    llama_port: int = Field(default=8081, alias="LLAMA_PORT")
    llama_ngl: int = Field(default=99, alias="LLAMA_NGL")
    api_key: str = Field(default="", alias="SUPPORT_API_KEY")
    # Exact-match allowlist only. A wildcard is refused at startup (E14).
    cors_origins: str = Field(
        default="http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8088,http://localhost:8088",
        alias="SUPPORT_CORS_ORIGINS",
    )
    # Answer path. Rupesh's decision (2026-08-28): a generative model stays in
    # the path. "generative_grounded" = retrieve → Gemma writes the reply →
    # numeric grounding check → fetched refusal strings. "extractive" = the
    # deployed product's architecture (verbatim passage only).
    answer_path: str = Field(default="generative_grounded", alias="SUPPORT_ANSWER_PATH")
    # Voice (STT/TTS) is not deployed. Controls render only when this is true
    # AND /v1/capabilities reports available === true.
    voice_enabled: bool = Field(default=False, alias="SUPPORT_VOICE_ENABLED")
    # Dev-only state harness at /dev/states. Off in production.
    dev_harness: bool = Field(default=False, alias="SUPPORT_DEV_HARNESS")
    # Public proof document that carries the two byte-frozen refusal strings.
    proof_url: str = Field(
        default="https://bhasa-api.fly.dev/proof/grounding_public.json",
        alias="BHASA_PROOF_URL",
    )
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_tts_model: str = Field(default="gemini-2.5-flash-preview-tts", alias="GEMINI_TTS_MODEL")
    gemini_voice: str = Field(default="Kore", alias="GEMINI_VOICE")

    @property
    def data_dir(self) -> Path:
        return SERVICE_ROOT / "data"

    @property
    def knowledge_path(self) -> Path:
        return self.data_dir / "knowledge" / "product.json"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def store_dir(self) -> Path:
        path = _WRITABLE_DATA_ROOT / "store"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_path(self) -> Path:
        return self.store_dir / "support.db"

    @property
    def refusal_cache_path(self) -> Path:
        return self.store_dir / "refusal_strings.json"

    @property
    def terminology_path(self) -> Path:
        return self.data_dir / "knowledge" / "terminology.json"

    @property
    def chips_path(self) -> Path:
        return self.data_dir / "knowledge" / "chips.json"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def resolve_gguf(self) -> Path | None:
        if self.local_gguf:
            path = Path(self.local_gguf)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path if path.exists() else None
        matches = sorted(self.models_dir.glob("*.gguf")) if self.models_dir.exists() else []
        preferred = [p for p in matches if "q4_k_m" in p.name.lower() or "q4" in p.name.lower()]
        return (preferred or matches)[0] if (preferred or matches) else None


def get_settings() -> Settings:
    return Settings()
