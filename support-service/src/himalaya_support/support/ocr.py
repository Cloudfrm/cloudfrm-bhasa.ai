from __future__ import annotations

import base64
import logging

import httpx

from himalaya_support.config import Settings
from himalaya_support.inference.client import InferenceError

logger = logging.getLogger(__name__)


class DevanagariOCR:
    """Attachment OCR via himalaya-ai/glm-ocr-devanagari-finetuned."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def read_image(self, image_bytes: bytes, mime: str = "image/png") -> str:
        token = self.settings.hf_token.strip()
        if not token:
            raise InferenceError("HF_TOKEN is required for Himalaya OCR")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        url = f"https://api-inference.huggingface.co/models/{self.settings.ocr_model}"
        payload = {"inputs": {"image": f"data:{mime};base64,{encoded}"}}
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
            )
        if response.status_code >= 400:
            raise InferenceError(f"OCR failed ({response.status_code}): {response.text[:400]}")
        data = response.json()
        if isinstance(data, list) and data:
            first = data[0]
            return str(first.get("generated_text") or first.get("text") or first)
        if isinstance(data, dict):
            return str(data.get("generated_text") or data.get("text") or data)
        return str(data)
