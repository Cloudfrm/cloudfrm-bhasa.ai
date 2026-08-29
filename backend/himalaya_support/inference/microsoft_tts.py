from __future__ import annotations

import asyncio

from himalaya_support.inference.client import InferenceError

NE_VOICE = "ne-NP-HemkalaNeural"
EN_VOICE = "en-US-JennyNeural"


def synthesize(text: str, locale: str = "ne") -> bytes:
    spoken = (text or "").strip()
    if not spoken:
        raise InferenceError("Nothing to speak")
    voice = NE_VOICE if locale == "ne" else EN_VOICE
    try:
        import edge_tts
    except ImportError as exc:
        raise InferenceError("edge-tts is not installed") from exc

    async def _run() -> bytes:
        chunks: list[bytes] = []
        communicate = edge_tts.Communicate(spoken, voice)
        async for item in communicate.stream():
            if item["type"] == "audio":
                chunks.append(item["data"])
        return b"".join(chunks)

    try:
        audio = asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            audio = loop.run_until_complete(_run())
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Edge TTS failed: {exc}") from exc
    if not audio:
        raise InferenceError("Edge TTS returned no audio")
    return audio
