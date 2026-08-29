from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx

from himalaya_support.config import SERVICE_ROOT, Settings

logger = logging.getLogger(__name__)


def llama_binary() -> Path:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return SERVICE_ROOT / "bin" / "llama-vulkan" / name


def llama_ready(base_url: str) -> bool:
    root = base_url.rstrip("/").removesuffix("/v1")
    for path in ("/health", "/v1/models"):
        try:
            response = httpx.get(f"{root}{path}", timeout=2.0)
            if response.status_code < 500:
                return True
        except httpx.HTTPError:
            continue
    return False


def start_llama(settings: Settings) -> subprocess.Popen[bytes] | None:
    if not settings.manage_llama:
        return None
    base = settings.inference_base_url or f"http://127.0.0.1:{settings.llama_port}/v1"
    if llama_ready(base):
        logger.info("Gemma runtime already on %s", base)
        return None
    binary = llama_binary()
    model = settings.resolve_gguf()
    if not binary.exists() or model is None:
        logger.warning("No local Gemma runtime (missing llama-server or GGUF)")
        return None
    args = [
        str(binary),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.llama_port),
        "-c",
        str(settings.llama_n_ctx),
        "-n",
        str(settings.max_new_tokens),
        "--parallel",
        "1",
        "--alias",
        "himalaya-gemma",
        "-ngl",
        str(settings.llama_ngl),
    ]
    kwargs: dict = {"cwd": str(binary.parent), "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    logger.info("Starting local Gemma runtime on port %s", settings.llama_port)
    proc = subprocess.Popen(args, **kwargs)
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Gemma runtime exited before it became ready")
        if llama_ready(base):
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Gemma runtime did not start in time")


def stop_llama(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
