from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from himalaya_support.api.routes import get_engine, router
from himalaya_support.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Build the engine (fetches the refusal strings, indexes the corpus,
    # verifies the sample chips) before the first request.
    engine = get_engine()
    if engine.refusals is None:
        logger.error("Refusal strings unavailable: /v1/support/chat will return 503 until the proof document can be fetched")
    engine.chips()
    yield


app = FastAPI(
    title="bhasa officer desk",
    version="1.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def _cors_allowlist(raw: str) -> list[str]:
    """Exact-match origins only (E14). A wildcard is refused, not honoured."""
    origins = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if any(item == "*" or "*" in item for item in origins):
        logger.error("SUPPORT_CORS_ORIGINS contains a wildcard; ignoring it and using the local allowlist")
        origins = [item for item in origins if "*" not in item]
    if not origins:
        origins = ["http://127.0.0.1:8000", "http://localhost:8000"]
    return origins


CORS_ALLOWLIST = _cors_allowlist(settings.cors_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWLIST,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-Api-Key", "Authorization"],
)
app.include_router(router, prefix="/v1")

STATIC_DIR = Path(__file__).with_name("static")
CHAT_PAGE = STATIC_DIR / "chat.html"
DASHBOARD_PAGE = STATIC_DIR / "dashboard.html"
STATES_PAGE = STATIC_DIR / "states.html"


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    key = settings.api_key.strip()
    if (
        not key
        or request.url.path in {"/", "/chat", "/v1/health", "/v1/capabilities"}
        or request.url.path.startswith("/static/")
        or request.method == "OPTIONS"
    ):
        return await call_next(request)
    given = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        given = auth[7:].strip()
    if given != key:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def dashboard_page() -> FileResponse:
    return FileResponse(DASHBOARD_PAGE, headers={"Cache-Control": "no-store"})


@app.get("/chat", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE, headers={"Cache-Control": "no-store"})


@app.get("/dev/states", include_in_schema=False)
def states_page():
    """Dev-only harness that renders every non-happy state in isolation.
    Not served unless SUPPORT_DEV_HARNESS=true."""
    if not settings.dev_harness:
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return FileResponse(STATES_PAGE, headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "himalaya_support.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
