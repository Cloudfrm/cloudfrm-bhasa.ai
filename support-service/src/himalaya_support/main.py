from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from himalaya_support.api.routes import router
from himalaya_support.config import get_settings
from himalaya_support.inference.runtime import start_llama, stop_llama

settings = get_settings()
_llama_proc = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _llama_proc
    _llama_proc = start_llama(settings)
    yield
    stop_llama(_llama_proc)


app = FastAPI(
    title="Bhasa",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

origins = [item.strip() for item in settings.cors_origins.split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/v1")

CHAT_PAGE = Path(__file__).with_name("static") / "chat.html"
DASHBOARD_PAGE = Path(__file__).with_name("static") / "dashboard.html"
STATIC_DIR = Path(__file__).with_name("static")


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    key = settings.api_key.strip()
    if (
        not key
        or request.url.path in {"/", "/chat", "/v1/health"}
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
