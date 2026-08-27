"""E3/E4/E5/E14: capabilities at runtime, gated voice, health with timestamp, one count store."""
import pytest
from fastapi.testclient import TestClient

from himalaya_support.api import routes
from himalaya_support.config import get_settings
from himalaya_support.support.engine import SupportEngine
from himalaya_support.support.refusals import RefusalStrings


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(type(settings), "db_path", property(lambda self: tmp_path / "api.db"))
    fake = RefusalStrings("G-placeholder", "Q-placeholder", "test", "now")
    engine = SupportEngine(settings, refusals=fake)
    routes.get_engine.cache_clear()
    monkeypatch.setattr(routes, "get_engine", lambda: engine)
    from himalaya_support.main import app

    with TestClient(app) as tc:
        yield tc


def test_capabilities_reports_voice_not_deployed(client):
    data = client.get("/v1/capabilities").json()
    assert data["answer_path"] == "extractive"
    assert data["stt"] == {"available": False, "model": None, "reason": "not_deployed"}
    assert data["tts"] == {"available": False, "model": None, "reason": "not_deployed"}
    assert data["rate_limit"]["retry_after_header"] is False
    assert "checked_at" in data


def test_voice_routes_are_gated(client):
    assert client.post("/v1/support/speak", json={"text": "नमस्ते", "locale": "ne"}).status_code == 503
    assert client.post("/v1/support/calls/start", json={"locale": "ne"}).status_code == 503


def test_health_has_time_of_check(client):
    data = client.get("/v1/health").json()
    assert data["ok"] is True
    assert data["checked_at"].endswith("+00:00")


def test_chat_response_shape(client):
    res = client.post("/v1/support/chat", json={"message": "बचत खाताको ब्याजदर कति हो?", "locale": "auto"})
    assert res.status_code == 200
    data = res.json()
    assert data["kind"] == "answer"
    assert data["passage"]["id"] == "kb-001"
    assert data["reply"] == data["passage"]["text"]
    assert data["language"] == "ne"


def test_counts_come_from_one_store(client):
    """E4: the overview stat and the inbox count are the same list, so they cannot diverge."""
    for q in ("बचत खाताको ब्याजदर कति हो?", "कार्ड हराएमा के गर्नुपर्छ?"):
        client.post("/v1/support/chat", json={"message": q})
    rows = client.get("/v1/support/conversations?channel=chat").json()
    assert len(rows) == 2
    # the desk derives BOTH numbers from this one array — see static/desk-core.js deskCounts()
    import json
    import pathlib

    core = (pathlib.Path(__file__).resolve().parents[1] / "src/himalaya_support/static/desk-core.js").read_text(encoding="utf-8")
    assert "function deskCounts" in core
    html = (pathlib.Path(__file__).resolve().parents[1] / "src/himalaya_support/static/desk.js").read_text(encoding="utf-8")
    assert html.count("/v1/support/conversations?channel=chat") == 1, "conversations must be fetched from exactly one place"


def test_cors_wildcard_refused():
    from himalaya_support.main import _cors_allowlist

    assert "*" not in _cors_allowlist("*")
    assert _cors_allowlist("http://a.example, *") == ["http://a.example"]
