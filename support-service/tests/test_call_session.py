from himalaya_support.api.routes import hangup_call, start_call
from himalaya_support.api.schemas import CallHangupRequest, CallStartRequest
from himalaya_support.config import get_settings
from himalaya_support.inference.client import InferenceError
from himalaya_support.store.db import SupportStore
from himalaya_support.support.engine import SupportEngine


def _engine(tmp_path):
    engine = SupportEngine(get_settings())
    engine.store = SupportStore(tmp_path / "call.db")
    engine.speak = lambda text, locale="ne": (_ for _ in ()).throw(InferenceError("skip tts"))
    return engine


def test_hangup_is_idempotent(tmp_path):
    store = SupportStore(tmp_path / "support.db")
    call_id = store.get_or_create_conversation(None, None, "ne", channel="call")
    assert store.end_conversation(call_id) == "ended"
    assert store.end_conversation(call_id) == "ended"
    assert store.end_conversation("missing-id") == "ended"
    assert store.get_open_call() is None


def test_end_open_calls_leaves_chat_alone(tmp_path):
    store = SupportStore(tmp_path / "support.db")
    chat_id = store.get_or_create_conversation(None, None, "ne", channel="chat")
    call_id = store.get_or_create_conversation(None, None, "ne", channel="call")
    ended = store.end_open_calls()
    assert ended == [call_id]
    assert store.end_open_calls() == []
    chats = store.list_conversations(channel="chat")
    assert chats[0]["id"] == chat_id
    assert chats[0]["status"] == "open"


def test_second_start_after_hangup_greets_again(tmp_path):
    engine = _engine(tmp_path)
    first = engine.start_call("en")
    assert first["status"] == "live"
    assert "Namaste" in first["reply"]
    hung = engine.end_call(first["conversation_id"])
    assert hung["ok"] is True
    assert hung["status"] == "ended"
    again = engine.end_call(first["conversation_id"])
    assert again["ok"] is True
    second = engine.start_call("en")
    assert second["status"] == "live"
    assert second["conversation_id"] != first["conversation_id"]
    assert "Namaste" in second["reply"]
    open_call = engine.store.get_open_call()
    assert open_call["id"] == second["conversation_id"]


def test_start_replaces_previous_open_call(tmp_path):
    engine = _engine(tmp_path)
    first = engine.start_call("ne")
    second = engine.start_call("ne")
    assert second["conversation_id"] != first["conversation_id"]
    open_call = engine.store.get_open_call()
    assert open_call["id"] == second["conversation_id"]
    ended = engine.store.list_conversations(channel="call")
    first_row = next(row for row in ended if row["id"] == first["conversation_id"])
    assert first_row["status"] == "ended"


def test_hangup_without_id_ends_active_call(tmp_path):
    engine = _engine(tmp_path)
    started = engine.start_call("ne")
    result = engine.end_call(None)
    assert result["ok"] is True
    assert started["conversation_id"] in (result["ended"] or [])
    assert engine.store.get_open_call() is None
    idle = engine.end_call(None)
    assert idle["ok"] is True
    assert idle["ended"] == []


def test_call_routes_start_and_idempotent_hangup(tmp_path, monkeypatch):
    from himalaya_support.api import routes as api_routes

    engine = _engine(tmp_path)
    monkeypatch.setattr(api_routes, "get_engine", lambda: engine)
    started = start_call(CallStartRequest(locale="en"))
    assert started["status"] == "live"
    first = hangup_call(CallHangupRequest(conversation_id=started["conversation_id"]))
    second = hangup_call(CallHangupRequest(conversation_id=started["conversation_id"]))
    assert first["ok"] is True and second["ok"] is True
    retry = start_call(CallStartRequest(locale="en"))
    assert retry["conversation_id"] != started["conversation_id"]
    assert retry["status"] == "live"
