import json
import re

from himalaya_support.config import get_settings
from himalaya_support.inference.gemini import pcm_to_wav
from himalaya_support.rag.retriever import Retriever
from himalaya_support.support.engine import detect_language
from himalaya_support.support.finetune import SFTRecorder
from himalaya_support.support.gold_seed import PAIRS
from himalaya_support.support.honorific import uses_informal_register
from himalaya_support.support.tools import extract_tool_calls, strip_tool_markup


def test_detect_language():
    assert detect_language("पासवर्ड बिर्सें") == "ne"
    assert detect_language("I forgot my password") == "en"


def test_honorific_markers():
    assert uses_informal_register("तिमीले पासवर्ड रिसेट गर्नुहोस्")
    assert not uses_informal_register("तपाईंले पासवर्ड रिसेट गर्नुहोस्")


def test_tool_call_parse():
    text = 'I will open a ticket.\n<tool_call>\n{"name": "create_ticket", "arguments": {"subject": "Refund"}}\n</tool_call>'
    calls = extract_tool_calls(text)
    assert calls[0]["name"] == "create_ticket"
    assert calls[0]["arguments"]["subject"] == "Refund"
    assert "<tool_call>" not in strip_tool_markup(text)


def test_retriever_finds_product_knowledge():
    retriever = Retriever(get_settings())
    hits = retriever.search("forgot password reset code", k=3)
    assert hits
    assert any("password" in hit["text"].lower() or "पासवर्ड" in hit["text"] for hit in hits)


def test_pcm_to_wav_header():
    wav = pcm_to_wav(b"\x00\x00" * 24, rate=24000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_gold_pairs_match_reply_script():
    assert len(PAIRS) >= 70
    eval_count = sum(1 for row in PAIRS if row["split"] == "eval")
    assert eval_count >= 10
    for row in PAIRS:
        assistant = row["messages"][-1]["content"]
        if row["language"] == "ne":
            assert re.search(r"[\u0900-\u097F]", assistant), row["id"]
        else:
            assert not re.search(r"[\u0900-\u097F]", assistant), row["id"]


def test_sft_rate_and_export(tmp_path):
    rec = SFTRecorder(get_settings())
    rec.dir = tmp_path
    rec.path = tmp_path / "sft_pairs.jsonl"
    rec.gold_path = tmp_path / "gold.jsonl"
    rec.train_path = tmp_path / "train.jsonl"
    rec.eval_path = tmp_path / "eval.jsonl"
    pair_id = rec.record("sanchai", "सञ्चै छ।", language="ne", intent="greeting", sources=[], teacher="test")
    assert rec.rate(pair_id, True)
    exported = rec.export()
    assert exported["gold"] == len(PAIRS)
    assert exported["kept"] == 1
    assert exported["train"] + exported["eval"] == exported["gold"] + 1
    train_line = rec.train_path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(train_line)["messages"][0]["role"] == "system"


def test_translit_romanized_nepali():
    from himalaya_support.adapt.translit import classify_input, to_devanagari

    assert classify_input("mero pin birse") == "romanized_nepali"
    result = to_devanagari("mero pin birse")
    assert "पिन" in result.out or "मेरो" in result.out


def test_numeric_grounding_blocks_invented_amount():
    from himalaya_support.adapt.grounding import check_numeric_grounding

    ok, failures = check_numeric_grounding(
        "ब्याज ९९ प्रतिशत हो।",
        "बचत खातामा वार्षिक ५ प्रतिशत ब्याज दिइन्छ।",
        [],
    )
    assert not ok
    assert failures


def test_honorific_gate_rejects_informal():
    from himalaya_support.adapt.honorific import check_honorific

    assert check_honorific("तपाईंले पिन रिसेट गर्नुहोस्", "high")["ok"]
    assert not check_honorific("तिमीले पिन हाल", "high")["ok"]


def test_banking_kb_in_retriever():
    retriever = Retriever(get_settings())
    hits = retriever.search("बचत खाताको ब्याजदर कति हो", k=5)
    assert hits
    assert any("५" in hit["text"] or "प्रतिशत" in hit["text"] for hit in hits)


def test_ticket_confirmation_words():
    from himalaya_support.adapt.actions import parse_confirmation

    assert parse_confirmation("हो") is True
    assert parse_confirmation("होइन") is False
    assert parse_confirmation("पिन बिर्सें") is None


def test_speech_normalize_strips_markdown():
    from himalaya_support.adapt.speech import normalize_for_speech

    spoken = normalize_for_speech("रु. ५०० https://example.com **bold**")
    assert "रुपैयाँ" in spoken
    assert "लिङ्क" in spoken
    assert "**" not in spoken


def test_english_types_to_nepali_unicode():
    from himalaya_support.adapt.to_nepali import latin_to_nepali

    pin = latin_to_nepali("I forgot my PIN")
    assert pin["mode"] == "translate"
    assert "पिन" in pin["out"]
    assert "बिर्स" in pin["out"]
    roman = latin_to_nepali("mero pin birse")
    assert "पिन" in roman["out"] or "मेरो" in roman["out"]


def test_store_splits_chat_and_call(tmp_path):
    from himalaya_support.store.db import SupportStore

    store = SupportStore(tmp_path / "support.db")
    chat_id = store.get_or_create_conversation(None, None, "ne", channel="chat")
    call_id = store.get_or_create_conversation(None, None, "ne", channel="call")
    store.add_message(chat_id, "user", "मेरो पिन बिर्सें")
    store.add_message(call_id, "assistant", "नमस्ते, कल उठ्यो")
    chats = store.list_conversations(channel="chat")
    calls = store.list_conversations(channel="call")
    assert any(row["id"] == chat_id for row in chats)
    assert any(row["id"] == call_id for row in calls)
    assert all(row["id"] != call_id for row in chats)
    messages = store.list_messages(call_id)
    assert messages[0]["content"] == "नमस्ते, कल उठ्यो"




def _fresh_app(monkeypatch, **env):
    """Import the app under a given environment, fresh each time."""
    import importlib
    import sys

    for key in ("SUPPORT_ENV", "SUPPORT_API_KEY", "SUPPORT_CORS_ORIGINS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for mod in [m for m in list(sys.modules) if m.startswith("himalaya_support")]:
        del sys.modules[mod]
    return importlib.import_module("himalaya_support.main")


def test_production_refuses_to_boot_without_a_key(monkeypatch):
    """An unset key left every /v1 route open, silently.

    Starting insecure is worse than not starting: the failure is invisible
    until someone finds the open API.
    """
    import pytest

    with pytest.raises(RuntimeError, match="SUPPORT_API_KEY"):
        _fresh_app(monkeypatch, SUPPORT_ENV="production")

    with pytest.raises(RuntimeError, match="SUPPORT_CORS_ORIGINS"):
        _fresh_app(monkeypatch, SUPPORT_ENV="production",
                   SUPPORT_API_KEY="k", SUPPORT_CORS_ORIGINS="*")

    # development is unaffected, so the local workflow still runs keyless
    assert _fresh_app(monkeypatch, SUPPORT_ENV="development") is not None


def test_api_key_is_enforced_and_reveals_nothing(monkeypatch):
    from fastapi.testclient import TestClient

    main_mod = _fresh_app(
        monkeypatch,
        SUPPORT_ENV="production",
        SUPPORT_API_KEY="k-123",
        SUPPORT_CORS_ORIGINS="https://desk.example.np",
    )
    client = TestClient(main_mod.app)

    assert client.get("/v1/support/tickets").status_code == 401
    assert client.get("/v1/support/tickets", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/v1/support/tickets", headers={"x-api-key": "k-123"}).status_code == 200
    assert client.get("/v1/support/tickets",
                      headers={"authorization": "Bearer k-123"}).status_code == 200

    # the rejection says nothing about why
    assert client.get("/v1/support/tickets").json() == {"detail": "Unauthorized"}
    # liveness stays reachable, and carries no configuration
    health = client.get("/v1/health")
    assert health.status_code == 200
    assert set(health.json()) == {"ok"}


def test_unhandled_errors_do_not_leak_internals(monkeypatch):
    """The handler answered with "{ExceptionType}: {message}", which hands a
    caller the exception class and whatever the provider put in the message."""
    from fastapi.testclient import TestClient

    main_mod = _fresh_app(monkeypatch, SUPPORT_ENV="development")

    @main_mod.app.get("/v1/_boom")
    def _boom():
        raise ValueError("connection string postgres://user:hunter2@db:5432 failed")

    client = TestClient(main_mod.app, raise_server_exceptions=False)
    response = client.get("/v1/_boom")
    assert response.status_code == 500
    body = response.text
    assert body == '{"detail":"Internal server error"}'
    for leak in ("ValueError", "hunter2", "postgres://", "Traceback", "himalaya_support"):
        assert leak not in body
