"""Grounded generative answer path (the model is mocked; the guards are real)."""
import pytest

from himalaya_support.config import get_settings
from himalaya_support.inference.client import ChatResult, InferenceError
from himalaya_support.support.engine import SupportEngine
from himalaya_support.support.refusals import RefusalStrings

GENERAL, QUANTITY = "GENERAL-REFUSAL-PLACEHOLDER", "QUANTITY-REFUSAL-PLACEHOLDER"


class FakeClient:
    def __init__(self, text=None, error=False):
        self.text, self.error, self.calls = text, error, []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        if self.error:
            raise InferenceError("model down")
        return ChatResult(self.text, "fake-gemma", "fake", {})

    def probe(self):
        return {"backend": "fake", "model": "fake-gemma", "reachable": not self.error, "detail": "test"}


def make(tmp_path, monkeypatch, client, path="generative_grounded"):
    monkeypatch.setenv("SUPPORT_ANSWER_PATH", path)
    settings = get_settings()
    monkeypatch.setattr(type(settings), "db_path", property(lambda self: tmp_path / "g.db"))
    assert settings.answer_path == path
    return SupportEngine(settings, refusals=RefusalStrings(GENERAL, QUANTITY, "test", "now"), client=client)


def test_model_writes_reply_and_passage_is_provenance(tmp_path, monkeypatch):
    client = FakeClient("बचत खातामा वार्षिक ५ प्रतिशत ब्याज दिइन्छ, त्यसैले हरेक वर्ष ५ प्रतिशत पाउनुहुन्छ।")
    r = make(tmp_path, monkeypatch, client).chat("बचत खाताको ब्याजदर कति हो?")
    assert r["kind"] == "answer" and r["generated"] is True and r["backend"] == "fake"
    assert r["reply"] == client.text
    assert r["passage"]["id"] == "kb-001" and "५ प्रतिशत" in r["passage"]["text"]
    # the model only ever sees the retrieved passage
    assert "५ प्रतिशत" in client.calls[0][0]["content"]


def test_ungrounded_figure_becomes_quantity_refusal(tmp_path, monkeypatch):
    client = FakeClient("बचत खातामा वार्षिक ७ प्रतिशत ब्याज दिइन्छ।")  # 7 is not in the document
    r = make(tmp_path, monkeypatch, client).chat("बचत खाताको ब्याजदर कति हो?")
    assert r["kind"] == "refusal" and r["refusal_type"] == "quantity"
    assert r["reply"].encode("utf-8") == QUANTITY.encode("utf-8")
    assert r["generated"] is False and r["note"].startswith("ungrounded_quantity")


def test_no_evidence_means_refusal_and_model_is_not_called(tmp_path, monkeypatch):
    client = FakeClient("काठमाडौं नेपालको राजधानी हो।")
    r = make(tmp_path, monkeypatch, client).chat("नेपालको राजधानी कुन हो?")
    assert r["kind"] == "refusal" and r["refusal_type"] == "general"
    assert r["reply"].encode("utf-8") == GENERAL.encode("utf-8")
    assert client.calls == []  # never lets the model freewheel without evidence


def test_model_unreachable_falls_back_to_verbatim_passage(tmp_path, monkeypatch):
    client = FakeClient(error=True)
    engine = make(tmp_path, monkeypatch, client)
    r = engine.chat("बचत खाताको ब्याजदर कति हो?")
    assert r["kind"] == "answer" and r["generated"] is False
    assert r["backend"] == "extractive_fallback" and r["note"] == "llm_unreachable"
    assert r["reply"] == r["passage"]["text"]
    doc = engine.retriever.document(r["passage"]["id"])
    assert r["reply"] in doc.text


def test_thin_model_reply_falls_back(tmp_path, monkeypatch):
    r = make(tmp_path, monkeypatch, FakeClient("हजुर।")).chat("बचत खाताको ब्याजदर कति हो?")
    assert r["generated"] is False and r["backend"] == "extractive_fallback" and r["note"] == "model_reply_thin"


def test_credential_never_reaches_the_model(tmp_path, monkeypatch):
    client = FakeClient("ok")
    r = make(tmp_path, monkeypatch, client).chat("my password is Sunita@2081 and login fails")
    assert r["kind"] == "credential_decline" and client.calls == []


def test_capabilities_declare_generative_and_the_difference(tmp_path, monkeypatch):
    caps = make(tmp_path, monkeypatch, FakeClient("x")).capabilities()
    assert caps["answer_path"] == "generative_grounded"
    assert "differs from the deployed product" in caps["answer_path_note"]
    assert caps["llm"]["reachable"] is True and caps["llm"]["model"] == "fake-gemma"


def test_extractive_mode_never_calls_the_model(tmp_path, monkeypatch):
    client = FakeClient("should not be used")
    engine = make(tmp_path, monkeypatch, client, path="extractive")
    r = engine.chat("बचत खाताको ब्याजदर कति हो?")
    assert client.calls == [] and r["generated"] is False and r["backend"] == "extractive"
    assert engine.capabilities()["answer_path"] == "extractive"
