"""The answer path is extractive: verbatim passage or a fetched refusal string."""
import unicodedata

import pytest

from himalaya_support.config import get_settings
from himalaya_support.support.engine import SupportEngine
from himalaya_support.support.extractive import is_quantity_question
from himalaya_support.support.refusals import fetch_refusals


@pytest.fixture(scope="module")
def engine():
    settings = get_settings()
    refusals = fetch_refusals(settings.proof_url, settings.refusal_cache_path)
    if refusals is None:
        pytest.skip("refusal strings unavailable (no network and no cache) — cannot test refusals honestly")
    return SupportEngine(settings, refusals=refusals)


def test_proof_case_a_quantity_in_corpus(engine):
    a = engine.answerer.answer("व्यक्तिगत कर्जाको ब्याजदर कति हो?", "ne")
    assert a.kind == "answer"
    assert a.passage is not None
    # verbatim: the reply is a substring of the source document, byte-for-byte after NFC
    doc = engine.retriever.document(a.passage.doc_id)
    assert unicodedata.normalize("NFC", a.reply) in unicodedata.normalize("NFC", doc.text)
    assert "१३" in a.reply


def test_proof_case_b_quantity_not_in_corpus(engine):
    a = engine.answerer.answer("व्यवसायिक कर्जाको प्रशोधन शुल्क कति हो?", "ne")
    assert a.kind == "refusal"
    assert a.refusal_type == "quantity"
    assert engine.refusals.classify(a.reply) == "quantity"
    assert a.reply.encode("utf-8") == engine.refusals.quantity.encode("utf-8")


def test_proof_case_c_outside_corpus(engine):
    a = engine.answerer.answer("नेपालको राजधानी कुन हो?", "ne")
    assert a.kind == "refusal"
    assert a.refusal_type == "general"
    assert a.reply.encode("utf-8") == engine.refusals.general.encode("utf-8")


def test_every_answer_is_verbatim(engine):
    questions = [
        "बचत खाताको ब्याजदर कति हो?", "कार्ड हरायो, के गर्ने?", "मेरो पिन बिर्सें",
        "How do I reset my mobile banking PIN?", "When is the loan instalment due?",
        "ऋण किस्ता कहिले तिर्ने?", "एटिएमबाट दैनिक कति रुपैयाँसम्म झिक्न सकिन्छ?",
    ]
    for q in questions:
        lang = "en" if q.isascii() else "ne"
        a = engine.answerer.answer(q, lang)
        assert a.kind in {"answer", "refusal"}
        if a.kind == "answer":
            doc = engine.retriever.document(a.passage.doc_id)
            assert unicodedata.normalize("NFC", a.reply) in unicodedata.normalize("NFC", doc.text), q
            assert a.passage.language == lang, q  # never translated


def test_wrong_language_refuses_instead_of_translating(engine):
    # The savings rate exists only in a Nepali document; an English question must not get a translation.
    a = engine.answerer.answer("What is the savings interest rate?", "en")
    assert a.kind == "refusal"
    assert all(c["reason"] != "selected" for c in a.considered)


def test_training_data_rows_are_not_quotable(engine):
    sources = {doc.source for doc in engine.retriever.documents}
    assert "product_knowledge" in sources
    assert not any("function_calling" in s or "json_mode" in s or "honorific" in s for s in sources)


def test_bare_rakam_is_not_a_quantity_question():
    assert not is_quantity_question("रकम पठाएँ तर पुगेन")
    assert is_quantity_question("बचत खाताको ब्याजदर कति हो?")
    assert is_quantity_question("What is the daily ATM limit?")


def test_chips_are_all_answerable(engine):
    chips = engine.chips()
    assert chips["ne"] and chips["en"]
    for lang, questions in chips.items():
        for q in questions:
            assert engine.answerer.answer(q, lang).kind == "answer", q
    assert "रकम" not in "".join(chips["ne"]) or all(
        engine.answerer.answer(q, "ne").kind == "answer" for q in chips["ne"]
    )


def test_refusal_strings_are_never_retyped_in_source(engine):
    """Byte-exact refusal strings live only in the fetched proof document."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    needles = [engine.refusals.general, engine.refusals.quantity]
    for path in list(root.rglob("*.py")) + list(root.rglob("*.html")) + list(root.rglob("*.js")):
        text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
        for needle in needles:
            assert needle not in text, f"refusal string retyped in {path}"
