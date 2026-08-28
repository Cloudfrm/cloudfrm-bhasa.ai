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


def test_knowledge_rows_never_leak_qa_scaffolding():
    """Corpus rows are stored as "प्रश्न: ...\nजवाफ: ...".

    Splicing a row in whole showed the corpus's own question back to the
    officer as though Bhasa had asked it, answering a different question and
    marked grounded. Only the answer half may ever be shown.
    """
    from himalaya_support.adapt.pipeline import compose_from_knowledge, split_knowledge_row

    question, answer = split_knowledge_row(
        "प्रश्न: बचत खाताको ब्याजदर कति हो?\nजवाफ: बचत खातामा वार्षिक ५ प्रतिशत ब्याज दिइन्छ।"
    )
    assert question == "बचत खाताको ब्याजदर कति हो?"
    assert answer == "बचत खातामा वार्षिक ५ प्रतिशत ब्याज दिइन्छ।"

    reply = compose_from_knowledge(
        "बचत खाताको ब्याजदर कति हो?",
        [{"text": "प्रश्न: बचत खाताको ब्याजदर कति हो?\nजवाफ: बचत खातामा वार्षिक ५ प्रतिशत ब्याज दिइन्छ।"}],
        "ne",
    )
    for marker in ("हजुर, यसरी पूरा गर्नुहोस्", "प्रश्न:", "जवाफ:"):
        assert marker not in reply, f"leaked template marker: {marker}"
    assert "वार्षिक ५ प्रतिशत" in reply


def test_finish_reply_rejects_template_markers():
    """A reply carrying our scaffolding is a generation failure, not an answer."""
    import pytest

    from himalaya_support.adapt.pipeline import ReplyGenerationError, finish_reply

    # Long enough to survive the thin-reply check and free of digits so the
    # numeric grounding check passes it through to the final guard, which is
    # the shape the live leak actually had.
    leaked = (
        "हजुर, यसरी पूरा गर्नुहोस्। प्रश्न: कार्ड हराएमा के गर्नुपर्छ? "
        "जवाफ: तुरुन्तै ग्राहक सेवामा फोन गरेर कार्ड ब्लक गर्न अनुरोध गर्नुपर्छ। "
        "त्यसपछि शाखामा परिचयपत्र लिएर जानुहोस् र नयाँ कार्डका लागि निवेदन दिनुहोस्।"
    )
    with pytest.raises(ReplyGenerationError):
        finish_reply(leaked, [], "ne", user_message="कार्ड हरायो, के गर्ने?")


def test_mismatched_knowledge_row_is_not_presented_as_the_answer():
    """A row answering a different question must not be stated confidently."""
    from himalaya_support.adapt.pipeline import compose_from_knowledge

    reply = compose_from_knowledge(
        "शाखा कति बजे खुल्छ?",
        [{"text": "प्रश्न: डेबिट कार्डको वार्षिक शुल्क कति हो?\nजवाफ: डेबिट कार्डको वार्षिक शुल्क ५०० रुपैयाँ हो।"}],
        "ne",
    )
    assert "५०० रुपैयाँ" not in reply


def test_devanagari_danda_is_not_part_of_a_token():
    """ऀ-ॿ spans U+0900–U+097F, which includes "।".

    Indexing "बिर्सनुभयो।" with its danda meant it never matched the same
    word written mid-sentence, so "मेरो पिन बिर्सें" missed the PIN-reset
    article entirely.
    """
    from himalaya_support.rag.retriever import tokenize

    assert "।" not in "".join(tokenize("पिन बिर्सनुभयो। दर्ता नम्बर हाल्नुहोस्।"))
    assert set(tokenize("बिर्सें")) == set(tokenize("बिर्सनुभयो।"))


def test_forgotten_pin_retrieves_the_reset_article():
    from himalaya_support.config import get_settings
    from himalaya_support.rag.retriever import Retriever

    hits = Retriever(get_settings()).search("मेरो पिन बिर्सें", k=3)
    assert hits, "no retrieval hits for a forgotten PIN"
    assert "लगइन" in hits[0]["title"] or "पिन" in hits[0]["text"][:200]


def test_behavioural_rows_are_never_shown_as_answers():
    """The greeting row is an instruction to Bhasa, not a fact for the member.

    Shown verbatim it handed the member our own directions: "when the member
    says नमस्ते ... reply respectfully ... offer help with login, transfers".
    It carries none of the प्रश्न:/जवाफ: markers, which is why the first
    cleanup pass missed it entirely.
    """
    from himalaya_support.adapt.pipeline import compose_from_knowledge, is_directive_row

    greeting = (
        "सदस्यले नमस्ते, सञ्चै, sanchai वा hello भने भाषा ग्राहक सेवाबाट "
        "सम्मानपूर्वक जवाफ दिनुहोस्। लगइन, रकम, ऋण, केवाईसी वा कार्डमा सहयोग प्रस्ताब गर्नुहोस्"
    )
    assert is_directive_row(greeting)
    assert is_directive_row("anything", ["greeting"])
    # a genuine procedure written in the imperative is not a directive row
    assert not is_directive_row(
        "सहकारी वा बैंकको मोबाइल एप खोल्नुहोस्, साइन इन > पिन/पासवर्ड बिर्सनुभयो।"
    )

    reply = compose_from_knowledge("नमस्ते", [{"text": greeting, "tags": ["greeting"]}], "ne")
    assert "सम्मानपूर्वक जवाफ दिनुहोस्" not in reply
    assert "सदस्यले" not in reply


def test_finish_reply_never_returns_a_directive_as_the_answer():
    """The property that matters is that a behavioural rule never reaches an
    officer. finish_reply may either replace it or reject it outright —
    both are safe, and which one fires depends on the grounding check — so
    assert the outcome rather than the mechanism.
    """
    from himalaya_support.adapt.pipeline import (
        ReplyGenerationError,
        finish_reply,
        is_directive_row,
    )

    directive = (
        "सदस्यले नमस्ते वा hello भने भाषा ग्राहक सेवाबाट सम्मानपूर्वक जवाफ दिनुहोस्। "
        "लगइन, रकम, ऋण, केवाईसी वा कार्डमा सहयोग प्रस्ताब गर्नुहोस् र विनम्र रहनुहोस्।"
    )
    try:
        reply, _ = finish_reply(directive, [], "ne", user_message="नमस्ते")
    except ReplyGenerationError:
        return  # rejected outright, which is also correct
    assert not is_directive_row(reply)
    assert "सम्मानपूर्वक जवाफ दिनुहोस्" not in reply


def test_identical_text_dedupes_across_different_client_keys():
    """Two clicks whose first response was lost send the same text under two
    different keys. Keying only on the client key cannot catch that."""
    from himalaya_support.api import routes

    routes._idempotency.clear()
    routes._content_keys.clear()

    payload = routes.ChatRequest(message="कार्ड हरायो, के गर्ने?", conversation_id=None)
    content = routes._content_key(payload)

    first, owns_first = routes._claim("key-attempt-1", content)
    assert owns_first
    second, owns_second = routes._claim("key-attempt-2", content)
    assert not owns_second, "a duplicate with a new key must join the first request"
    assert second is first

    # A different message is unaffected.
    other = routes.ChatRequest(message="मेरो पिन बिर्सें", conversation_id=None)
    _, owns_other = routes._claim("key-attempt-3", routes._content_key(other))
    assert owns_other

    routes._idempotency.clear()
    routes._content_keys.clear()
