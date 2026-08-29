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


def test_question_type_disagreement_is_detected():
    """Asking "when" and asking "what happens" are different questions.

    Word overlap alone ranked the late-penalty row above the due-date row for
    "ऋण किस्ता कहिले तिर्ने?" because both repeat किस्ता.
    """
    from himalaya_support.adapt.pipeline import asks_a_different_question, question_kinds

    assert question_kinds("ऋण किस्ता कहिले तिर्ने?") == {"when"}
    assert question_kinds("कर्जाको किस्ता तिर्न ढिलो भएमा के हुन्छ?") == {"what"}
    # के must match whole-word only: केवाईसी is not a "what" question
    assert "what" not in question_kinds("केवाईसी कसरी गर्ने?")
    assert question_kinds("केवाईसी कसरी गर्ने?") == {"how"}

    assert asks_a_different_question(
        "कर्जाको किस्ता तिर्न ढिलो भएमा के हुन्छ?", "ऋण किस्ता कहिले तिर्ने?"
    )
    # same question type -> not demoted
    assert not asks_a_different_question("कार्ड हराएमा के गर्नुपर्छ?", "कार्ड हरायो, के गर्ने?")
    # a prose row states no question, and a statement asks none: never demote
    assert not asks_a_different_question("", "ऋण किस्ता कहिले तिर्ने?")
    assert not asks_a_different_question("कार्ड हराएमा के गर्नुपर्छ?", "मेरो पिन बिर्सें")


def test_each_loan_question_gets_its_own_answer():
    """The regression this re-ranking exists for, end to end."""
    from himalaya_support.adapt.pipeline import finish_reply
    from himalaya_support.config import get_settings
    from himalaya_support.support.engine import SupportEngine

    engine = SupportEngine(get_settings())

    def answer(query: str) -> str:
        reply, _ = finish_reply("", engine._pick_snippet(query, "ne"), "ne", user_message=query)
        return reply

    due = answer("ऋण किस्ता कहिले तिर्ने?")
    assert "मितिमा" in due, f"expected the due-date row, got: {due[:120]}"

    late = answer("कर्जाको किस्ता तिर्न ढिलो भएमा के हुन्छ?")
    assert "जरिवाना" in late, f"expected the penalty row, got: {late[:120]}"


def test_conversations_past_the_first_page_are_reachable(tmp_path):
    """The limit used to be the whole story.

    list_conversations capped at 50 with no way to ask for the next page, so
    on a desk with more than 50 conversations the oldest silently vanished
    from the inbox — present in the database, unreachable in the UI.
    """
    from himalaya_support.store.db import SupportStore

    store = SupportStore(tmp_path / "support.db")
    made = []
    for _ in range(55):
        made.append(store.get_or_create_conversation(None, None, "ne", channel="chat"))

    assert store.count_conversations(channel="chat") == 55

    first = store.list_conversations(channel="chat", limit=50, offset=0)
    assert len(first) == 50
    second = store.list_conversations(channel="chat", limit=50, offset=50)
    assert len(second) == 5, "the tail must be reachable, not truncated away"

    seen = {row["id"] for row in first} | {row["id"] for row in second}
    assert seen == set(made), "every conversation must appear across the pages"
    assert not ({row["id"] for row in first} & {row["id"] for row in second}), "pages must not overlap"


def test_conversation_count_is_independent_of_page_size(tmp_path):
    """The badge counts the desk, not the rows that fitted in one response."""
    from himalaya_support.store.db import SupportStore

    store = SupportStore(tmp_path / "support.db")
    for _ in range(12):
        store.get_or_create_conversation(None, None, "ne", channel="chat")
    store.get_or_create_conversation(None, None, "ne", channel="call")

    assert len(store.list_conversations(channel="chat", limit=5)) == 5
    assert store.count_conversations(channel="chat") == 12
    assert store.count_conversations(channel="call") == 1
    assert store.count_conversations() == 13


def test_open_ticket_count_is_filtered_in_sql_not_after_truncation(tmp_path):
    """The Dashboard tiles had the same bug as the sidebar badge.

    They read .length on an unparameterized (so page-capped) response, which
    plateaus at the page size. Worse for tickets: "open" was filtered on the
    client, i.e. after truncation, so the count could be wrong even when few
    tickets were open.
    """
    from himalaya_support.store.db import SupportStore

    store = SupportStore(tmp_path / "support.db")
    for i in range(60):
        store.create_ticket({"subject": f"open {i}"})
    resolved = [store.create_ticket({"subject": f"done {i}"}) for i in range(10)]
    for ticket in resolved:
        store.update_ticket(ticket["id"], status="resolved")

    assert store.count_tickets() == 70
    assert store.count_tickets(status="open") == 60
    assert store.count_tickets(status="resolved") == 10

    # A page is a page, but the count is the truth.
    page = store.list_tickets(status="open", limit=8)
    assert len(page) == 8
    assert all(row["status"] != "resolved" for row in page)

    # Filtering after truncation is what used to go wrong: take the first 50
    # of everything and count the open ones, and you do not get 60.
    naive = [r for r in store.list_tickets(limit=50) if r["status"] != "resolved"]
    assert len(naive) != store.count_tickets(status="open")

    assert len(store.list_tickets(status="open", limit=100)) == 60
    assert len(store.list_tickets(status="open", limit=50, offset=50)) == 10


def test_export_streams_every_message_with_filters(tmp_path):
    """Every chat and call was already stored; there was just no way out."""
    from himalaya_support.store.db import SupportStore

    store = SupportStore(tmp_path / "support.db")
    chat_id = store.get_or_create_conversation(None, None, "ne", channel="chat")
    call_id = store.get_or_create_conversation(None, None, "ne", channel="call")
    store.add_message(chat_id, "user", "मेरो पिन बिर्सें")
    store.add_message(chat_id, "assistant", "पिन रिसेट गर्नुहोस्।")
    store.add_message(call_id, "user", "कार्ड हरायो")

    assert store.count_export() == {"messages": 3, "conversations": 2}
    assert store.count_export(channel="chat")["messages"] == 2
    assert store.count_export(channel="call")["conversations"] == 1

    rows = list(store.iter_export_rows(channel="chat"))
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert {r["channel"] for r in rows} == {"chat"}
    assert rows[0]["content"] == "मेरो पिन बिर्सें"
    # every column a training pipeline needs
    for key in ("conversation_id", "channel", "locale", "role", "content", "created_at"):
        assert key in rows[0]

    later = rows[1]["created_at"]
    assert all(r["created_at"] >= later for r in store.iter_export_rows(since=later))


def test_export_rows_survive_line_per_record_serialisation(tmp_path):
    """JSONL is line-per-record, so a newline inside content must be escaped
    rather than splitting one message across two lines."""
    import json

    from himalaya_support.store.db import SupportStore

    store = SupportStore(tmp_path / "support.db")
    cid = store.get_or_create_conversation(None, None, "ne", channel="chat")
    hostile = 'पहिलो\nदोस्रो, अल्पविराम\t"उद्धरण"'
    store.add_message(cid, "user", hostile)

    row = next(iter(store.iter_export_rows()))
    line = json.dumps(row, ensure_ascii=False)
    assert "\n" not in line, "an embedded newline would break line-per-record"
    assert json.loads(line)["content"] == hostile
    assert "पहिलो" in line, "Devanagari should stay readable, not backslash-u escaped"


def test_multiline_message_keeps_its_line_breaks():
    """The record of what a member said should look like what they typed.

    normalize() collapses every run of whitespace, which is right for a
    retrieval key — but that same flattened text was being stored as the
    message, so a pasted list of transaction numbers became one line.
    """
    from himalaya_support.adapt.devanagari import normalize
    from himalaya_support.adapt.pipeline import prepare_user_text

    raw = "मेरो कारोबार पुगेन। विवरण:\nTXN-1001\nTXN-1002\n\n\n\nकृपया हेर्नुहोस्।"

    # the retrieval key stays flat, deliberately
    assert "\n" not in normalize(raw)

    kept = normalize(raw, keep_lines=True)
    assert kept.count("\n") >= 3
    assert "\r" not in kept
    assert "\n\n\n" not in kept, "runaway blank lines should collapse to one"

    prepared = prepare_user_text(raw)
    assert "\n" in prepared["display"], "the stored form must keep line breaks"
    assert "\n" not in prepared["search"], "the retrieval key must stay one line"
    assert [l for l in prepared["display"].split("\n") if l.strip()].__len__() == 4


def test_reference_codes_are_not_transliterated():
    """TXN-1001 became ट्क्ष्ण-1001, and 24x7 became 24क्ष7.

    A token that fuses letters and digits is a reference, not romanized
    Nepali — and it is exactly the detail a member pastes when a transfer
    fails. Letter-only words must still convert.
    """
    from himalaya_support.adapt.to_nepali import latin_to_nepali

    assert latin_to_nepali("TXN-1001")["out"] == "TXN-1001"
    assert "TXN-1001" in latin_to_nepali("मेरो कारोबार TXN-1001 पुगेन")["out"]
    assert "24x7" in latin_to_nepali("ATM 24x7")["out"]

    # unchanged: letter-only words are still translated/transliterated
    assert "पिन" in latin_to_nepali("I forgot my PIN")["out"]
    roman = latin_to_nepali("mero pin birse")["out"]
    assert "पिन" in roman or "मेरो" in roman


def test_training_corpora_never_ground_an_answer():
    """The retrieval index holds fine-tuning slices alongside knowledge.

    A message that is essentially just an amount matches no banking article,
    which let a function-calling training row win — returning a tool
    definition named initiateDispute to a member asking about a transfer.
    """
    from himalaya_support.adapt.pipeline import compose_from_knowledge, is_training_artifact

    tools_row = 'Tools: [{"type": "function", "function": {"name": "initiateDispute"}}]'
    schema_row = 'Schema: {"properties": {"totalShipments": {"type": "integer"}}}'

    assert is_training_artifact(tools_row)
    assert is_training_artifact(schema_row)
    # recognised by source too, for a dataset shaped differently
    assert is_training_artifact("anything", "himalaya-ai/nepali-hermes-function-calling-v1")
    assert is_training_artifact("anything", "himalaya-ai/nepali-json-mode-singleturn")
    # real knowledge is not an artifact
    assert not is_training_artifact("बचत खातामा वार्षिक ५ प्रतिशत ब्याज दिइन्छ।", "product_knowledge")

    reply = compose_from_knowledge(
        "NPR 5,000",
        [{"text": tools_row, "source": "himalaya-ai/nepali-hermes-function-calling-v1"}],
        "ne",
    )
    assert "initiateDispute" not in reply
    assert "Tools:" not in reply


def test_finish_reply_never_returns_training_data():
    from himalaya_support.adapt.pipeline import (
        ReplyGenerationError,
        finish_reply,
        is_training_artifact,
    )

    leaked = (
        'Schema: {"properties": {"totalShipments": {"title": "Total Shipments", '
        '"type": "integer"}}, "title": "LogisticsDashboard", "type": "object"}'
    )
    try:
        reply, _ = finish_reply(leaked, [], "ne", user_message="NPR 5,000")
    except ReplyGenerationError:
        return  # rejected outright is also correct
    assert not is_training_artifact(reply)


def test_reference_formats_survive_the_composer_path():
    """The guard has to hold in unicoder(), not just latin_to_nepali().

    The composer converts through /v1/support/unicoder before posting, and
    that function treated leftover Latin as "conversion unfinished" and
    force-converted the raw text — walking straight past the guard, because a
    preserved TXN-1001 is itself Latin.
    """
    from himalaya_support.adapt.to_nepali import unicoder

    for text, must_survive in [
        ("TXN-1001 pugena", "TXN-1001"),
        ("NARBNPKA", "NARBNPKA"),            # SWIFT/BIC, letters only
        ("NIBLNPKT", "NIBLNPKT"),
        ("ram.thapa@example.com", "ram.thapa@example.com"),
        ("https://bank.com.np/help", "https://bank.com.np/help"),
        ("NPR 5,000", "NPR"),                # currency code
        ("ATM 24x7", "24x7"),
    ]:
        assert must_survive in unicoder(text)["nepali"], f"{must_survive} lost from {text!r}"

    # and the carve-out still converts
    assert "पिन" in unicoder("I forgot my PIN")["nepali"]
    assert "पिन" in unicoder("mero pin birse")["nepali"]


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


# --- crisis handling -------------------------------------------------------

def test_distress_is_detected_in_all_three_scripts():
    """A member typed "I want to die." and got the banking menu.

    Detection runs on the raw message because transliteration destroys the
    signal: "I am going to kill myself" became "म हुँ गोइङ किल्ल ंय्सेल्फ".
    """
    from himalaya_support.adapt.crisis import looks_like_crisis

    for text in [
        "I want to die.",
        "म मर्न चाहन्छु",
        "I am going to kill myself",
        "I have no reason to live, my loan is too much",
        "मलाई मर्न मन छ",
        "marna manchu",
        "I don't want to live any more",
    ]:
        assert looks_like_crisis(text), f"missed: {text!r}"

    # ordinary banking language must not trip it
    for text in [
        "my card is dead",
        "the app killed my session",
        "मेरो पिन बिर्सें",
        "ऋण किस्ता कहिले तिर्ने?",
        "I want to close my account",
        "kill the pending transfer",
    ]:
        assert not looks_like_crisis(text), f"false positive: {text!r}"


def test_crisis_reply_names_no_helpline_until_one_is_verified():
    """No number is invented. Until someone confirms the real ones, the reply
    acknowledges the member and says a person will follow up, and that is all.
    """
    import re

    from himalaya_support.adapt.crisis import crisis_reply, load_crisis_config

    config = load_crisis_config(None)  # nothing configured
    for language in ("ne", "en"):
        reply = crisis_reply(language, config)
        assert reply.strip()
        assert not re.search(r"\d{3,}", reply), "a phone number appeared from nowhere"
        for financial in ("ऋण", "loan", "interest", "ब्याज", "जरिवाना"):
            assert financial not in reply

    # once verified, the resource is shown
    configured = {
        "resources": [{"name": "Example Service", "contact": "0000000", "hours": "24 hours"}],
        "message": config["message"],
    }
    assert "Example Service" in crisis_reply("en", configured)


def test_crisis_turn_bypasses_retrieval_and_flags_a_person(tmp_path):
    from himalaya_support.config import get_settings
    from himalaya_support.store.db import SupportStore
    from himalaya_support.support.engine import SupportEngine

    engine = SupportEngine(get_settings())
    engine.store = SupportStore(tmp_path / "scratch.db")

    out = engine.chat("I have no reason to live, my loan is too much",
                      locale="ne", proofread=False)

    assert out["crisis"] is True
    assert out["retrieved"] == []
    assert out["intent"]["needs_human"] is True
    assert out["tickets"], "a wellbeing ticket must be raised"
    for financial in ("ऋण", "किस्ता", "जरिवाना", "ब्याज"):
        assert financial not in out["reply"], "financial content in a crisis turn"

    ticket = engine.store.get_ticket(out["tickets"][0])
    assert ticket["category"] == "wellbeing"
    assert ticket["priority"] == "urgent"


# --- small talk ------------------------------------------------------------

def test_courtesies_are_answered_not_searched():
    from himalaya_support.adapt.smalltalk import classify

    expected = {
        "Hello": "greeting", "Hi": "greeting", "Namaste": "greeting",
        "नमस्ते": "greeting", "Good morning": "greeting",
        "Thank you": "thanks", "Thanks": "thanks", "धन्यवाद": "thanks",
        "Thank you so much": "thanks",
        "Sorry": "apology", "माफ गर्नुहोस्": "apology",
        "Excuse me": "attention",
        "Bye": "farewell", "Take care": "farewell",
        "Well done": "praise", "thumbs up": "praise",
        "Good luck": "well_wish",
        "ok": "affirm", "yes": "affirm", "हुन्छ": "affirm",
        "no": "decline", "no thanks": "decline",
    }
    for text, category in expected.items():
        assert classify(text) == category, f"{text!r} classified as {classify(text)!r}"


def test_smalltalk_does_not_swallow_a_real_question():
    """A greeting plus a banking question is a banking question."""
    from himalaya_support.adapt.smalltalk import classify

    for text in [
        "Hello, my PIN is forgotten, what do I do?",
        "नमस्ते, ऋण किस्ता कहिले तिर्ने?",
        "thanks, what is the savings interest rate?",
        "ok so when does the branch open",
        "no money arrived in my account",
    ]:
        assert classify(text) is None, f"over-matched: {text!r}"


def test_bare_affirmation_never_reaches_retrieval(tmp_path):
    """"ok" returned OTP validity rules and "yes" returned a loan interest
    rate — while the fallback had just invited the member to say yes."""
    from himalaya_support.config import get_settings
    from himalaya_support.store.db import SupportStore
    from himalaya_support.support.engine import SupportEngine

    engine = SupportEngine(get_settings())
    engine.store = SupportStore(tmp_path / "scratch.db")

    for text, forbidden in [("ok", "ओटिपी"), ("yes", "ब्याजदर"), ("हुन्छ", "ब्याजदर")]:
        out = engine.chat(text, locale="ne", proofread=False)
        assert out["pipeline"] == "smalltalk"
        assert out["retrieved"] == []
        assert forbidden not in out["reply"]

    # but a pending ticket offer still wins
    convo = engine.store.get_or_create_conversation(None, None, "ne", channel="chat")
    engine._pending_ticket[convo] = {
        "conversation_id": convo, "user_id": None, "subject": "s",
        "description": "d", "category": "other", "priority": "normal",
    }
    out = engine.chat("yes", conversation_id=convo, locale="ne", proofread=False)
    assert out["tickets"], "a pending offer must still be confirmable"


def test_crisis_recall_survives_nepali_inflection():
    """The first phrase list was written English-first and it showed.

    "मलाई बाँच्ने इच्छा छैन" was caught while "मलाई बाँच्नु मन छैन" was not —
    one sentence, a different verbal noun and a different word for wanting.
    Every case below is a Nepali or indirect form that the phrase-only
    matcher missed.
    """
    from himalaya_support.adapt.crisis import looks_like_crisis

    missed_before = [
        "मलाई बाँच्ने कुनै कारण छैन, मेरो ऋण धेरै भयो",
        "मलाई बाँच्नु मन छैन",
        "मेरो जीवन अर्थहीन छ, सबै समाप्त गर्न चाहन्छु",
        "honestly I would rather not be alive anymore",
        "malai bachnu man chaina",
        "बाँच्न मन लाग्दैन",
        "मेरो जीवनको कुनै अर्थ छैन",
        "म सबैको लागि बोझ भएँ",
        "nothing left to live for",
        "I am tired of living",
        "everyone would be better off without me",
    ]
    for text in missed_before:
        assert looks_like_crisis(text), text


def test_crisis_detection_does_not_fire_on_money_talk():
    """Widening recall must not turn ordinary banking into a crisis.

    The traps that matter are the ones a cooperative member really writes:
    life insurance, life savings, and the many ways of saying "I cannot pay".
    """
    from himalaya_support.adapt.crisis import looks_like_crisis

    ordinary = [
        "my card is dead and the app killed my session",
        "I am dying to know my balance",
        "kill the pending transfer",
        "I don't want to lose my life savings",
        "जीवन बीमा को बारेमा जान्न चाहन्छु",
        "मेरो जीवन बीमाको किस्ता कति हो?",
        "म यो ऋण तिर्न सक्दिनँ",
        "किस्ता तिर्न मन छैन",
        "काम समाप्त गर्न चाहन्छु",
        "मेरो हजुरबुबा मर्नुभयो, खाता बन्द गर्नुपर्‍यो",
        "I can't go on with this loan application",
        "I do not want to live in Kathmandu anymore, closing my account",
    ]
    for text in ordinary:
        assert not looks_like_crisis(text), text


def test_confirmation_understands_the_language_the_offer_was_made_in():
    """The copy says "say yes to open a ticket" and this knew only Devanagari.

    It must also refuse to read a decision out of a message that merely
    contains one: "ok so my card is blocked" is not consent.
    """
    from himalaya_support.adapt.actions import parse_confirmation

    for text in ("yes", "Yes.", "yeah", "sure", "yes please",
                 "yes please open a ticket", "हो", "हुन्छ", "ठीक छ", "hunchha"):
        assert parse_confirmation(text) is True, text
    # "ok" acknowledges the answer; it does not consent to a ticket.
    assert parse_confirmation("ok") is None
    for text in ("no", "no thanks", "होइन", "पर्दैन", "no do not open a ticket"):
        assert parse_confirmation(text) is False, text
    for text in ("ok so my card is blocked and I need help",
                 "I have not received my money yet",
                 "पिन बिर्सें",
                 "correct the name on my account please"):
        assert parse_confirmation(text) is None, text


def test_offering_a_ticket_is_what_arms_the_yes(tmp_path):
    """The offer and the state that answers it are one decision.

    The reply has always ended "say yes and I will open a ticket", but only
    an intent flagged needs_ticket ever armed the pending state, so most of
    the time the app promised a ticket and then had no memory of promising
    it. Both locales, because the Nepali branch never made the offer at all.
    """
    from himalaya_support.config import get_settings
    from himalaya_support.store.db import SupportStore
    from himalaya_support.support.engine import SupportEngine

    engine = SupportEngine(get_settings())
    engine.store = SupportStore(tmp_path / "scratch.db")

    for locale, affirm in (("en", "yes"), ("ne", "हुन्छ")):
        first = engine.chat("what do I need to open an account?", locale=locale)
        assert first["pending_confirm"] == "create_ticket", locale
        second = engine.chat(affirm, conversation_id=first["conversation_id"], locale=locale)
        assert second["conversation_id"] == first["conversation_id"]
        assert len(second["tickets"]) == 1, f"{locale}: saying yes must open the ticket"


def test_offers_ticket_matches_every_way_the_desk_offers_one():
    from himalaya_support.adapt.pipeline import GROUNDING_FALLBACK, offers_ticket

    assert offers_ticket("यति गर्दा पनि नखुले शाखामा जानुहोस्, वा टिकट खोलौं हो?")
    assert offers_ticket("If that still fails, or say yes to open a ticket. Do not type PIN.")
    assert offers_ticket("Add one more sentence, or say yes and I will open a ticket.")
    assert offers_ticket("Open a ticket? Say yes or no.")
    assert not offers_ticket(GROUNDING_FALLBACK)
    assert not offers_ticket("नमस्ते! म कसरी सहयोग गर्न सक्छु?")


def test_multi_word_courtesies_are_matched_as_phrases():
    """"Have a nice day" and "Nice to meet you" fell through a bag-of-words
    classifier: adding their words to a category would have made "nice" plus
    any stray word match, and "good day" — a greeting — read as a well-wish."""
    from himalaya_support.adapt.smalltalk import MEETING, WELL_WISH, classify

    assert classify("Nice to meet you") == MEETING
    assert classify("nice to meet you!") == MEETING
    assert classify("Have a nice day") == WELL_WISH
    assert classify("have a good evening") == WELL_WISH
    # and the phrase must not swallow a real question attached to it
    assert classify("Have a nice day, but first when does the branch open?") is None


def test_courtesy_table_can_be_completed_without_a_deploy(tmp_path):
    """The built-in list started at twenty and the desk's table is longer.

    Reporting the gap one string at a time is nobody's good use of an
    afternoon, so the rest of the table goes in a config file. A bad category
    in that file must be ignored, not take the desk down.
    """
    import json

    from himalaya_support.adapt import smalltalk
    from himalaya_support.adapt.smalltalk import FAREWELL, PRAISE, classify

    knowledge = tmp_path / "product.json"
    knowledge.write_text("{}", encoding="utf-8")
    (tmp_path / "courtesy_phrases.json").write_text(json.dumps({
        "phrases": {"tata for now": "farewell", "bad one": "not_a_category"},
        "words": {"shabash": "praise"},
    }), encoding="utf-8")
    smalltalk._loaded.clear()

    assert classify("tata for now", knowledge) == FAREWELL
    assert classify("shabash", knowledge) == PRAISE
    assert classify("bad one", knowledge) is None      # unknown category ignored
    assert classify("Hello", knowledge) is not None    # and the desk still works
    smalltalk._loaded.clear()


def test_courtesies_reported_after_the_phrase_fix():
    """"Good day" is the pointed one: the reasoning for matching whole
    phrases used it as the example of what bag-of-words would get wrong, and
    then left it matching nothing at all. "How do you do?" was worse than
    falling through — it returned statement-download instructions."""
    from himalaya_support.adapt.smalltalk import classify

    for text in ("Good day", "Lovely day", "Bless you", "All the best",
                 "Safe travels", "Bravo", "How do you do?",
                 "Pleased to meet you"):
        assert classify(text) is not None, text
    # and none of them may swallow a real question
    assert classify("Good day, when does the branch open?") is None
    assert classify("how do you calculate the interest rate?") is None


def test_health_names_the_corpus_outside_production(monkeypatch, tmp_path):
    """Two instances that look identical in a browser, one of which is the
    real desk, is how test traffic ends up in the real desk."""
    from fastapi.testclient import TestClient

    scratch = tmp_path / "scratch" / "support.db"
    main_mod = _fresh_app(monkeypatch, SUPPORT_DB_PATH=str(scratch))
    body = TestClient(main_mod.app).get("/v1/health").json()
    assert body["store"] == "scratch"

    monkeypatch.delenv("SUPPORT_DB_PATH", raising=False)
    plain = _fresh_app(monkeypatch)
    assert TestClient(plain.app).get("/v1/health").json()["store"] == "desk"


def test_scratch_db_path_moves_every_write_off_the_desk(monkeypatch, tmp_path):
    """The standing instruction was to test against a scratch database, and
    there was no way to comply with it from a browser. This is that way."""
    from himalaya_support.config import get_settings

    scratch = tmp_path / "elsewhere" / "support.db"
    monkeypatch.setenv("SUPPORT_DB_PATH", str(scratch))
    settings = get_settings()
    assert settings.db_path == scratch
    assert settings.is_scratch_store is True

    monkeypatch.delenv("SUPPORT_DB_PATH")
    assert get_settings().is_scratch_store is False


def test_second_sample_from_the_courtesy_table():
    """Bon voyage, Sounds good, Got it, See you later, Fingers crossed,
    Stunning, My pleasure, No worries — the next eight reported after the
    first sample was fixed."""
    from himalaya_support.adapt.smalltalk import classify

    for text in ("Bon voyage", "Sounds good", "Got it", "See you later",
                 "Fingers crossed", "Stunning", "My pleasure", "No worries"):
        assert classify(text) is not None, text


def test_acknowledgements_do_not_swallow_real_messages():
    """"got" and "worries" turn up in real messages, so only the whole phrase
    may match: "I got charged twice" is a dispute, not an acknowledgement."""
    from himalaya_support.adapt.smalltalk import classify

    for text in ("I got charged twice on my card",
                 "I am worried about my balance",
                 "got it wrong, my transfer failed",
                 "no worries about the fee, but when does it post?",
                 "sounds good, what is the interest rate?",
                 "see you later, but first how do I block my card?"):
        assert classify(text) is None, text


def test_the_reply_quotes_what_the_member_typed(tmp_path):
    """A question typed in English is transliterated to search a Nepali
    corpus — which measurably helps retrieval — and the fallback then quoted
    that search key back at the member:

        "how do you calculate the interest rate?"
        → I understood: "कसरी दो तपाईं चल्चुलते ब्याजदर?"

    Round 15 fixed this for the stored message and missed the copy inside the
    reply. Romanized Nepali still gets the Devanagari back, because producing
    it is what the member asked the desk for.
    """
    from himalaya_support.config import get_settings
    from himalaya_support.store.db import SupportStore
    from himalaya_support.support.engine import SupportEngine

    engine = SupportEngine(get_settings())
    engine.store = SupportStore(tmp_path / "scratch.db")

    typed = "how do you calculate the interest rate?"
    reply = engine.chat(typed, locale="en")["reply"]
    if "I understood" in reply:
        assert typed in reply, reply
    assert "चल्चुलते" not in reply, "the transliterated search key reached the member"


def test_compose_from_knowledge_separates_the_key_from_the_quote():
    from himalaya_support.adapt.pipeline import compose_from_knowledge

    out = compose_from_knowledge(
        "कसरी दो तपाईं चल्चुलते ब्याजदर?", [], "en",
        echo="how do you calculate the interest rate?",
    )
    assert "how do you calculate the interest rate?" in out
    assert "चल्चुलते" not in out
    # with no echo given, behaviour is unchanged
    assert "abc" in compose_from_knowledge("abc", [], "en")
