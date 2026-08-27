"""E8/E9: word-run conversion, protected tokens, ranked candidates."""
import json
import pathlib

from himalaya_support.adapt.candidates import (
    convert_message,
    decide_run,
    detect_question_language,
    protected_reason,
    segment,
    word_candidates,
)

# input, expected detected type of the FIRST latin-ish run (or "none"), expectation on final text
PROOF_TABLE = [
    ("mero pin birse", "romanized_nepali", {"converted": True}),
    ("khata ma paisa chaina", "romanized_nepali", {"converted": True}),
    ("byajdar kati ho", "romanized_nepali", {"converted": True}),
    ("I forgot my password", "english", {"unchanged": True}),
    ("My transfer did not arrive", "english", {"unchanged": True}),
    ("I love you", "english", {"unchanged": True}),
    ("mero KYC update garnu paryo", "mixed", {"keeps": ["KYC"], "converted": True}),
    ("NIMB ko ATM ma card atkiyo", "mixed", {"keeps": ["NIMB", "ATM"], "converted": True}),
    ("OTP aayena", "mixed", {"keeps": ["OTP"], "converted": True}),
    ("SWIFT code chahiyo", "mixed", {"keeps": ["SWIFT"], "converted": True}),
    ("USD 500 pathaune", "mixed", {"keeps": ["USD", "500"], "converted": True}),
    ("mero number 9841234567 ho", "mixed", {"keeps": ["9841234567"], "converted": True}),
    ("email test@example.com ma pathaunu", "mixed", {"keeps": ["test@example.com"], "converted": True}),
    ("https://nimb.com.np kholna sakina", "mixed", {"keeps": ["https://nimb.com.np"], "converted": True}),
    ("मेरो pin birse", "mixed", {"keeps": ["मेरो"], "converted": True}),
    ("kista 12 din dhilo bhayo", "mixed", {"keeps": ["12"], "converted": True}),
    ("EMI kahile tirne", "mixed", {"keeps": ["EMI"], "converted": True}),
    ("QR scan garda error aayo", "mixed", {"keeps": ["QR", "error"], "converted": True}),
    # English sentences that contain banking loanwords must stay English (no Nepali context → no auto-convert)
    ("my mobile banking app crashed", "loanwords_only", {"unchanged": True}),
    ("How do I reset my mobile banking PIN?", "loanwords_only", {"unchanged": True, "keeps": ["PIN"]}),
    ("please block my card", "loanwords_only", {"unchanged": True}),
]


def _kind(text: str) -> str:
    runs = [r for r in segment(text) if r.kind in {"latin", "protected", "devanagari", "digits", "url", "email"}]
    latin = [r for r in runs if r.kind == "latin"]
    if latin and all(r.parses for r in latin) and len(latin) == len(runs):
        return "romanized_nepali"
    if latin and not any(r.parses for r in latin) and len(latin) == len(runs):
        return "english"
    if latin and not any(r.auto for r in latin):
        return "loanwords_only"  # English sentence with banking loanwords: nothing auto-converts
    return "mixed"


def test_proof_table():
    rows = []
    for text, expected_kind, expect in PROOF_TABLE:
        result = convert_message(text)
        out = result["text"]
        kind = _kind(text)
        assert kind == expected_kind, (text, kind)
        if expect.get("unchanged"):
            assert out == text, (text, out)
        if expect.get("converted"):
            assert out != text, (text, out)
        for keep in expect.get("keeps", []):
            assert keep in out, (text, keep, out)
        cands = {r["text"]: [c["text"] for c in r["candidates"]] for r in result["runs"] if r["candidates"]}
        rows.append({"input": text, "type": kind, "candidates": cands, "sent": out})
    # Leave a machine-readable copy of the proof table for the report.
    out_path = pathlib.Path(__file__).resolve().parent / "_proof_table.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def test_protected_tokens_never_convert():
    for token in ["NIMB", "SCB", "ATM", "KYC", "PIN", "OTP", "USD", "NPR", "SWIFT", "EMI", "IPO", "CIB", "QR", "SMS"]:
        assert protected_reason(token) in {"protected_term", "all_caps"}, token
        assert decide_run(token).candidates == []
    assert protected_reason("ABC") == "all_caps"
    assert protected_reason("a1b") == "digits"
    assert protected_reason("http://x.y") == "url"
    assert protected_reason("a@b.co") == "email"
    assert protected_reason("खाता") == "devanagari"
    assert protected_reason("khata") is None


def test_lowercase_pin_is_a_loanword_but_caps_is_protected():
    assert decide_run("pin").parses
    assert not decide_run("PIN").parses


def test_candidates_ranked_and_capped():
    cands = word_candidates("khata")
    assert cands
    assert cands[0]["text"] == "खाता"
    assert cands[0]["source"] in {"lexicon", "banking"}
    assert len(cands) <= 5
    assert len({c["text"] for c in cands}) == len(cands)
    # banking terminology beats phonetic guesses
    byaj = word_candidates("byajdar")
    assert byaj[0]["text"] == "ब्याजदर"


def test_session_choice_ranks_first_next_time():
    base = convert_message("mero khata")["text"]
    chosen = convert_message("mero khata", {"khata": "खता"})["text"]
    assert base != chosen and "खता" in chosen


def test_language_detection_ignores_protected_tokens():
    assert detect_question_language("mero KYC update garnu paryo") == "ne"
    assert detect_question_language("How do I update my KYC?") == "en"
    assert detect_question_language("NIMB ATM OTP") == "ne"  # nothing but protected tokens: default ne
    assert detect_question_language("मेरो पिन बिर्सें") == "ne"
    assert detect_question_language("I sent NPR 5000 but it did not arrive") == "en"
