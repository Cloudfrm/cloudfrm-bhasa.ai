"""E21: credential-shaped input is detected before send, never echoed, never stored raw."""
import pytest

from himalaya_support.config import get_settings
from himalaya_support.support.credentials import DECLINE, REDACTION, check_credentials
from himalaya_support.support.engine import SupportEngine
from himalaya_support.support.refusals import RefusalStrings

# The two traps from the generic test set: a pasted password, and a card number with CVV.
TRAP_PASSWORD = "my password is Sunita@2081 and login still fails"
TRAP_CARD_CVV = "card 4111 1111 1111 1111 cvv 123 block garnu paryo"


def test_trap_pasted_password():
    check = check_credentials(TRAP_PASSWORD)
    assert check.detected and "password" in check.kinds
    assert "Sunita@2081" not in check.redacted
    assert REDACTION in check.redacted


def test_trap_card_with_cvv():
    check = check_credentials(TRAP_CARD_CVV)
    assert check.detected
    assert {"card", "cvv"} <= set(check.kinds)
    assert "4111" not in check.redacted and "123" not in check.redacted


def test_otp_and_pin_shapes():
    assert check_credentials("otp 482913 aayo").kinds == ["pin_otp"]
    assert check_credentials("मेरो पिन १२३४ हो").kinds == ["pin_otp"]
    assert check_credentials("482913").detected  # bare 6 digits = OTP-shaped


def test_quantities_are_not_credentials():
    for text in [
        "बचत खातामा वार्षिक ५ प्रतिशत ब्याज दिइन्छ",
        "I sent NPR 5000 yesterday",
        "loan 250000 rupees",
        "30 मिनेट लक हुन्छ",
        "2026-08-28",
        "kista 12 din dhilo bhayo",
    ]:
        assert not check_credentials(text).detected, text


@pytest.fixture
def engine(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(type(settings), "db_path", property(lambda self: tmp_path / "t.db"))
    fake = RefusalStrings("G", "Q", "test", "now")  # placeholders; not the real strings
    return SupportEngine(settings, refusals=fake)


def test_engine_declines_and_stores_redacted(engine):
    for trap in (TRAP_PASSWORD, TRAP_CARD_CVV):
        result = engine.chat(trap, locale="auto")
        assert result["kind"] == "credential_decline"
        assert result["reply"] in DECLINE.values()
        assert "Sunita@2081" not in result["echo"] and "4111" not in result["echo"]
        stored = engine.store.list_messages(result["conversation_id"])
        assert all("Sunita@2081" not in m["content"] and "4111 1111" not in m["content"] for m in stored)
        assert stored[0]["meta"].get("redacted")


def test_engine_never_logs_raw_credential(engine, caplog):
    caplog.set_level("DEBUG")
    engine.chat(TRAP_PASSWORD)
    assert "Sunita@2081" not in caplog.text
