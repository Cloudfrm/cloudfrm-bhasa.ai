"""E2/E1/E3/E7/E11: prohibited strings can never return; NFKC is never applied."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "himalaya_support" / "static"
TERMS = json.loads((ROOT / "data" / "knowledge" / "terminology.json").read_text(encoding="utf-8"))

USER_FACING = sorted(STATIC.glob("*.html")) + sorted(STATIC.glob("*.js")) + [
    ROOT / "src" / "himalaya_support" / "support" / "engine.py",
    ROOT / "src" / "himalaya_support" / "support" / "credentials.py",
    ROOT / "data" / "knowledge" / "product.json",
]


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_product_name_is_never_the_common_noun():
    for path in USER_FACING:
        text = _text(path)
        for needle in TERMS["product_name"]["prohibited_as_product_name"]:
            assert needle not in text, f"{path.name}: prohibited product-name form {needle!r}"


def test_prohibited_claims_absent():
    for path in USER_FACING:
        text = _text(path)
        for needle in TERMS["prohibited_claims"]:
            assert needle not in text, f"{path.name}: prohibited claim {needle!r}"


def test_nfkc_never_applied():
    src = ROOT / "src"
    for path in list(src.rglob("*.py")) + list(src.rglob("*.js")) + list(src.rglob("*.html")):
        text = _text(path)
        # allow the word in comments/docstrings, never as a call argument
        assert not re.search(r'normalize\(\s*["\']NFKC["\']', text), path
        assert not re.search(r'\.normalize\(\s*["\']NFKC["\']\s*\)', text), path


def test_numeral_and_calendar_rules_recorded():
    assert TERMS["numerals"]["chrome_ne"] == "deva"
    assert TERMS["numerals"]["chrome_en"] == "latn"
    assert TERMS["numerals"]["evidence"] == "as_published"
    assert TERMS["time"]["clock"] == "24h"
    assert TERMS["calendar"]["chrome"] == "gregorian_labelled"
    assert TERMS["normalization"] == {"form": "NFC", "never": "NFKC"}


def test_no_credential_form_in_ui():
    for path in STATIC.glob("*.html"):
        text = _text(path).lower()
        assert 'type="password"' not in text
        assert "api key" not in text and "api_key" not in text and "apikey" not in text, path.name
