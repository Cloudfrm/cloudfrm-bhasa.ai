"""F2 -- numeric normalization for ingestion and re-verification.

Every quantity extracted from a source document is stored BOTH raw and
normalized, and every comparison (the D2 independent second read, the
supersession test's cross-version checks) happens on the normalized value,
never on the string: १०,००,००० must verify against 1000000.

The numeral grammar itself is NOT defined here. `app.grounding` (the gate)
and `app.numerics` are the single source of truth for digit alphabets and
unit-word values; this module imports them and adds only what ingestion
needs on top: decimal support (grounding's grammar is integer-only because
spoken amounts are), currency/percent token detection, and Latin aliases
for the unit words (rate sheets write "10 lakh" as often as "१० लाख").

Values are Decimal end to end. 8.5 vs 8.50 must compare equal and 3.77
must not pick up binary-float noise on its way through the store.

NFC always; NFKC never -- ingestion text is evidence, and evidence is
stored in the canonical composed form of what the document actually says,
not a compatibility folding of it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from himalaya_support.adapt.grounding import UNITS
from himalaya_support.adapt.numerics import DIGITS, canonical_integer

_DEVA_TO_ASCII = str.maketrans(DIGITS, "0123456789")

# Latin spellings of the same unit words. Values deliberately come from the
# imported UNITS map -- if the gate's grammar ever changes, this follows.
_LATIN_UNIT_ALIASES: dict[str, str] = {
    "lakh": "लाख", "lac": "लाख", "lakhs": "लाख",
    "crore": "करोड", "crores": "करोड", "karod": "करोड",
    "hajar": "हजार", "thousand": "हजार",
}

_CURRENCY_TOKENS = re.compile(
    r"(?:\bNPR\b|\bRs\.?\b|रु\.?|रू\.?|रुपैयाँ|रुपैया)", re.IGNORECASE)
_PERCENT_TOKENS = re.compile(r"%|प्रतिशत|\bP\.?A\.?\b|\bp\.a\.\b", re.IGNORECASE)

# One number, optional Indian/Western comma grouping, optional decimal part.
_NUMBER = re.compile(r"^\d{1,3}(?:,\d{2,3})*(?:\.\d+)?$|^\d+(?:\.\d+)?$")
_NUMBER_WITH_UNIT = re.compile(
    r"^(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(सय|हजार|लाख|करोड)$")


@dataclass(frozen=True)
class NormalizedQuantity:
    raw: str
    value: Decimal
    currency: str | None  # "NPR" or None
    unit: str             # "percent" | "amount" | "number"

    def as_export(self) -> dict:
        return {"raw": self.raw, "normalized_value": str(self.value),
                "currency": self.currency, "unit": self.unit}


def nfc(text: str) -> str:
    """Canonical composition only. NFKC is deliberately never used on
    ingested evidence."""
    return unicodedata.normalize("NFC", text or "")


def normalize_quantity(raw: str) -> NormalizedQuantity | None:
    """Parse ONE quantity out of a string that should contain exactly one.

    Returns None when the string does not reduce to a single unambiguous
    quantity -- the caller flags that for human review rather than guessing.
    """
    if raw is None:
        return None
    original = raw
    text = nfc(str(raw)).strip()
    if not text:
        return None

    currency = "NPR" if _CURRENCY_TOKENS.search(text) else None
    text = _CURRENCY_TOKENS.sub(" ", text)
    is_percent = bool(_PERCENT_TOKENS.search(text))
    text = _PERCENT_TOKENS.sub(" ", text)

    text = text.translate(_DEVA_TO_ASCII)
    # Token removal can leave orphan punctuation at the edges ("P.A." keeps
    # its final dot when the \b lands before it). Edge punctuation is never
    # part of a quantity; interior dots (decimals) are untouched.
    text = text.strip(" .,|;:")
    words = text.split()
    words = [_LATIN_UNIT_ALIASES.get(w.lower(), w) for w in words]
    text = " ".join(words).strip()
    if not text:
        return None

    unit = "percent" if is_percent else ("amount" if currency or any(
        w in UNITS for w in words) else "number")

    value = _parse_value(text)
    if value is None:
        return None
    return NormalizedQuantity(raw=nfc(str(original)).strip(), value=value,
                              currency=currency, unit=unit)


def _parse_value(text: str) -> Decimal | None:
    compact = text.replace(" ", "")
    if _NUMBER.match(compact):
        try:
            return Decimal(compact.replace(",", ""))
        except InvalidOperation:
            return None

    m = _NUMBER_WITH_UNIT.match(text) or _NUMBER_WITH_UNIT.match(
        re.sub(r"\s+", " ", text))
    if m:
        try:
            base = Decimal(m.group(1).replace(",", ""))
        except InvalidOperation:
            return None
        return base * UNITS[m.group(2)]

    # Word forms ("दस लाख", "10 लाख 50 हजार"): delegate to the frozen
    # integer grammar. It raises on anything it does not recognise.
    try:
        return Decimal(canonical_integer(text))
    except (ValueError, KeyError):
        return None


def same_quantity(a: NormalizedQuantity | Decimal | None,
                  b: NormalizedQuantity | Decimal | None) -> bool:
    """Equality on normalized values. 8.5 == 8.50; १०,००,००० == 1000000."""
    if a is None or b is None:
        return False
    va = a.value if isinstance(a, NormalizedQuantity) else a
    vb = b.value if isinstance(b, NormalizedQuantity) else b
    return va == vb
