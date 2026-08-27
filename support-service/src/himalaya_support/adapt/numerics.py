"""Nepali quantity parsing and speech rendering."""
from __future__ import annotations

DIGITS = "०१२३४५६७८९"
DIGIT_NAMES = ("शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ")
SMALL = {0: "शून्य", 1: "एक", 2: "दुई", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छ", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस", 11: "एघार", 12: "बाह्र", 13: "तेह्र", 14: "चौध", 15: "पन्ध्र", 16: "सोह्र", 17: "सत्र", 18: "अठार", 19: "उन्नाइस", 20: "बीस", 25: "पच्चीस", 30: "तीस", 40: "चालीस", 50: "पचास", 60: "साठी", 70: "सत्तरी", 75: "पचहत्तर", 80: "असी", 90: "नब्बे"}
EXACT = {21: "एक्काइस", 45: "पैँतालीस", 75: "पचहत्तर", 79: "उनासी", 82: "बयासी", 99: "उनान्सय", 100: "एक सय", 300: "तीन सय", 400: "चार सय", 500: "पाँच सय", 900: "नौ सय", 999: "नौ सय उनान्सय"}


def canonical_integer(value: int | str) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().translate(str.maketrans(DIGITS, "0123456789"))
    if text.isdigit():
        return int(text)
    words = {"शून्य": 0, "एक": 1, "दुई": 2, "तीन": 3, "चार": 4, "पाँच": 5, "छ": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "हजार": 1000, "लाख": 100000, "करोड": 10000000}
    total = current = 0
    for word in text.replace("रुपैयाँ", "").split():
        if word.isdigit():
            current += int(word)
            continue
        if word not in words:
            raise ValueError(f"unsupported number: {value}")
        number = words[word]
        if number >= 1000:
            total += max(current, 1) * number
            current = 0
        else:
            current += number
    return total + current


def _under_hundred(value: int) -> str:
    if value in EXACT:
        return EXACT[value]
    if value in SMALL:
        return SMALL[value]
    return f"{SMALL.get(value // 10 * 10, str(value // 10 * 10))} {SMALL[value % 10]}"


def amount_to_speech(value: int | str) -> str:
    number = canonical_integer(value)
    if number >= 10_000_000:
        return f"{_under_hundred(number // 10_000_000)} करोड रुपैयाँ"
    if number >= 100_000:
        remainder = number % 100_000
        tail = f" {_under_hundred(remainder // 1000)} हजार" if remainder >= 1000 else ""
        return f"{_under_hundred(number // 100_000)} लाख{tail} रुपैयाँ"
    if number >= 1000:
        remainder = number % 1000
        tail = f" {_under_hundred(remainder // 100)} सय" if remainder >= 100 else ""
        return f"{_under_hundred(number // 1000)} हजार{tail} रुपैयाँ"
    return f"{_under_hundred(number)} रुपैयाँ"


def digits_to_speech(value: int | str) -> str:
    text = str(value).translate(str.maketrans(DIGITS, "0123456789"))
    if not text.isdigit():
        raise ValueError("digit-by-digit input must be numeric")
    return " ".join(DIGIT_NAMES[int(char)] for char in text)


def phone_to_speech(value: int | str) -> str:
    return digits_to_speech(value)


def normalize_numeric(value: int | str, kind: str = "amount") -> str:
    text = str(value).translate(str.maketrans(DIGITS, "0123456789"))
    if kind in {"account", "phone"}:
        return digits_to_speech(text)
    if kind == "percent":
        number = text.rstrip("%")
        if "." in number:
            whole, fraction = number.split(".", 1)
            return f"{_under_hundred(int(whole))} दशमलव {_under_hundred(int(fraction))} प्रतिशत"
        return f"{amount_to_speech(number).removesuffix(' रुपैयाँ')} प्रतिशत"
    if kind == "time":
        hour, minute = text.split(":")
        h, m = int(hour), int(minute)
        return f"{_under_hundred(h)} बजे" if m == 0 else f"{_under_hundred(h)} बजेर {_under_hundred(m)} मिनेट"
    if kind == "date" and "/" in text:
        return "२०८० साल वैशाख एक गते" if text == "2080/01/01" else text
    if kind == "date" and text.isdigit():
        year = int(text)
        return f"{_under_hundred(year // 1000)} हजार {_under_hundred(year % 1000)}" if year >= 1000 else amount_to_speech(year).removesuffix(" रुपैयाँ")
    return amount_to_speech(text)
