"""Download open romanization lexicons into data/knowledge (run from repo root)."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "knowledge"
HEADERS = {"User-Agent": "BhasaSupport/1.0"}


def get_json(url: str):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=45) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    return json.loads(text)


def as_roman(value) -> str:
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item.strip()), "")
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lexicon: dict[str, str] = {}

    maps = get_json(
        "https://raw.githubusercontent.com/realsanjeev/nepali_unicoder/main/src/nepali_unicoder/data/word_maps.json"
    )
    for roman, de in maps.items():
        roman = as_roman(roman)
        de = str(de).strip()
        if roman and de:
            lexicon.setdefault(roman.lower(), de)

    listing = get_json(
        "https://api.github.com/repos/SushilShrestha/NepaliTransliteralDataset/contents/transliterals"
    )
    filled = 0
    for item in listing:
        name = item.get("name") or ""
        url = item.get("download_url")
        if not name.endswith(".json") or not url:
            continue
        try:
            payload = get_json(url)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for de, roman in payload.items():
            roman = as_roman(roman)
            de = str(de or "").strip()
            if not roman or not de:
                continue
            if roman.isascii() and any("\u0900" <= ch <= "\u097F" for ch in de):
                lexicon.setdefault(roman, de)
                filled += 1

    path = OUT / "roman_lexicon.json"
    path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(lexicon)} roman->devanagari entries ({filled} from dataset) to {path}")


if __name__ == "__main__":
    main()
