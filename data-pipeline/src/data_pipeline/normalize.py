from __future__ import annotations

import re
import unicodedata
from typing import cast

# Zero-width characters: ZWSP, ZWNJ, ZWJ, BOM/ZWNBSP
_ZERO_WIDTH = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
}

# Curly/typographic quotes -> straight quotes
_QUOTE_MAP: dict[str, str] = {
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u00ab": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u00bb": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
}

# Em-dash, en-dash, minus sign -> hyphen-minus
_DASH_MAP: dict[str, str] = {
    "\u2014": "-",  # EM DASH
    "\u2013": "-",  # EN DASH
    "\u2212": "-",  # MINUS SIGN
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in _ZERO_WIDTH)
    _translation: dict[str, str | int | None] = cast(
        "dict[str, str | int | None]", {**_QUOTE_MAP, **_DASH_MAP}
    )
    text = text.translate(str.maketrans(_translation))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
