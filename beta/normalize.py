"""Text normalization applied identically to human and LLM reviews.

The point is to remove differences that come from *how the data was produced*
(crawler vs. API output) rather than *who wrote it*, so the classifier can't
win by spotting markdown, curly quotes, or line breaks.
"""
from __future__ import annotations

import re
import unicodedata

_MARKDOWN = [
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),
    (re.compile(r"__(.+?)__", re.S), r"\1"),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M), ""),
    (re.compile(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", re.M), ""),
]
_CHAR_MAP = str.maketrans(
    {
        "“": '"', "”": '"', "„": '"',
        "‘": "'", "’": "'", "‚": "'",
        "…": "...",
        " ": " ",
        "～": "~", "〜": "~",
        "！": "!", "？": "?", "，": ",", "．": ".",
    }
)
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
_WS = re.compile(r"\s+")
# Ablation: keep only Hangul syllables, Latin letters, digits and spaces.
# Drops punctuation, emoji, and bare jamo such as ㅋ ㅎ ㅠ.
_NON_CORE = re.compile(r"[^0-9A-Za-z가-힣 ]")


def normalize(text: object) -> str:
    if not isinstance(text, str):
        return ""
    # NFC, not NFKC: NFKC rewrites compatibility jamo (ㅋ U+314B) into
    # conjoining jamo (U+110F), which silently changes "ㅋㅋ".
    t = unicodedata.normalize("NFC", text)
    t = _ZERO_WIDTH.sub("", t)
    t = t.translate(_CHAR_MAP)
    for pattern, repl in _MARKDOWN:
        t = pattern.sub(repl, t)
    return _WS.sub(" ", t).strip()


def strip_surface(text: object) -> str:
    """Ablation view: no punctuation, emoji or jamo."""
    t = _NON_CORE.sub(" ", normalize(text))
    return _WS.sub(" ", t).strip()


def strip_surface_nospace(text: object) -> str:
    """Ablation view that also removes spacing (LLMs space perfectly, people don't)."""
    return strip_surface(text).replace(" ", "")


_FOREIGN_SCRIPT = re.compile(r"[一-鿿぀-ヿ]")  # Han, Hiragana, Katakana
_LATIN = re.compile(r"[A-Za-z]")
_LETTER = re.compile(r"[A-Za-z가-힣]")


def is_korean_text(text: object, max_latin_ratio: float = 0.3, min_latin: int = 8) -> bool:
    """False for Chinese/Japanese characters or text that is largely English.

    Some models (Qwen 7B in the pilot) drift into Chinese or English mid-review.
    Applied to BOTH classes at build time so it cannot become a label shortcut.
    Short acronyms ("BGM 맛집", "TV에 나온") stay: English only counts once there
    are at least `min_latin` Latin letters making up over `max_latin_ratio` of letters.
    """
    if not isinstance(text, str) or _FOREIGN_SCRIPT.search(text):
        return False
    letters = len(_LETTER.findall(text))
    latin = len(_LATIN.findall(text))
    return not (latin >= min_latin and latin / max(letters, 1) > max_latin_ratio)


def near_dup_key(text: object) -> str:
    return strip_surface_nospace(text)


def bin_labels(edges: list[int]) -> list[str]:
    labels = [f"<={edges[0]}"]
    labels += [f"{lo + 1}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">{edges[-1]}")
    return labels


def length_bin(n: int, edges: list[int]) -> str:
    labels = bin_labels(edges)
    for i, edge in enumerate(edges):
        if n <= edge:
            return labels[i]
    return labels[-1]
