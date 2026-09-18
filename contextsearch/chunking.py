"""Split page text into chunks that keep their place in the source.

Every chunk is a verbatim slice ``page_text[char_start:char_end]``. That is
what lets a search result point at an exact span on an exact page, and what
the document viewer highlights. Nothing here rewrites text.

Two strategies:

* ``recursive`` (default) splits paragraphs into sentence-like units and packs
  units up to ``chunk_words``. When a chunk overflows it prefers to cut after
  a sentence end rather than mid-sentence, then carries roughly
  ``chunk_overlap_words`` of trailing units into the next chunk.
* ``fixed`` slides a plain window of ``chunk_words`` words with a step of
  ``chunk_words - chunk_overlap_words``. Simple and predictable; useful as a
  baseline in comparisons.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .readers.base import ParsedDocument

if TYPE_CHECKING:
    from .config import Settings


@dataclass
class Chunk:
    page: int
    index: int
    char_start: int
    char_end: int
    text: str
    word_count: int


@dataclass
class _Unit:
    """A run of words that should stay together: a sentence, a heading, a line."""

    start: int  # char offset of first word
    end: int  # char offset after last word
    words: int
    sentence_end: bool


_WORD = re.compile(r"\S+")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"[.!?]+[\"'\)\]]*$")


class Chunker(ABC):
    name: str = ""

    @abstractmethod
    def split(self, text: str, chunk_words: int, overlap_words: int, min_words: int) -> list[tuple[int, int]]:
        """Return (char_start, char_end) spans covering the text in order."""


class FixedChunker(Chunker):
    name = "fixed"

    def split(self, text: str, chunk_words: int, overlap_words: int, min_words: int) -> list[tuple[int, int]]:
        words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
        if not words:
            return []
        step = max(1, chunk_words - overlap_words)
        spans: list[tuple[int, int]] = []
        for i in range(0, len(words), step):
            window = words[i : i + chunk_words]
            spans.append((window[0][0], window[-1][1]))
            if i + chunk_words >= len(words):
                break
        return _merge_short_tail(spans, text, min_words)


class RecursiveChunker(Chunker):
    name = "recursive"

    def split(self, text: str, chunk_words: int, overlap_words: int, min_words: int) -> list[tuple[int, int]]:
        units = _units(text, chunk_words)
        if not units:
            return []
        spans: list[tuple[int, int]] = []
        current: list[_Unit] = []
        current_words = 0

        def close(upto: int | None = None) -> None:
            """Emit the current chunk, optionally only up to unit index ``upto``, and start the next one with the overlap."""
            nonlocal current, current_words
            if not current:
                return
            emit = current if upto is None else current[: upto + 1]
            rest = [] if upto is None else current[upto + 1 :]
            spans.append((emit[0].start, emit[-1].end))
            carry: list[_Unit] = []
            carried = 0
            for unit in reversed(emit):
                if carried + unit.words > overlap_words or len(carry) >= len(emit) - 1:
                    break
                carry.insert(0, unit)
                carried += unit.words
            current = carry + rest
            current_words = sum(u.words for u in current)

        for unit in units:
            if current and current_words + unit.words > chunk_words:
                # Prefer ending on a sentence in the last 40% of the chunk.
                cut = None
                seen = 0
                for i, u in enumerate(current):
                    seen += u.words
                    if u.sentence_end and seen >= 0.6 * current_words:
                        cut = i
                close(cut)
                # The carried overlap may already be full when a single unit is long.
                while current and current_words + unit.words > chunk_words and len(current) > 1:
                    close()
            current.append(unit)
            current_words += unit.words
        close()
        # Overlap-only tails (all carried, nothing new) are dropped by close();
        # here we also merge a small final piece into its predecessor.
        return _merge_short_tail(_dedupe(spans), text, min_words)


def _units(text: str, chunk_words: int) -> list[_Unit]:
    """Sentence-like units that never cross a paragraph break. Long ones are split by words."""
    units: list[_Unit] = []
    pos = 0
    boundaries = [m.end() for m in _PARAGRAPH_BREAK.finditer(text)] + [len(text)]
    for boundary in boundaries:
        paragraph = text[pos:boundary]
        base = pos
        pos = boundary
        words = list(_WORD.finditer(paragraph))
        if not words:
            continue
        unit_start = words[0].start()
        count = 0
        for i, w in enumerate(words):
            count += 1
            token = w.group()
            ends_sentence = bool(_SENTENCE_END.search(token)) and not _looks_like_abbreviation(token)
            newline_next = i + 1 < len(words) and "\n" in paragraph[w.end() : words[i + 1].start()]
            last = i + 1 == len(words)
            if ends_sentence or newline_next or last or count >= chunk_words:
                units.append(_Unit(base + unit_start, base + w.end(), count, ends_sentence or last))
                if not last:
                    unit_start = words[i + 1].start()
                count = 0
    return units


_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "no.",
    "fig.",
    "inc.",
    "ltd.",
    "st.",
}


def _looks_like_abbreviation(token: str) -> bool:
    t = token.lower().strip("\"'()[]")
    if t in _ABBREVIATIONS:
        return True
    # Single letters with a dot ("J. Smith") and dotted initials ("U.S.")
    return bool(re.fullmatch(r"(?:[a-z]\.)+", t))


def _dedupe(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for span in spans:
        if out and span[0] >= out[-1][0] and span[1] <= out[-1][1]:
            continue  # fully inside the previous chunk, adds nothing
        out.append(span)
    return out


def _merge_short_tail(spans: list[tuple[int, int]], text: str, min_words: int) -> list[tuple[int, int]]:
    if len(spans) >= 2:
        last = spans[-1]
        if len(text[last[0] : last[1]].split()) < min_words:
            prev = spans[-2]
            spans = [*spans[:-2], (prev[0], max(prev[1], last[1]))]
    return spans


CHUNKERS: dict[str, Chunker] = {"recursive": RecursiveChunker(), "fixed": FixedChunker()}


def get_chunker(name: str) -> Chunker:
    try:
        return CHUNKERS[name]
    except KeyError:
        raise ValueError(f"unknown chunker {name!r}; choose from {sorted(CHUNKERS)}") from None


def chunk_document(doc: ParsedDocument, settings: Settings) -> list[Chunk]:
    chunker = get_chunker(settings.chunker)
    chunks: list[Chunk] = []
    for page in doc.pages:
        for start, end in chunker.split(
            page.text, settings.chunk_words, settings.chunk_overlap_words, settings.chunk_min_words
        ):
            text = page.text[start:end]
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    page=page.number,
                    index=len(chunks),
                    char_start=start,
                    char_end=end,
                    text=text,
                    word_count=len(text.split()),
                )
            )
    return chunks
