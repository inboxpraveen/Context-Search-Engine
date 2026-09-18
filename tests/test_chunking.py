"""Chunks must be verbatim slices with exact offsets, cover all text and end on sentences where possible."""

from __future__ import annotations

import random

from contextsearch.chunking import FixedChunker, RecursiveChunker, chunk_document
from contextsearch.readers import Page, ParsedDocument
from tests.conftest import make_settings


def _prose(seed: int = 1, paragraphs: int = 6) -> str:
    rng = random.Random(seed)
    words = "the roaster cools the beans while the packing line waits for the next batch of bags".split()
    out = []
    for _ in range(paragraphs):
        sentences = []
        for _ in range(rng.randint(3, 8)):
            n = rng.randint(4, 22)
            sentences.append(
                " ".join(rng.choice(words) for _ in range(n)).capitalize() + rng.choice([".", ".", "!", "?"])
            )
        # hard wrap every 12 words, like text extracted from a PDF
        tokens = " ".join(sentences).split()
        out.append("\n".join(" ".join(tokens[i : i + 12]) for i in range(0, len(tokens), 12)))
    return "\n\n".join(out)


def _check_coverage(text: str, spans: list[tuple[int, int]]) -> None:
    covered = set()
    last_start = -1
    for start, end in spans:
        assert 0 <= start < end <= len(text)
        assert start > last_start, "spans must move forward"
        assert text[start:end].strip() == text[start:end], "spans must not start or end on whitespace"
        covered.update(range(start, end))
        last_start = start
    missing = [i for i, c in enumerate(text) if not c.isspace() and i not in covered]
    assert not missing, f"{len(missing)} characters are not covered by any chunk"


def test_recursive_chunks_cover_text_and_end_on_sentences():
    text = _prose()
    spans = RecursiveChunker().split(text, chunk_words=60, overlap_words=10, min_words=15)
    _check_coverage(text, spans)
    assert len(spans) > 3
    for start, end in spans:
        assert text[end - 1] in ".!?", f"chunk ending {text[end - 20 : end]!r} is not a sentence end"
        assert len(text[start:end].split()) <= 60 + 15  # target plus a merged short tail


def test_recursive_overlap_carries_trailing_sentence():
    text = _prose(seed=3)
    spans = RecursiveChunker().split(text, chunk_words=60, overlap_words=12, min_words=15)
    overlaps = [spans[i][1] - spans[i + 1][0] for i in range(len(spans) - 1)]
    assert any(o > 0 for o in overlaps), "consecutive chunks should share text"


def test_fixed_chunks_have_exact_word_windows():
    text = " ".join(f"w{i}" for i in range(250))
    spans = FixedChunker().split(text, chunk_words=100, overlap_words=20, min_words=10)
    _check_coverage(text, spans)
    assert [len(text[s:e].split()) for s, e in spans] == [100, 100, 90]
    assert text[spans[1][0] :].startswith("w80 "), "second window starts 20 words before the end of the first"


def test_short_tail_is_merged_into_previous_chunk():
    text = " ".join(f"w{i}" for i in range(105))
    spans = FixedChunker().split(text, chunk_words=100, overlap_words=0, min_words=40)
    assert len(spans) == 1
    assert spans[0] == (0, len(text))


def test_long_sentence_is_split_by_words():
    text = " ".join(f"w{i}" for i in range(500))  # no punctuation at all
    spans = RecursiveChunker().split(text, chunk_words=100, overlap_words=10, min_words=20)
    _check_coverage(text, spans)
    assert all(len(text[s:e].split()) <= 120 for s, e in spans)


def test_chunk_document_keeps_page_numbers_and_offsets(tmp_path):
    settings = make_settings(tmp_path, chunk_words=30, chunk_overlap_words=5, chunk_min_words=5)
    doc = ParsedDocument(pages=[Page(1, _prose(seed=5, paragraphs=2)), Page(2, ""), Page(3, "One short page.")])
    chunks = chunk_document(doc, settings)
    assert {c.page for c in chunks} == {1, 3}
    for chunk in chunks:
        page = next(p for p in doc.pages if p.number == chunk.page)
        assert page.text[chunk.char_start : chunk.char_end] == chunk.text
    assert [c.index for c in chunks] == list(range(len(chunks)))
