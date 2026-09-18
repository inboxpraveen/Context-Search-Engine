"""Keyword retrieval on SQLite FTS5 with BM25 ranking.

The user's query is turned into a safe FTS5 expression: quoted phrases stay
phrases, every other word becomes a quoted term, and common English words are
dropped (they still live in the index, so a quoted phrase can match them).
The porter tokenizer stems both the index and the query, so "returns" finds
"return" and "returned".

The expression is first run with AND between terms. If that finds fewer rows
than asked for, the OR form runs as well and its extra rows are appended,
so precise matches always come first and partial matches fill the rest.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from ..store import HL_CLOSE, HL_OPEN
from .types import RetrieverHit, RetrieverResult

if TYPE_CHECKING:
    from ..store import Store

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)

_PHRASE = re.compile(r'"([^"]+)"')
_TOKEN = re.compile(r"\w+", re.UNICODE)
_MARKED = re.compile(f"{HL_OPEN}(.+?){HL_CLOSE}")


class ParsedQuery:
    def __init__(self, phrases: list[str], terms: list[str], dropped: list[str]):
        self.phrases = phrases
        self.terms = terms
        self.dropped = dropped

    @property
    def units(self) -> list[str]:
        return self.phrases + self.terms

    @property
    def empty(self) -> bool:
        return not self.units

    def expression(self, operator: str = "AND", prefix_last: bool = False) -> str:
        parts = [f'"{_escape(p)}"' for p in self.phrases]
        terms = [f'"{_escape(t)}"' for t in self.terms]
        if prefix_last and terms:
            terms[-1] += "*"
        parts += terms
        return f" {operator} ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"phrases": self.phrases, "terms": self.terms, "dropped": self.dropped}


def parse_query(query: str, drop_stopwords: bool = True) -> ParsedQuery:
    phrases = [p.strip() for p in _PHRASE.findall(query) if p.strip()]
    rest = _PHRASE.sub(" ", query)
    tokens = [t.lower() for t in _TOKEN.findall(rest)]
    dropped: list[str] = []
    terms: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in seen:
            continue
        if drop_stopwords and tok in STOPWORDS:
            dropped.append(tok)
            continue
        seen.add(tok)
        terms.append(tok)
    if not terms and not phrases and dropped:
        # A query made only of stopwords ("to be or not to be") still deserves an answer.
        terms = list(dict.fromkeys(dropped))
        dropped = []
    return ParsedQuery(phrases, terms, dropped)


def _escape(text: str) -> str:
    return text.replace('"', '""')


def matched_terms(highlighted: str) -> list[str]:
    """Distinct surface forms FTS5 marked in a highlighted chunk, lowercased, in order."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _MARKED.finditer(highlighted):
        word = m.group(1).lower()
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def covered_terms(parsed: ParsedQuery, surface_forms: list[str], text: str) -> list[str]:
    """Which query terms and phrases the chunk matched.

    FTS5 marks the surface forms it matched ("refunds" for the query term
    "refund"), so each form is assigned to the query term it shares the longest
    prefix with. Phrases count when they occur verbatim in the chunk.
    """
    covered: list[str] = []
    lowered = text.lower()
    for phrase in parsed.phrases:
        if phrase.lower() in lowered:
            covered.append(phrase)
    for term in parsed.terms:
        for form in surface_forms:
            shared = 0
            for a, b in zip(term, form, strict=False):
                if a != b:
                    break
                shared += 1
            if shared >= min(len(term), len(form), 3) and shared >= min(len(term), len(form)) - 3:
                covered.append(term)
                break
    return covered


class LexicalRetriever:
    name = "lexical"
    kind = "lexical"

    def __init__(self, store: Store, drop_stopwords: bool = True):
        self.store = store
        self.drop_stopwords = drop_stopwords

    def search(
        self,
        query: str,
        limit: int,
        prefix_last: bool = False,
        source_id: str | None = None,
        ext: str | None = None,
        document_id: str | None = None,
    ) -> RetrieverResult:
        started = time.perf_counter()
        parsed = parse_query(query, self.drop_stopwords)
        details: dict[str, Any] = {"query": parsed.to_dict()}
        if parsed.empty:
            return RetrieverResult(self.name, self.kind, [], (time.perf_counter() - started) * 1000, details)

        filters = {"source_id": source_id, "ext": ext, "document_id": document_id}
        expression = parsed.expression("AND", prefix_last)
        rows = self.store.search_fts(expression, limit, **filters)
        details["expression"] = expression
        details["and_matches"] = len(rows)
        if len(rows) < limit and len(parsed.units) > 1:
            seen = {r[0] for r in rows}
            extra = [
                r
                for r in self.store.search_fts(parsed.expression("OR", prefix_last), limit, **filters)
                if r[0] not in seen
            ]
            rows += extra[: limit - len(rows)]
            details["or_fallback"] = True
            details["or_matches"] = len(extra)
        hits = [
            RetrieverHit(chunk_id=cid, score=score, rank=i + 1, highlight=hl) for i, (cid, score, hl) in enumerate(rows)
        ]
        return RetrieverResult(self.name, self.kind, hits, (time.perf_counter() - started) * 1000, details)
