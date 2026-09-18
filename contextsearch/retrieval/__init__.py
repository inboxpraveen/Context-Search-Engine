"""Retrieval: keyword (FTS5/BM25), semantic (embeddings), fusion and reranking."""

from .dense import DenseRetriever
from .fusion import FusedCandidate, fuse, rrf, weighted
from .lexical import LexicalRetriever, ParsedQuery, parse_query
from .pipeline import Hit, Searcher, SearchRequest, SearchResponse, build_snippet, confidence
from .types import RetrieverHit, RetrieverResult

__all__ = [
    "DenseRetriever",
    "FusedCandidate",
    "Hit",
    "LexicalRetriever",
    "ParsedQuery",
    "RetrieverHit",
    "RetrieverResult",
    "SearchRequest",
    "SearchResponse",
    "Searcher",
    "build_snippet",
    "confidence",
    "fuse",
    "parse_query",
    "rrf",
    "weighted",
]
