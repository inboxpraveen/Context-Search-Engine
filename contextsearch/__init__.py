"""contextsearch: hybrid document search you can run yourself.

Keyword (BM25) and semantic (embedding) retrieval over PDFs, Word and
PowerPoint files, text and HTML, fused into one ranked list, with the
evidence for every result: page, character span, matched terms, per-retriever
scores and stage timings.

Quick start::

    from contextsearch import Engine

    with Engine.open("./data") as engine:
        engine.add_folder("/path/to/documents")[1].wait()
        for hit in engine.search("late delivery refund").hits:
            print(hit.confidence_label, hit.document.relpath, hit.chunk.page, hit.snippet)
"""

__version__ = "3.0.0"

from .config import Settings
from .engine import Engine
from .errors import ContextSearchError
from .retrieval import Hit, SearchRequest, SearchResponse

__all__ = ["ContextSearchError", "Engine", "Hit", "SearchRequest", "SearchResponse", "Settings", "__version__"]
