"""Embedding models behind one small interface.

An :class:`Embedder` turns texts into unit-length float32 vectors, so cosine
similarity is a plain dot product everywhere else in the code. The default
implementation wraps sentence-transformers and accepts any model on the
Hugging Face hub that library can load. Known model families get the query
and document prefixes their authors recommend (see :data:`PRESETS`); anything
else uses the prefixes from the settings, which default to none.

Models load lazily on first use and are shared. The embedding dimension is
read from the loaded model, never typed by a user.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import ModelError
from .observability import get_logger

if TYPE_CHECKING:
    from .config import Settings

log = get_logger("contextsearch.embedding")

# The symlink notice from huggingface_hub on Windows is noise for our users.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BGE_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelPreset:
    name: str
    dimension: int
    query_prefix: str = ""
    document_prefix: str = ""
    note: str = ""
    multilingual: bool = False
    size_mb: int = 0


PRESETS: dict[str, ModelPreset] = {
    p.name: p
    for p in (
        ModelPreset(
            "BAAI/bge-small-en-v1.5",
            384,
            BGE_QUERY,
            "",
            "Default. Small, fast on CPU, strong on English retrieval benchmarks.",
            size_mb=130,
        ),
        ModelPreset(
            "BAAI/bge-base-en-v1.5",
            768,
            BGE_QUERY,
            "",
            "Same family, roughly 3x slower, a little more accurate.",
            size_mb=440,
        ),
        ModelPreset(
            "BAAI/bge-m3",
            1024,
            "",
            "",
            "Multilingual, handles long passages (8k tokens). Heavy on CPU.",
            multilingual=True,
            size_mb=2200,
        ),
        ModelPreset(
            "sentence-transformers/all-MiniLM-L6-v2",
            384,
            "",
            "",
            "Very fast general-purpose model. Good baseline for comparisons.",
            size_mb=90,
        ),
        ModelPreset(
            "sentence-transformers/all-mpnet-base-v2",
            768,
            "",
            "",
            "Older general-purpose model with solid quality.",
            size_mb=420,
        ),
        ModelPreset(
            "intfloat/e5-small-v2",
            384,
            "query: ",
            "passage: ",
            "Small E5 model. Needs the query/passage prefixes, applied automatically.",
            size_mb=130,
        ),
        ModelPreset("intfloat/e5-base-v2", 768, "query: ", "passage: ", "Base E5 model.", size_mb=440),
        ModelPreset(
            "intfloat/multilingual-e5-small",
            384,
            "query: ",
            "passage: ",
            "Multilingual E5, 100 languages.",
            multilingual=True,
            size_mb=470,
        ),
        ModelPreset(
            "nomic-ai/nomic-embed-text-v1.5",
            768,
            "search_query: ",
            "search_document: ",
            "Long context (8k). Needs trust_remote_code.",
            size_mb=550,
        ),
        ModelPreset(
            "Snowflake/snowflake-arctic-embed-s",
            384,
            BGE_QUERY,
            "",
            "Small Arctic model tuned for retrieval.",
            size_mb=130,
        ),
        ModelPreset(
            "mixedbread-ai/mxbai-embed-large-v1",
            1024,
            BGE_QUERY,
            "",
            "Large English model, strong quality, slow on CPU.",
            size_mb=1300,
        ),
        ModelPreset(
            "Qwen/Qwen3-Embedding-0.6B",
            1024,
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "",
            "Multilingual, instruction-aware. Needs a GPU to be comfortable.",
            multilingual=True,
            size_mb=1200,
        ),
    )
}

_FAMILY_PREFIXES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"bge-(small|base|large)-(en|zh)-v1\.5", re.I), BGE_QUERY, ""),
    (re.compile(r"(multilingual-)?e5-", re.I), "query: ", "passage: "),
    (re.compile(r"nomic-embed-text", re.I), "search_query: ", "search_document: "),
    (re.compile(r"snowflake-arctic-embed", re.I), BGE_QUERY, ""),
]


def prefixes_for(model_name: str, settings: Settings | None = None) -> tuple[str, str]:
    """Query and document prefixes for a model: preset, then family rule, then settings."""
    preset = PRESETS.get(model_name)
    if preset:
        return preset.query_prefix, preset.document_prefix
    for pattern, q, d in _FAMILY_PREFIXES:
        if pattern.search(model_name):
            return q, d
    if settings is not None:
        return settings.embedding_query_prefix, settings.embedding_document_prefix
    return "", ""


class Embedder(ABC):
    """Texts in, unit vectors out."""

    name: str = ""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": self.dimension, "kind": type(self).__name__}

    def warm_up(self) -> None:
        self.embed_queries(["warm up"])


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        device: str = "",
        batch_size: int = 32,
        max_seq_length: int = 0,
        query_prefix: str = "",
        document_prefix: str = "",
        cache_dir: str = "",
        trust_remote_code: bool = False,
    ):
        self.name = model_name
        self.device_hint = device
        self.batch_size = max(1, batch_size)
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.cache_dir = cache_dir or None
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        self._dimension = 0
        self._device = ""
        self._lock = threading.RLock()
        self._load_seconds = 0.0

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import time

            started = time.perf_counter()
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ModelError(
                    "sentence-transformers is not installed. Install with: pip install 'contextsearch[dense]'"
                ) from exc
            device = self.device_hint or pick_device()
            try:
                model = SentenceTransformer(
                    self.name,
                    device=device,
                    cache_folder=self.cache_dir,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception as exc:
                raise ModelError(
                    f"cannot load embedding model {self.name!r}: {type(exc).__name__}: {exc}", {"model": self.name}
                ) from exc
            if self.max_seq_length:
                model.max_seq_length = self.max_seq_length
            model.eval()
            torch.set_grad_enabled(False)
            self._model = model
            self._device = device
            # Renamed in sentence-transformers 6; support both.
            get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            self._dimension = int(get_dim() or 0)
            if not self._dimension:
                self._dimension = int(model.encode(["probe"], normalize_embeddings=True).shape[1])
            self._load_seconds = time.perf_counter() - started
            log.info(
                "embedding model loaded",
                extra_fields={
                    "model": self.name,
                    "device": device,
                    "dimension": self._dimension,
                    "seconds": round(self._load_seconds, 1),
                },
            )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        if not self._dimension:
            preset = PRESETS.get(self.name)
            if preset and not self.loaded:
                return preset.dimension
            self.load()
        return self._dimension

    @property
    def device(self) -> str:
        return self._device

    # ---------------------------------------------------------------- encoding
    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        self.load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        payload = [prefix + t if prefix else t for t in texts]
        with self._lock:  # bounds memory: one batch in flight per model
            try:
                vectors = self._model.encode(
                    payload,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise ModelError(
                    f"embedding failed with {self.name}: {type(exc).__name__}: {exc}", {"model": self.name}
                ) from exc
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.document_prefix)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def info(self) -> dict[str, Any]:
        preset = PRESETS.get(self.name)
        return {
            "name": self.name,
            "kind": "sentence-transformers",
            "dimension": self.dimension if (self.loaded or preset) else None,
            "loaded": self.loaded,
            "device": self._device or None,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "max_seq_length": getattr(self._model, "max_seq_length", None)
            if self.loaded
            else (self.max_seq_length or None),
            "load_seconds": round(self._load_seconds, 1) if self.loaded else None,
            "note": preset.note if preset else "",
        }


class HashingEmbedder(Embedder):
    """A deterministic bag-of-words embedder for tests and demos.

    It hashes each word into one of ``dimension`` buckets and L2-normalises
    the counts. Texts that share words land close together, which is enough to
    exercise the dense and hybrid code paths without downloading a model. It
    is not a substitute for a real model.
    """

    def __init__(self, dimension: int = 64, name: str = "test/hashing-64"):
        self.name = name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.blake2b(word.encode(), digest_size=4).hexdigest(), 16) % self._dimension
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return (
            np.vstack([self._vector(t) for t in texts]) if texts else np.zeros((0, self._dimension), dtype=np.float32)
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


def pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def dense_available() -> tuple[bool, str]:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


class EmbedderRegistry:
    """One embedder instance per model name, created on demand."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embedders: dict[str, Embedder] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> Embedder:
        with self._lock:
            embedder = self._embedders.get(model_name)
            if embedder is None:
                embedder = self._create(model_name)
                self._embedders[model_name] = embedder
            return embedder

    def register(self, embedder: Embedder) -> None:
        with self._lock:
            self._embedders[embedder.name] = embedder

    def has(self, model_name: str) -> bool:
        with self._lock:
            return model_name in self._embedders

    def loaded(self) -> list[str]:
        with self._lock:
            return [name for name, e in self._embedders.items() if getattr(e, "loaded", True)]

    def _create(self, model_name: str) -> Embedder:
        if model_name.startswith("test/hashing-"):
            dim = int(model_name.rsplit("-", 1)[1])
            return HashingEmbedder(dimension=dim, name=model_name)
        query_prefix, document_prefix = prefixes_for(model_name, self.settings)
        return SentenceTransformerEmbedder(
            model_name,
            device=self.settings.embedding_device,
            batch_size=self.settings.embedding_batch_size,
            max_seq_length=self.settings.embedding_max_seq_length,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            cache_dir=self.settings.embedding_cache_dir,
            trust_remote_code="nomic" in model_name.lower(),
        )


class QueryEmbeddingCache:
    """Small LRU so repeated queries skip the model."""

    def __init__(self, size: int = 512):
        self.size = max(0, size)
        self._items: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, model: str, text: str) -> np.ndarray | None:
        if not self.size:
            return None
        key = (model, text)
        with self._lock:
            vec = self._items.get(key)
            if vec is not None:
                self._items.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return vec

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def put(self, model: str, text: str, vec: np.ndarray) -> None:
        if not self.size:
            return
        with self._lock:
            self._items[(model, text)] = vec
            while len(self._items) > self.size:
                self._items.popitem(last=False)
