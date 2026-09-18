"""Cross-encoder reranking.

A cross-encoder reads the query and a candidate passage together and outputs
one relevance score. It is far more precise than comparing two independent
embeddings, and far more expensive: one forward pass per candidate. That is
why it runs only on the top fused candidates, and only when asked.

Scores are mapped through a sigmoid so they sit in 0 to 1 and can be shown as
a confidence. The ms-marco and bge-reranker families were trained with a
binary objective, so this reading of the logit is the intended one.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import ModelError
from .observability import get_logger

if TYPE_CHECKING:
    from .config import Settings

log = get_logger("contextsearch.rerank")

RERANKER_PRESETS: dict[str, str] = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Default. Fast, English, about 90 MB.",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "Same family, twice the depth, a bit more accurate.",
    "BAAI/bge-reranker-base": "Multilingual, stronger than MiniLM, about 1.1 GB.",
    "BAAI/bge-reranker-v2-m3": "Multilingual, long inputs, best quality of the list, slow on CPU.",
    "mixedbread-ai/mxbai-rerank-base-v1": "English, good quality and speed balance.",
}


class Reranker(ABC):
    name: str = ""

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance of each passage to the query, in 0 to 1."""

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "kind": type(self).__name__}


class CrossEncoderReranker(Reranker):
    def __init__(
        self, model_name: str, device: str = "", batch_size: int = 16, max_length: int = 512, cache_dir: str = ""
    ):
        self.name = model_name
        self.device_hint = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir or None
        self._model: Any = None
        self._device = ""
        self._apply_sigmoid = True
        self._lock = threading.RLock()
        self._load_seconds = 0.0

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            started = time.perf_counter()
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ModelError(
                    "sentence-transformers is not installed. Install with: pip install 'contextsearch[dense]'"
                ) from exc
            from .embedding import pick_device

            device = self.device_hint or pick_device()
            try:
                model = CrossEncoder(self.name, device=device, max_length=self.max_length, cache_folder=self.cache_dir)
            except TypeError:
                model = CrossEncoder(self.name, device=device, max_length=self.max_length)
            except Exception as exc:
                raise ModelError(
                    f"cannot load reranker {self.name!r}: {type(exc).__name__}: {exc}", {"model": self.name}
                ) from exc
            # Newer sentence-transformers return raw logits (Identity activation);
            # older ones apply a sigmoid themselves. Apply ours only when needed.
            activation = getattr(model, "activation_fn", None) or getattr(model, "default_activation_function", None)
            self._apply_sigmoid = activation is None or type(activation).__name__ in ("Identity", "NoneType")
            self._model = model
            self._device = device
            self._load_seconds = time.perf_counter() - started
            log.info(
                "reranker loaded",
                extra_fields={"model": self.name, "device": device, "seconds": round(self._load_seconds, 1)},
            )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        self.load()
        pairs = [(query, p) for p in passages]
        with self._lock:
            try:
                raw = self._model.predict(
                    pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
                )
            except Exception as exc:
                raise ModelError(
                    f"reranking failed with {self.name}: {type(exc).__name__}: {exc}", {"model": self.name}
                ) from exc
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if self._apply_sigmoid:
            raw = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(raw, 0.0, 1.0)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "cross-encoder",
            "loaded": self.loaded,
            "device": self._device or None,
            "load_seconds": round(self._load_seconds, 1) if self.loaded else None,
            "note": RERANKER_PRESETS.get(self.name, ""),
        }


class OverlapReranker(Reranker):
    """Scores by query-word overlap. For tests; keeps the rerank path runnable without a model."""

    name = "test/overlap"

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        q = {w for w in query.lower().split() if w}
        out = []
        for p in passages:
            words = set(p.lower().split())
            out.append(len(q & words) / len(q) if q else 0.0)
        return np.asarray(out, dtype=np.float32)


class RerankerRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._items: dict[str, Reranker] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> Reranker:
        with self._lock:
            reranker = self._items.get(model_name)
            if reranker is None:
                if model_name == OverlapReranker.name:
                    reranker = OverlapReranker()
                else:
                    reranker = CrossEncoderReranker(
                        model_name,
                        device=self.settings.embedding_device,
                        batch_size=max(1, self.settings.embedding_batch_size // 2),
                        cache_dir=self.settings.embedding_cache_dir,
                    )
                self._items[model_name] = reranker
            return reranker

    def register(self, reranker: Reranker) -> None:
        with self._lock:
            self._items[reranker.name] = reranker
