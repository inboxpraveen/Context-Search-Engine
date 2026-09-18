"""Logging, request ids, traces and metrics.

Three small pieces that together make a search explainable after the fact:

* :func:`configure_logging` sets up text or JSON log lines. Every line carries
  the current request id when there is one.
* :class:`Trace` records named spans (lexical, dense:model, fusion, rerank)
  with their durations and a few attributes. The finished trace is returned in
  the search response and written to the log, so the same request id ties the
  two together.
* :class:`Metrics` keeps counters and latency histograms in memory and renders
  them in the Prometheus text format for ``/api/v1/system/metrics``.

Nothing here needs a third-party package.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(value: str | None) -> str:
    rid = value or new_request_id()
    _request_id.set(rid)
    return rid


def current_request_id() -> str:
    return _request_id.get()


# ------------------------------------------------------------------ logging


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = current_request_id()
        if rid:
            payload["request_id"] = rid
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = current_request_id()
        prefix = f"[{rid}] " if rid else ""
        base = f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {record.levelname:<7} {record.name}: {prefix}{record.getMessage()}"
        extra = getattr(record, "extra_fields", None)
        if extra:
            base += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class _ExtraAdapter(logging.LoggerAdapter):
    """Lets callers write ``log.info("indexed", extra_fields={...})``."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        fields = kwargs.pop("extra_fields", None)
        if fields:
            kwargs.setdefault("extra", {})["extra_fields"] = fields
        return msg, kwargs


def get_logger(name: str) -> _ExtraAdapter:
    return _ExtraAdapter(logging.getLogger(name), {})


def configure_logging(level: str = "INFO", fmt: str = "text", log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter: logging.Formatter = _JsonFormatter() if fmt == "json" else _TextFormatter()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    # Library chatter that does not help an operator.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers", "transformers", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ------------------------------------------------------------------ traces


@dataclass
class Span:
    name: str
    started: float
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {"name": self.name, "duration_ms": round(self.duration_ms, 2)}
        if self.attributes:
            out["attributes"] = self.attributes
        return out


class Trace:
    """Timing and attributes for one request, safe to read after it finishes."""

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or current_request_id() or new_request_id()
        self.started = time.perf_counter()
        self.spans: list[Span] = []
        self.attributes: dict[str, Any] = {}
        self.warnings: list[str] = []
        self._ended: float | None = None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, started=time.perf_counter(), attributes=dict(attributes))
        try:
            yield span
        finally:
            span.duration_ms = (time.perf_counter() - span.started) * 1000
            self.spans.append(span)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def finish(self) -> None:
        """Freeze the total. Until this is called the total keeps growing."""
        if self._ended is None:
            self._ended = time.perf_counter()

    @property
    def total_ms(self) -> float:
        end = self._ended if self._ended is not None else time.perf_counter()
        return (end - self.started) * 1000

    def timings(self) -> dict[str, float]:
        out = {s.name: round(s.duration_ms, 2) for s in self.spans}
        out["total"] = round(self.total_ms, 2)
        return out

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "request_id": self.request_id,
            "total_ms": round(self.total_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
        }
        if self.attributes:
            out["attributes"] = self.attributes
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out


# ------------------------------------------------------------------ metrics

_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


class Metrics:
    """In-process counters and histograms, rendered as Prometheus text."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value_ms: float, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            buckets = self._hist.setdefault(key, [0.0] * (len(_BUCKETS_MS) + 3))
            for i, edge in enumerate(_BUCKETS_MS):
                if value_ms <= edge:
                    buckets[i] += 1
            buckets[len(_BUCKETS_MS)] += 1  # +Inf
            buckets[len(_BUCKETS_MS) + 1] += value_ms  # sum
            buckets[len(_BUCKETS_MS) + 2] += 1  # count

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {f"{n}{_fmt_labels(lab)}": v for (n, lab), v in self._counters.items()}
            hist = {
                f"{n}{_fmt_labels(lab)}": {
                    "count": int(b[-1]),
                    "sum_ms": round(b[-2], 1),
                    "avg_ms": round(b[-2] / b[-1], 1) if b[-1] else 0.0,
                }
                for (n, lab), b in self._hist.items()
            }
            return {"counters": counters, "latency": hist, "gauges": dict(self._gauges)}

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}_total{_fmt_labels(labels)} {value:g}")
            for name, value in sorted(self._gauges.items()):
                lines.append(f"{name} {value:g}")
            for (name, labels), buckets in sorted(self._hist.items()):
                base = dict(labels)
                for i, edge in enumerate(_BUCKETS_MS):
                    lines.append(
                        f"{name}_bucket{_fmt_labels(tuple(sorted({**base, 'le': str(edge)}.items())))} {buckets[i]:g}"
                    )
                lines.append(
                    f"{name}_bucket{_fmt_labels(tuple(sorted({**base, 'le': '+Inf'}.items())))} {buckets[len(_BUCKETS_MS)]:g}"
                )
                lines.append(f"{name}_sum{_fmt_labels(labels)} {buckets[len(_BUCKETS_MS) + 1]:g}")
                lines.append(f"{name}_count{_fmt_labels(labels)} {buckets[len(_BUCKETS_MS) + 2]:g}")
        return "\n".join(lines) + "\n"


def _fmt_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"


metrics = Metrics()
