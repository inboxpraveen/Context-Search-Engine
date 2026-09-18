"""Optional OCR through RapidOCR (ONNX models, pip-installable, no system binary).

The engine is created on first use and shared. When the ``ocr`` extra is not
installed, :func:`available` is False and callers skip OCR with a warning
instead of failing the document.
"""

from __future__ import annotations

import threading
from typing import Any

from ..errors import ModelError
from ..observability import get_logger

log = get_logger("contextsearch.ocr")

_lock = threading.Lock()
_engine: Any = None
_checked = False
_import_error = ""


def available() -> bool:
    global _checked, _import_error
    if not _checked:
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except Exception as exc:  # ImportError or a broken onnxruntime install
            _import_error = f"{type(exc).__name__}: {exc}"
        _checked = True
    return not _import_error


def why_unavailable() -> str:
    available()
    return _import_error


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                if not available():
                    raise ModelError(
                        f"OCR is not available: {_import_error}. Install with: pip install 'contextsearch[ocr]'"
                    )
                from rapidocr_onnxruntime import RapidOCR

                _engine = RapidOCR()
                log.info("OCR engine loaded (RapidOCR)")
    return _engine


def image_to_text(image: Any) -> tuple[str, float]:
    """OCR a PIL image, numpy array or file path.

    Returns the recognised lines joined with newlines, in reading order, and
    the mean recognition confidence (0 when nothing was found).
    """
    import numpy as np

    engine = _get_engine()
    if hasattr(image, "convert"):  # PIL image
        image = np.asarray(image.convert("RGB"))
    with _lock:  # the ONNX session is not documented as thread-safe
        result, _elapsed = engine(image)
    if not result:
        return "", 0.0
    # Each row is [box, text, score]. Boxes come in reading order already;
    # sorting by the top-left corner guards against out-of-order rows.
    rows = sorted(result, key=lambda r: (round(r[0][0][1] / 10), r[0][0][0]))
    lines = [str(r[1]).strip() for r in rows if str(r[1]).strip()]
    scores = [float(r[2]) for r in rows]
    return "\n".join(lines), (sum(scores) / len(scores)) if scores else 0.0
