"""Reader contract and the registry that maps file extensions to readers.

A reader turns one file into a :class:`ParsedDocument`: a list of pages of
plain text plus a title. Page numbers are real for PDFs and slide numbers for
PPTX. Formats without pages (DOCX, text, HTML) produce a single page, and the
character offsets inside it are what the search results point at.

Adding a format means writing a class with ``name``, ``extensions`` and
``read()`` and calling :func:`register`. See ``docs/EXTENDING.md``.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ReadError, UnsupportedFileError

if TYPE_CHECKING:
    from ..config import Settings


@dataclass
class Page:
    number: int
    text: str
    ocr: bool = False


@dataclass
class ParsedDocument:
    pages: list[Page]
    title: str = ""
    reader: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def has_text(self) -> bool:
        return any(p.text.strip() for p in self.pages)


class Reader(ABC):
    name: str = ""
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def read(self, path: Path, settings: Settings) -> ParsedDocument:
        """Parse the file. Raise :class:`ReadError` for anything that goes wrong."""


_readers: dict[str, Reader] = {}


def register(reader: Reader, replace: bool = True) -> None:
    for ext in reader.extensions:
        ext = ext.lower().lstrip(".")
        if ext in _readers and not replace:
            continue
        _readers[ext] = reader


def unregister_extension(ext: str) -> None:
    _readers.pop(ext.lower().lstrip("."), None)


def reader_for(path: Path | str) -> Reader:
    ext = Path(path).suffix.lower().lstrip(".")
    try:
        return _readers[ext]
    except KeyError:
        raise UnsupportedFileError(f"no reader for .{ext} files", {"path": str(path)}) from None


def supported_extensions() -> set[str]:
    return set(_readers)


def read_document(path: Path | str, settings: Settings) -> ParsedDocument:
    """Read a file with the registered reader, translating OS errors into ReadError."""
    path = Path(path)
    reader = reader_for(path)
    try:
        doc = reader.read(path, settings)
    except ReadError:
        raise
    except PermissionError as exc:
        # Typical on Windows when the file is open in Word or Acrobat. The sync
        # loop retries it on the next scan.
        raise ReadError(f"file is locked or not readable: {path.name}", {"path": str(path), "locked": True}) from exc
    except OSError as exc:
        raise ReadError(f"cannot read {path.name}: {exc}", {"path": str(path)}) from exc
    except Exception as exc:  # a parser bug should not take the whole scan down
        raise ReadError(
            f"{reader.name} failed on {path.name}: {type(exc).__name__}: {exc}", {"path": str(path)}
        ) from exc
    doc.reader = doc.reader or reader.name
    for page in doc.pages:
        page.text = normalize_text(page.text)
    if not doc.title:
        doc.title = guess_title(doc, path)
    return doc


_WS_RUN = re.compile("[ \t\f\v\u00a0]+")  # includes the no-break space
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    """Tidy extracted text without changing its reading order.

    Line breaks are kept because the chunker uses paragraph breaks. Runs of
    spaces collapse to one, trailing spaces go, and more than two blank lines
    become two. Characters are NFC-normalised so the same word always hashes
    and tokenises the same way.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    text = _WS_RUN.sub(" ", text)
    text = _TRAILING.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def guess_title(doc: ParsedDocument, path: Path) -> str:
    """First short line of text, without markdown heading markers."""
    for page in doc.pages:
        for line in page.text.splitlines():
            line = line.strip().lstrip("#=*- ").strip()
            if 3 <= len(line) <= 120:
                return line
    return path.stem
