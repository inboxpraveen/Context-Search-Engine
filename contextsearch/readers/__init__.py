"""Turn files into pages of text. See ``base.py`` for the contract."""

from .base import (
    Page,
    ParsedDocument,
    Reader,
    normalize_text,
    read_document,
    reader_for,
    register,
    supported_extensions,
    unregister_extension,
)
from .formats import DocxReader, HtmlReader, ImageReader, PdfReader, PptxReader, TextReader, register_builtin_readers

__all__ = [
    "DocxReader",
    "HtmlReader",
    "ImageReader",
    "Page",
    "ParsedDocument",
    "PdfReader",
    "PptxReader",
    "Reader",
    "TextReader",
    "normalize_text",
    "read_document",
    "reader_for",
    "register",
    "register_builtin_readers",
    "supported_extensions",
    "unregister_extension",
]
