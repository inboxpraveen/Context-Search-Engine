"""Exceptions raised by contextsearch.

Everything derives from :class:`ContextSearchError` so callers can catch one
type. The API layer maps each subclass to an HTTP status and a stable error
code, see ``api/errors.py``.
"""

from __future__ import annotations


class ContextSearchError(Exception):
    """Base class for all errors raised by this package."""

    code = "internal_error"
    status = 500

    def __init__(self, message: str = "", details: dict | None = None):
        super().__init__(message or self.__doc__ or self.code)
        self.message = message or self.__doc__ or self.code
        self.details = details or {}


class ConfigurationError(ContextSearchError):
    """A setting has an invalid value or a required dependency is missing."""

    code = "configuration_error"
    status = 500


class ValidationError(ContextSearchError):
    """The request or argument is malformed."""

    code = "validation_error"
    status = 400


class NotFoundError(ContextSearchError):
    """The requested document, source, job or model does not exist."""

    code = "not_found"
    status = 404


class ConflictError(ContextSearchError):
    """The operation clashes with the current state, for example adding a folder twice."""

    code = "conflict"
    status = 409


class UnsupportedFileError(ContextSearchError):
    """No reader is registered for this file type."""

    code = "unsupported_file"
    status = 415


class ReadError(ContextSearchError):
    """The file exists but could not be read or parsed."""

    code = "read_error"
    status = 422


class ModelError(ContextSearchError):
    """An embedding or reranking model could not be loaded or run."""

    code = "model_error"
    status = 503


class AuthError(ContextSearchError):
    """The API key is missing or wrong."""

    code = "unauthorized"
    status = 401


class ForbiddenError(ContextSearchError):
    """The path is outside the allowed roots or the action is disabled."""

    code = "forbidden"
    status = 403
