"""Application-level errors.

Infrastructure exceptions must never leak into the application layer: the
Ollama adapter maps them onto these types, and the Telegram adapter converts
them into safe user-facing messages.
"""


class ApplicationError(Exception):
    """Base class for errors that cross application boundaries."""


class InferenceError(ApplicationError):
    """Inference failed for a provider-specific reason (e.g. malformed response)."""


class InferenceTimeoutError(InferenceError):
    """Inference did not complete within the configured timeout."""


class InferenceUnavailableError(InferenceError):
    """The inference provider is unreachable or returned an error status."""


class EmptyInferenceResponseError(InferenceError):
    """The inference provider returned an empty response."""


class SessionStorageError(ApplicationError):
    """Чат-сессия недоступна: дисковая операция с файлом сессии не удалась."""
