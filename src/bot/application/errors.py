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


class EmbeddingError(ApplicationError):
    """Эмбеддинг-модель ответила некорректно (битый формат, размерность)."""


class EmbeddingTimeoutError(EmbeddingError):
    """Эмбеддинг-модель не уложилась в таймаут."""


class EmbeddingUnavailableError(EmbeddingError):
    """Эмбеддинг-провайдер недоступен или вернул ошибочный статус."""


class RagError(ApplicationError):
    """Базовая ошибка RAG-слоя: индексация или поиск не выполнились."""


class UnsupportedDocumentError(RagError):
    """Формат документа не поддерживается или файл не является текстом."""


class EmptyDocumentError(RagError):
    """Из документа не удалось извлечь текст: он пуст."""


class DocumentTooLargeError(RagError):
    """Документ превышает настроенный лимит (байты, текст или чанки)."""


class RagStorageError(RagError):
    """Rag-БД недоступна: дисковая операция с файлом rag-БД не удалась."""
