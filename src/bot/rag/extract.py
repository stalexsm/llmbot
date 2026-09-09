"""Извлечение текста из документа.

Тикет 02 закрывает текстовые форматы (.txt, .md); PDF и DOCX подключаются
парсерами pypdf и python-docx в тикете 06. Извлечение — чистая функция
«байты → текст»: ошибки формата и пустого содержимого — прикладные исключения,
а не сырые ``UnicodeDecodeError``.
"""

from pathlib import Path

from bot.application.errors import EmptyDocumentError, UnsupportedDocumentError

# .md — тот же простой текст: заголовки учитывает структурный чанкинг.
_PLAIN_TEXT_SUFFIXES = frozenset({".txt", ".md"})


def document_kind(name: str) -> str:
    """Тип документа: расширение имени без точки, в нижнем регистре."""
    return Path(name).suffix.lower().lstrip(".")


def extract_text(name: str, content: bytes) -> str:
    """Извлечь текст из файла по его имени и содержимому.

    Неподдерживаемое расширение и не-текстовое содержимое —
    ``UnsupportedDocumentError``; файл без текста — ``EmptyDocumentError``.
    """
    suffix = Path(name).suffix.lower()
    if suffix not in _PLAIN_TEXT_SUFFIXES:
        raise UnsupportedDocumentError(f"Document format is not supported: {suffix or '<none>'}")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDocumentError("Document content is not text") from exc
    if not text.strip():
        raise EmptyDocumentError("Document contains no text")
    return text
