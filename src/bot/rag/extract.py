"""Извлечение текста из документа.

Тикет 02 закрывает текстовые форматы (.txt, .md); PDF и DOCX подключаются
парсерами pypdf и python-docx в тикете 06. Извлечение — чистая функция
«байты → текст»: ошибки формата и пустого содержимого — прикладные исключения,
а не сырые ``UnicodeDecodeError``.
"""

from pathlib import Path

from bot.application.errors import EmptyDocumentError, UnsupportedDocumentError
from bot.rag.models import DocumentKind

# .md — тот же простой текст: заголовки учитывает структурный чанкинг.
_SUFFIX_KINDS: dict[str, DocumentKind] = {
    ".txt": DocumentKind.TXT,
    ".md": DocumentKind.MD,
}


def document_kind(name: str) -> DocumentKind:
    """Тип документа по расширению имени; неизвестное расширение — ошибка."""
    kind = _SUFFIX_KINDS.get(Path(name).suffix.lower())
    if kind is None:
        suffix = Path(name).suffix.lower()
        raise UnsupportedDocumentError(f"Document format is not supported: {suffix or '<none>'}")
    return kind


def extract_text(name: str, content: bytes) -> str:
    """Извлечь текст из файла по его имени и содержимому.

    Неподдерживаемое расширение и не-текстовое содержимое —
    ``UnsupportedDocumentError``; файл без текста — ``EmptyDocumentError``.
    """
    document_kind(name)  # валидация формата: тоже UnsupportedDocumentError
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDocumentError("Document content is not text") from exc
    if not text.strip():
        raise EmptyDocumentError("Document contains no text")
    return text
