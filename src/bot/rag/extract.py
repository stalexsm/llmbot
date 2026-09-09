"""Извлечение текста из документа: .txt/.md, .pdf (pypdf), .docx (python-docx).

Извлечение — чистая функция «байты → страницы» (``PageText``): у PDF каждая
страница получает номер — он поедет в чанки и Источники; у странично-слепых
форматов страница одна и без номера. Ошибки формата, повреждённого файла
и пустого содержимого — прикладные исключения: сырые исключения парсеров
(pypdf поднимает целый зоопарк, python-docx — ошибки zip-контейнера) наружу
не выходят.
"""

import io
import re
from pathlib import Path

from docx import Document as load_docx
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from bot.application.errors import (
    CorruptedDocumentError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from bot.rag.models import DocumentKind, PageText

# .md — тот же простой текст: заголовки учитывает структурный чанкинг.
_SUFFIX_KINDS: dict[str, DocumentKind] = {
    ".txt": DocumentKind.TXT,
    ".md": DocumentKind.MD,
    ".pdf": DocumentKind.PDF,
    ".docx": DocumentKind.DOCX,
}

# Стиль заголовка DOCX («Heading 1» … «Heading 6»): встроенные стили python-docx
# нормализует к английским именам независимо от языка документа.
_HEADING_STYLE_RE = re.compile(r"Heading ([1-9])")


def document_kind(name: str) -> DocumentKind:
    """Тип документа по расширению имени; неизвестное расширение — ошибка."""
    kind = _SUFFIX_KINDS.get(Path(name).suffix.lower())
    if kind is None:
        suffix = Path(name).suffix.lower()
        raise UnsupportedDocumentError(f"Document format is not supported: {suffix or '<none>'}")
    return kind


def extract_pages(name: str, content: bytes) -> tuple[PageText, ...]:
    """Извлечь текст из файла по его имени и содержимому, по страницам.

    Неподдерживаемое расширение и не-текстовое содержимое текстовых форматов —
    ``UnsupportedDocumentError``; повреждённый PDF или DOCX —
    ``CorruptedDocumentError``; файл без текста — ``EmptyDocumentError``.
    Пустые страницы PDF отбрасываются: чанк не ссылается на пустую страницу.
    """
    kind = document_kind(name)
    if kind is DocumentKind.PDF:
        pages = _pdf_pages(content)
    elif kind is DocumentKind.DOCX:
        pages = _docx_pages(content)
    else:
        pages = (_plain_page(content),)
    pages = tuple(page for page in pages if page.text.strip())
    if not pages:
        raise EmptyDocumentError("Document contains no text")
    return pages


def _plain_page(content: bytes) -> PageText:
    """Декодировать .txt/.md; не-текстовое содержимое — ошибка формата."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDocumentError("Document content is not text") from exc
    return PageText(page=None, text=text)


def _pdf_pages(content: bytes) -> tuple[PageText, ...]:
    """Страницы PDF с номерами с единицы; битый файл — ``CorruptedDocumentError``.

    Сбой на одной странице рвёт весь документ: частичный корпус с дырами
    хуже честного отказа (зашифрованные и сканированные файлы не читаются
    тоже — первый как повреждённый, вторая как пустой документ).
    """
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = tuple(
            PageText(page=number, text=(page.extract_text() or "").strip())
            for number, page in enumerate(reader.pages, start=1)
        )
    except Exception as exc:
        raise CorruptedDocumentError("PDF file is corrupted or unreadable") from exc
    return pages


def _docx_pages(content: bytes) -> tuple[PageText, ...]:
    """Абзацы DOCX одним блоком без страницы; битый файл — ошибка повреждения.

    Заголовки встроенных стилей становятся разметкой Markdown: структурный
    чанкинг режет документ по ним, как в .md.
    """
    try:
        document = load_docx(io.BytesIO(content))
        paragraphs = list(document.paragraphs)
    except Exception as exc:
        raise CorruptedDocumentError("DOCX file is corrupted or unreadable") from exc
    lines = [line for line in (_docx_line(paragraph) for paragraph in paragraphs) if line]
    return (PageText(page=None, text="\n\n".join(lines)),)


def _docx_line(paragraph: Paragraph) -> str:
    """Абзац → строка; заголовок «Heading N» → префикс из N решёток."""
    text = paragraph.text.strip()
    if not text:
        return ""
    style_name = paragraph.style.name if paragraph.style is not None else ""
    if (match := _HEADING_STYLE_RE.fullmatch(style_name)) is not None:
        return f"{'#' * min(int(match.group(1)), 6)} {text}"
    return text
