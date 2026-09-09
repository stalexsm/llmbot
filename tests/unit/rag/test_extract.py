"""Unit-тесты извлечения текста из документов (Шов 4).

Бинарные фикстуры .pdf/.docx коммитятся рядом (``fixtures/``): извлечение
проверяется на настоящих файлах, а повреждённые документы — мусорными
байтами и обрезанной фикстурой.
"""

from io import BytesIO

import pytest
from docx import Document as load_docx
from pypdf import PdfWriter

from bot.application.errors import (
    CorruptedDocumentError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from bot.rag.extract import document_kind, extract_pages
from bot.rag.models import DocumentKind
from tests.unit.rag.fixtures import load as load_fixture


def test_txt_is_decoded() -> None:
    pages = extract_pages("notes.txt", "текст заметки".encode())

    assert len(pages) == 1
    assert pages[0].page is None
    assert pages[0].text == "текст заметки"


def test_md_is_plain_text() -> None:
    content = "# Заголовок\n\nАбзац.".encode()

    pages = extract_pages("readme.md", content)

    assert len(pages) == 1
    assert pages[0].page is None
    assert pages[0].text == "# Заголовок\n\nАбзац."


def test_extension_case_is_ignored() -> None:
    pages = extract_pages("NOTES.TXT", "текст".encode())

    assert pages[0].text == "текст"


@pytest.mark.parametrize("name", ["archive.zip", "noextension"])
def test_unsupported_extension_is_rejected(name: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        extract_pages(name, "любое содержимое".encode())


def test_binary_content_with_text_extension_is_rejected() -> None:
    binary = bytes(range(256))

    with pytest.raises(UnsupportedDocumentError):
        extract_pages("dump.txt", binary)


@pytest.mark.parametrize("content", [b"", b"   \n\t"])
def test_empty_text_document_is_rejected(content: bytes) -> None:
    with pytest.raises(EmptyDocumentError):
        extract_pages("empty.txt", content)


def test_pdf_pages_carry_numbers_and_own_text() -> None:
    pages = extract_pages("sample.pdf", load_fixture("sample.pdf"))

    assert [page.page for page in pages] == [1, 2, 3]
    assert "28 calendar days" in pages[0].text
    assert "700 rubles" in pages[1].text
    assert "medical certificate" in pages[2].text
    # Содержимое страниц не перемешивается.
    assert "rubles" not in pages[0].text
    assert "Vacation" not in pages[2].text


def test_corrupted_pdf_is_rejected() -> None:
    with pytest.raises(CorruptedDocumentError):
        extract_pages("scan.pdf", b"%PDF-1.4 fake not really")


def test_truncated_pdf_is_rejected() -> None:
    whole = load_fixture("sample.pdf")

    with pytest.raises(CorruptedDocumentError):
        extract_pages("sample.pdf", whole[: len(whole) // 2])


def test_blank_pdf_is_empty_document() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(EmptyDocumentError):
        extract_pages("blank.pdf", buffer.getvalue())


def test_docx_is_extracted_with_headings() -> None:
    pages = extract_pages("sample.docx", load_fixture("sample.docx"))

    assert len(pages) == 1
    assert pages[0].page is None
    assert "# Регламент отпусков" in pages[0].text
    assert "## Командировки" in pages[0].text
    assert "28 календарных дней" in pages[0].text
    assert "700 рублей" in pages[0].text


def test_corrupted_docx_is_rejected() -> None:
    with pytest.raises(CorruptedDocumentError):
        extract_pages("fake.docx", b"just text, not a zip archive")


def test_docx_without_text_is_empty_document() -> None:
    buffer = BytesIO()
    load_docx().save(buffer)

    with pytest.raises(EmptyDocumentError):
        extract_pages("blank.docx", buffer.getvalue())


def test_kind_is_lowercase_extension_without_dot() -> None:
    assert document_kind("Report.TXT") == DocumentKind.TXT
    assert document_kind("notes.md") == DocumentKind.MD
    assert document_kind("guide.PDF") == DocumentKind.PDF
    assert document_kind("table.docx") == DocumentKind.DOCX


@pytest.mark.parametrize("name", ["noextension"])
def test_kind_for_unsupported_extension_is_rejected(name: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        document_kind(name)
