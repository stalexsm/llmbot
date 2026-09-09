"""Unit-тесты извлечения текста из документов (Шов 4)."""

import pytest

from bot.application.errors import EmptyDocumentError, UnsupportedDocumentError
from bot.rag.extract import document_kind, extract_text


def test_txt_is_decoded() -> None:
    assert extract_text("notes.txt", "текст заметки".encode()) == "текст заметки"


def test_md_is_plain_text() -> None:
    content = "# Заголовок\n\nАбзац.".encode()

    assert extract_text("readme.md", content) == "# Заголовок\n\nАбзац."


def test_extension_case_is_ignored() -> None:
    assert extract_text("NOTES.TXT", "текст".encode()) == "текст"


@pytest.mark.parametrize("name", ["scan.pdf", "table.docx", "archive.zip", "noextension"])
def test_unsupported_extension_is_rejected(name: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        extract_text(name, "любое содержимое".encode())


def test_binary_content_with_text_extension_is_rejected() -> None:
    binary = bytes(range(256))

    with pytest.raises(UnsupportedDocumentError):
        extract_text("dump.txt", binary)


@pytest.mark.parametrize("content", [b"", b"   \n\t"])
def test_empty_document_is_rejected(content: bytes) -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", content)


def test_kind_is_lowercase_extension_without_dot() -> None:
    assert document_kind("Report.TXT") == "txt"
    assert document_kind("guide.Pdf") == "pdf"
    assert document_kind("noextension") == ""
