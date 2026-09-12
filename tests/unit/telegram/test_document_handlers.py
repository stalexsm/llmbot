"""Unit-тесты документных хендлеров: загрузка, /documents, /delete, /clear."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.stdlib
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import CommandObject
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import Message
from pytest import MonkeyPatch

from bot.application.documents import DocumentIndexResult, DocumentService
from bot.application.errors import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    DocumentTooLargeError,
    EmbeddingError,
    EmptyDocumentError,
    RagStorageError,
    UnsupportedDocumentError,
)
from bot.domain.ids import DocumentId, TelegramUserId
from bot.rag.models import DocumentInfo, DocumentKind
from bot.telegram.formatter import escape_markdown_v2
from bot.telegram.handlers import TelegramHandlers
from bot.telegram.loader import DocumentLoader
from tests.fakes import make_rag_service, make_telegram_message, mock_telegram_api
from tests.unit.rag.fixtures import load as load_fixture

OWNER_ID = 7


def make_document_message(
    bot: Bot,
    *,
    file_name: str = "doc.txt",
    with_author: bool = True,
) -> Message:
    """aiogram Message с документом, без сетевого I/O."""
    data: dict = {
        "message_id": 43,
        "date": 1735689600,
        "chat": {"id": 100, "type": "private"},
        "document": {
            "file_id": "FILE123",
            "file_unique_id": "uniq",
            "file_name": file_name,
            "file_size": 12,
        },
    }
    if with_author:
        data["from"] = {"id": OWNER_ID, "is_bot": False, "first_name": "Tester"}
    return Message.model_validate(data).as_(bot)


def make_document_handlers(
    logger: structlog.stdlib.BoundLogger,
    *,
    documents: DocumentService | AsyncMock | None = None,
    loader: DocumentLoader | AsyncMock | None = None,
) -> tuple[TelegramHandlers, AsyncMock, AsyncMock]:
    documents_mock = documents or AsyncMock(spec=DocumentService)
    assert isinstance(documents_mock, AsyncMock)
    documents_mock.list_documents.return_value = ()
    loader_mock = loader or AsyncMock(spec=DocumentLoader)
    assert isinstance(loader_mock, AsyncMock)
    loader_mock.load.return_value = "текст документа".encode()
    handlers = TelegramHandlers(
        service=AsyncMock(),  # process_message в этих тестах не участвует
        documents=documents_mock,
        document_loader=loader_mock,
        logger=logger,
        allowed_chat_ids=frozenset(),
    )
    return handlers, documents_mock, loader_mock


def sent_methods(request_mock: AsyncMock) -> list[SendMessage | EditMessageText]:
    return [call.args[1] for call in request_mock.await_args_list]


async def drain_indexing(handlers: TelegramHandlers) -> None:
    """Дождаться фоновых задач индексации, запущенных хендлером."""
    tasks = list(handlers._indexing_tasks)
    assert tasks
    await asyncio.gather(*tasks)


async def test_document_acknowledges_then_reports_ready(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.index.return_value = DocumentIndexResult("doc.txt", 3)
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_document_message(bot)
    await handlers.handle_document(message)
    await drain_indexing(handlers)

    methods = sent_methods(request_mock)
    # Первое — мгновенное подтверждение; последнее — правка статуса в «готово».
    assert isinstance(methods[0], SendMessage)
    assert escape_markdown_v2("Документ получен: doc.txt") in (methods[0].text or "")
    assert isinstance(methods[-1], EditMessageText)
    assert "Документ готов" in (methods[-1].text or "")
    # Индексация — под владельцем с границы Telegram-слоя.
    assert documents.index.await_count == 1
    assert int(documents.index.await_args.args[0].owner_id) == OWNER_ID
    assert documents.index.await_args.args[0].name == "doc.txt"


async def test_document_without_author_is_rejected(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_document_message(bot, with_author=False)
    await handlers.handle_document(message)

    sent = sent_methods(request_mock)
    assert len(sent) == 1
    assert isinstance(sent[0], SendMessage)
    assert "автора" in sent[0].text
    documents.index.assert_not_awaited()


async def test_pdf_and_docx_reach_ready_through_real_services(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end через бот: .pdf и .docx доходят до «Документ готов».

    Настоящие DocumentService и RagService (фейковые эмбеддинги, реальный
    SQLite во временном файле, замоканный Telegram): загрузка → индексация
    → документ в корпусе владельца, доступный поиску.
    """
    documents = DocumentService(make_rag_service(tmp_path, logger), logger)
    loader = AsyncMock(spec=DocumentLoader)
    handlers = TelegramHandlers(
        service=AsyncMock(),  # process_message в этих тестах не участвует
        documents=documents,
        document_loader=loader,
        logger=logger,
        allowed_chat_ids=frozenset(),
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    for file_name, fixture in (("handbook.pdf", "sample.pdf"), ("handbook.docx", "sample.docx")):
        loader.load.return_value = load_fixture(fixture)
        await handlers.handle_document(make_document_message(bot, file_name=file_name))
        await drain_indexing(handlers)
        last = sent_methods(request_mock)[-1]
        assert isinstance(last, EditMessageText)
        assert "Документ готов" in (last.text or "")

    corpus = documents.list_documents(TelegramUserId(OWNER_ID))
    assert sorted(document.name for document in corpus) == ["handbook.docx", "handbook.pdf"]


async def test_unsupported_format_becomes_friendly_error(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.index.side_effect = UnsupportedDocumentError("bad format")
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_document(make_document_message(bot, file_name="doc.exe"))
    await drain_indexing(handlers)

    last = sent_methods(request_mock)[-1]
    assert isinstance(last, EditMessageText)
    assert escape_markdown_v2(".txt, .md, .pdf") in (last.text or "")


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (EmptyDocumentError("empty"), "не оказалось текста"),
        (CorruptedDocumentError("corrupt"), "повреждён"),
        (DocumentTooLargeError("big"), "слишком большой"),
        (EmbeddingError("embed"), "Эмбеддинг-модель"),
        (RagStorageError("db"), "Хранилище документов"),
    ],
)
async def test_indexing_errors_map_to_friendly_texts(
    bot: Bot,
    logger: structlog.stdlib.BoundLogger,
    monkeypatch: MonkeyPatch,
    error: Exception,
    fragment: str,
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.index.side_effect = error
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_document(make_document_message(bot))
    await drain_indexing(handlers)

    last = sent_methods(request_mock)[-1]
    assert isinstance(last, EditMessageText)
    assert escape_markdown_v2(fragment) in (last.text or "")


async def test_download_failure_is_reported_without_indexing(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, loader = make_document_handlers(logger)
    loader.load.side_effect = TelegramAPIError(method=None, message="boom")  # type: ignore
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_document(make_document_message(bot))
    await drain_indexing(handlers)

    documents.index.assert_not_awaited()
    last = sent_methods(request_mock)[-1]
    assert isinstance(last, EditMessageText)
    assert "скачать файл" in (last.text or "")


async def test_documents_command_lists_owner_corpus(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.list_documents = MagicMock(
        return_value=(
            DocumentInfo(
                id=DocumentId(1),
                owner_id=TelegramUserId(OWNER_ID),
                name="handbook.txt",
                kind=DocumentKind.TXT,
                created_at="2026-01-01T00:00:00Z",
            ),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_documents(make_telegram_message("/documents").as_(bot))

    sent = sent_methods(request_mock)
    assert len(sent) == 1
    assert isinstance(sent[0], SendMessage)
    assert escape_markdown_v2("handbook.txt") in sent[0].text


async def test_documents_command_reports_empty_corpus(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, _, _ = make_document_handlers(logger)
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_documents(make_telegram_message("/documents").as_(bot))

    sent = sent_methods(request_mock)[0]
    assert isinstance(sent, SendMessage)
    assert "пока нет документов" in sent.text


async def test_delete_removes_document_and_confirms(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    request_mock = mock_telegram_api(bot, monkeypatch)

    message = make_telegram_message("/delete doc.txt").as_(bot)
    command = CommandObject(command="delete", args="doc.txt")
    await handlers.handle_delete(message, command)

    documents.delete_document.assert_called_once()
    args = documents.delete_document.call_args.args
    assert int(args[0]) == OWNER_ID
    assert args[1] == "doc.txt"
    sent = sent_methods(request_mock)[0]
    assert isinstance(sent, SendMessage)
    assert escape_markdown_v2("Документ удалён: doc.txt") in sent.text


async def test_delete_without_argument_shows_usage(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_delete(
        make_telegram_message("/delete").as_(bot),
        CommandObject(command="delete", args=None),
    )

    documents.delete_document.assert_not_called()
    sent = sent_methods(request_mock)[0]
    assert isinstance(sent, SendMessage)
    assert "/delete" in sent.text


async def test_delete_unknown_name_answers_with_corpus_list(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.delete_document.side_effect = DocumentNotFoundError("nope")
    documents.list_documents = MagicMock(
        return_value=(
            DocumentInfo(
                id=DocumentId(1),
                owner_id=TelegramUserId(OWNER_ID),
                name="handbook.txt",
                kind=DocumentKind.TXT,
                created_at="2026-01-01T00:00:00Z",
            ),
        )
    )
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_delete(
        make_telegram_message("/delete nope.txt").as_(bot),
        CommandObject(command="delete", args="nope.txt"),
    )

    sent = sent_methods(request_mock)[0]
    assert isinstance(sent, SendMessage)
    assert "не найден" in sent.text
    assert escape_markdown_v2("handbook.txt") in sent.text


async def test_clear_wipes_corpus_and_reports_count(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.clear_documents.return_value = 3
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_clear(make_telegram_message("/clear").as_(bot))

    documents.clear_documents.assert_called_once()
    args = documents.clear_documents.call_args.args
    assert int(args[0]) == OWNER_ID
    sent = sent_methods(request_mock)[0]
    assert isinstance(sent, SendMessage)
    assert "удалено документов — 3" in sent.text


async def test_clear_empty_corpus_reports_no_documents(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.clear_documents.return_value = 0
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_clear(make_telegram_message("/clear").as_(bot))

    sent = sent_methods(request_mock)[0]
    assert isinstance(sent, SendMessage)
    assert "пока нет документов" in sent.text


async def test_clear_storage_failure_answers_error(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    documents.clear_documents.side_effect = RagStorageError("db down")
    request_mock = mock_telegram_api(bot, monkeypatch)

    await handlers.handle_clear(make_telegram_message("/clear").as_(bot))

    sent = sent_methods(request_mock)[0]
    assert isinstance(sent, SendMessage)
    assert "Хранилище документов" in sent.text


async def test_commands_without_author_get_friendly_error(
    bot: Bot, logger: structlog.stdlib.BoundLogger, monkeypatch: MonkeyPatch
) -> None:
    handlers, documents, _ = make_document_handlers(logger)
    request_mock = mock_telegram_api(bot, monkeypatch)

    from aiogram.filters import CommandObject

    message = make_telegram_message("/documents", with_author=False).as_(bot)
    await handlers.handle_documents(message)
    await handlers.handle_delete(
        make_telegram_message("/delete x.txt", with_author=False).as_(bot),
        CommandObject(command="delete", args="x.txt"),
    )

    assert documents.index.await_count == 0
    sent = sent_methods(request_mock)
    assert len(sent) == 2
    for method in sent:
        assert isinstance(method, SendMessage)
        assert "автора" in (method.text or "")
