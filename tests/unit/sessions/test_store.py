"""Unit tests for the SQLite chat-session store (real filesystem, tmp dirs)."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
import structlog.stdlib

from bot.application.errors import SessionStorageError
from bot.domain.ids import TelegramChatId, ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall
from bot.sessions.migrations import apply_migrations
from bot.sessions.store import ChatSessionStore

CHAT = TelegramChatId(100)


def make_store(
    database: Path,
    logger: structlog.stdlib.BoundLogger,
    history_limit: int | None = None,
) -> ChatSessionStore:
    # Как в композиционном корне: схема — миграции alembic до первой операции.
    apply_migrations(database)
    return ChatSessionStore(database=database, history_limit=history_limit, logger=logger)


def fetch_all_rows(database: Path) -> list[tuple[object, ...]]:
    """Сырые строки таблицы сообщений: для проверок того, что лежит на диске."""
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT chat_id, role, content, created_at FROM messages ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    return list(rows)


def test_append_then_load_roundtrip(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    store = make_store(tmp_path / "chats.db", logger)

    store.append(
        CHAT,
        InferenceMessage(role=MessageRole.USER, content="Привет"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Ответ"),
    )

    assert store.load(CHAT) == (
        InferenceMessage(role=MessageRole.USER, content="Привет"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Ответ"),
    )


def test_history_survives_process_restart(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    database = tmp_path / "chats.db"
    first = make_store(database, logger)
    first.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))
    # Новый экземпляр = новый процесс: тот же файл БД на диске.
    second = make_store(database, logger)

    assert second.load(CHAT) == (InferenceMessage(role=MessageRole.USER, content="вопрос"),)


def test_chats_are_isolated(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    store = make_store(tmp_path / "chats.db", logger)

    store.append(TelegramChatId(1), InferenceMessage(role=MessageRole.USER, content="один"))
    store.append(TelegramChatId(2), InferenceMessage(role=MessageRole.USER, content="два"))

    assert store.load(TelegramChatId(1))[0].content == "один"
    assert store.load(TelegramChatId(2))[0].content == "два"


def test_window_reads_last_messages_by_descending_id(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # ADR-0003: чтение окна — ORDER BY id DESC LIMIT N; наружу — последние
    # N сообщений в хронологическом порядке.
    store = make_store(tmp_path / "chats.db", logger, history_limit=2)

    store.append(
        CHAT,
        InferenceMessage(role=MessageRole.USER, content="первый"),
        InferenceMessage(role=MessageRole.USER, content="второй"),
        InferenceMessage(role=MessageRole.USER, content="третий"),
    )

    assert store.load(CHAT) == (
        InferenceMessage(role=MessageRole.USER, content="второй"),
        InferenceMessage(role=MessageRole.USER, content="третий"),
    )


def test_load_without_limit_returns_full_history(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "chats.db", logger)
    store.append(
        CHAT,
        *(
            InferenceMessage(role=MessageRole.USER, content=f"сообщение {number}")
            for number in range(5)
        ),
    )

    assert len(store.load(CHAT)) == 5


def test_messages_have_created_at_timestamp(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # Временная метка сообщения, которой в JSONL-формате не было (ADR-0003).
    database = tmp_path / "chats.db"
    store = make_store(database, logger)
    store.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))

    rows = fetch_all_rows(database)

    assert len(rows) == 1
    created_at = rows[0][3]
    assert isinstance(created_at, str)
    datetime.fromisoformat(created_at.replace("Z", "+00:00"))  # parseable ISO timestamp


def test_reset_deletes_chat_rows(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    # /new удаляет строки чата (DELETE по chat_id): их не остаётся в БД вовсе,
    # а чужие чаты не затронуты.
    database = tmp_path / "chats.db"
    store = make_store(database, logger)
    store.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))
    store.append(TelegramChatId(200), InferenceMessage(role=MessageRole.USER, content="другой чат"))

    store.reset(CHAT)

    assert store.load(CHAT) == ()
    assert store.load(TelegramChatId(200))[0].content == "другой чат"
    assert not any(row[0] == CHAT for row in fetch_all_rows(database))


def test_reset_without_existing_database_is_noop(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path / "chats.db", logger)

    store.reset(CHAT)  # must not raise

    assert store.load(CHAT) == ()


def test_system_messages_are_not_persisted(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # Системный промпт не хранится в сессии: собирается заново при запросе.
    database = tmp_path / "chats.db"
    store = make_store(database, logger)

    store.append(
        CHAT,
        InferenceMessage(role=MessageRole.SYSTEM, content="system prompt"),
        InferenceMessage(role=MessageRole.USER, content="вопрос"),
    )

    assert store.load(CHAT) == (InferenceMessage(role=MessageRole.USER, content="вопрос"),)
    assert [row[1] for row in fetch_all_rows(database)] == ["user"]


def test_tool_exchange_is_not_persisted(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # В сессию пишется только диалог: результат инструмента (особенно ошибка)
    # в истории сбивает модель на следующих ответах.
    database = tmp_path / "chats.db"
    store = make_store(database, logger)
    call = ToolCall(name=ToolId("exec"), arguments='{"command": "ls -la"}')

    store.append(
        CHAT,
        InferenceMessage(role=MessageRole.USER, content="покажи файлы"),
        InferenceMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(call,),
        ),
        InferenceMessage(
            role=MessageRole.TOOL,
            content="exit_code: 1\nstderr:\nls: boom",
            tool_name=ToolId("exec"),
        ),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Вот файлы: file.txt"),
    )

    assert store.load(CHAT) == (
        InferenceMessage(role=MessageRole.USER, content="покажи файлы"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Вот файлы: file.txt"),
    )
    # Сырая таблица: ни tool-сообщений, ни tool_calls, ни результатов нет вовсе.
    rows = fetch_all_rows(database)
    assert [row[1] for row in rows] == ["user", "assistant"]
    assert all("boom" not in str(row[2]) for row in rows)


def test_legacy_jsonl_sessions_are_not_imported(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # Истории старого JSONL-формата — dev-данные локальной машины:
    # не импортируются и не трогаются (ADR-0003).
    legacy = tmp_path / "100.jsonl"
    legacy.write_text('{"role": "user", "content": "старый диалог"}\n', encoding="utf-8")
    store = make_store(tmp_path / "chats.db", logger)

    assert store.load(CHAT) == ()
    assert legacy.read_text(encoding="utf-8") == '{"role": "user", "content": "старый диалог"}\n'


def test_corrupt_database_maps_to_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    database = tmp_path / "chats.db"
    database.write_bytes(b"this is definitely not a sqlite database")
    # Без apply_migrations: миграция по битому файлу невозможна, хранилище
    # обязано маппить сбой на прикладное исключение на любой операции.
    store = ChatSessionStore(database=database, logger=logger)

    with pytest.raises(SessionStorageError):
        store.load(CHAT)
    with pytest.raises(SessionStorageError):
        store.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))
    with pytest.raises(SessionStorageError):
        store.reset(CHAT)


def test_unopenable_database_maps_to_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("не каталог", encoding="utf-8")
    store = ChatSessionStore(database=blocker / "chats.db", logger=logger)

    with pytest.raises(SessionStorageError):
        store.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))
    with pytest.raises(SessionStorageError):
        store.reset(CHAT)
    with pytest.raises(SessionStorageError):
        store.load(CHAT)
