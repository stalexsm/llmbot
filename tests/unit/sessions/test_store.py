"""Unit tests for the JSONL chat-session store (real filesystem, tmp dirs)."""

from pathlib import Path

import structlog.stdlib

from bot.domain.ids import TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.sessions.store import ChatSessionStore

CHAT = TelegramChatId(100)


def make_store(directory: Path, logger: structlog.stdlib.BoundLogger) -> ChatSessionStore:
    return ChatSessionStore(directory=directory, logger=logger)


def test_append_then_load_roundtrip(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    store = make_store(tmp_path, logger)

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
    first = make_store(tmp_path, logger)
    first.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))
    # Новый экземпляр = новый процесс: тот же каталог на диске.
    second = make_store(tmp_path, logger)

    assert second.load(CHAT) == (InferenceMessage(role=MessageRole.USER, content="вопрос"),)


def test_chats_are_isolated(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    store = make_store(tmp_path, logger)

    store.append(TelegramChatId(1), InferenceMessage(role=MessageRole.USER, content="один"))
    store.append(TelegramChatId(2), InferenceMessage(role=MessageRole.USER, content="два"))

    assert store.load(TelegramChatId(1))[0].content == "один"
    assert store.load(TelegramChatId(2))[0].content == "два"


def test_reset_empties_session(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    store = make_store(tmp_path, logger)
    store.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))

    store.reset(CHAT)

    assert store.load(CHAT) == ()


def test_reset_without_existing_file_is_noop(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path, logger)

    store.reset(CHAT)  # must not raise

    assert store.load(CHAT) == ()


def test_load_skips_malformed_lines(tmp_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    store = make_store(tmp_path, logger)
    session_file = tmp_path / "100.jsonl"
    session_file.write_text(
        '{"role": "user", "content": "ok"}\n'
        "not-json\n"
        '{"role": "wizard", "content": "неизвестная роль"}\n'
        '{"content": "нет роли"}\n',
        encoding="utf-8",
    )

    assert store.load(CHAT) == (InferenceMessage(role=MessageRole.USER, content="ok"),)


def test_load_does_not_return_system_messages(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # Системный промпт не хранится в сессии: собирается заново при запросе.
    store = make_store(tmp_path, logger)
    store.append(
        CHAT,
        InferenceMessage(role=MessageRole.SYSTEM, content="system prompt"),
        InferenceMessage(role=MessageRole.USER, content="вопрос"),
    )

    assert store.load(CHAT) == (InferenceMessage(role=MessageRole.USER, content="вопрос"),)
