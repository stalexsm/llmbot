"""Unit tests for the JSONL chat-session store (real filesystem, tmp dirs)."""

from pathlib import Path

import pytest
import structlog.stdlib

from bot.application.errors import SessionStorageError
from bot.domain.ids import TelegramChatId, ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall
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


def test_append_does_not_write_system_messages(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path, logger)

    store.append(
        CHAT,
        InferenceMessage(role=MessageRole.SYSTEM, content="system prompt"),
        InferenceMessage(role=MessageRole.USER, content="вопрос"),
    )

    # Читаем сырой файл: системное сообщение не должно быть записано вовсе.
    raw = (tmp_path / "100.jsonl").read_text(encoding="utf-8")
    assert "system prompt" not in raw
    assert '"user"' in raw


def test_tool_call_and_result_roundtrip_as_linked_pair(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    # Пара «вызов инструмента → результат»: assistant с tool_calls + tool-сообщение.
    store = make_store(tmp_path, logger)
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
            content="exit_code: 0\nstdout:\nfile.txt",
            tool_name=ToolId("exec"),
        ),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Вот файлы: file.txt"),
    )

    assert store.load(CHAT) == (
        InferenceMessage(role=MessageRole.USER, content="покажи файлы"),
        InferenceMessage(role=MessageRole.ASSISTANT, content="", tool_calls=(call,)),
        InferenceMessage(
            role=MessageRole.TOOL,
            content="exit_code: 0\nstdout:\nfile.txt",
            tool_name=ToolId("exec"),
        ),
        InferenceMessage(role=MessageRole.ASSISTANT, content="Вот файлы: file.txt"),
    )


def test_load_skips_lines_with_malformed_tool_calls(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    store = make_store(tmp_path, logger)
    session_file = tmp_path / "100.jsonl"
    session_file.write_text(
        # arguments не строка — строка битая, пропускается целиком.
        '{"role": "assistant", "content": "", "tool_calls": '
        '[{"name": "exec", "arguments": {"command": "ls"}}]}\n'
        # tool_calls не список — тоже битая строка.
        '{"role": "assistant", "content": "", "tool_calls": "oops"}\n'
        # tool_name неизвестного инструмента сохраняется как есть: валидация имён — дело хранилища.
        '{"role": "tool", "content": "ok", "tool_name": "weather"}\n'
        # Битая строка не ломает следующие валидные.
        '{"role": "user", "content": "ок"}\n',
        encoding="utf-8",
    )

    assert store.load(CHAT) == (
        InferenceMessage(role=MessageRole.TOOL, content="ok", tool_name=ToolId("weather")),
        InferenceMessage(role=MessageRole.USER, content="ок"),
    )


def test_storage_failures_map_to_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("не каталог", encoding="utf-8")
    store = ChatSessionStore(directory=blocker / "chats", logger=logger)

    with pytest.raises(SessionStorageError):
        store.append(CHAT, InferenceMessage(role=MessageRole.USER, content="вопрос"))
    with pytest.raises(SessionStorageError):
        store.reset(CHAT)


def test_load_failure_maps_to_application_error(
    tmp_path: Path, logger: structlog.stdlib.BoundLogger
) -> None:
    session_file = tmp_path / "100.jsonl"
    session_file.mkdir()  # каталог вместо файла: read_text упадёт с OSError

    store = make_store(tmp_path, logger)

    with pytest.raises(SessionStorageError):
        store.load(CHAT)
