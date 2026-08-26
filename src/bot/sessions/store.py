"""Файловое хранилище чат-сессий (JSONL, append-only).

Одна чат-сессия — один файл ``<chat_id>.jsonl`` в каталоге данных; строка
файла — одно сообщение. Системный промпт сессии не принадлежит хранилищу:
он собирается заново при каждом запросе и на диске не живёт.
"""

import json
from pathlib import Path

import structlog

from bot.application.errors import SessionStorageError
from bot.domain.ids import TelegramChatId, ToolId
from bot.domain.messages import InferenceMessage, MessageRole
from bot.domain.tools import ToolCall


class ChatSessionStore:
    """Persisted history of chat sessions, one JSONL file per chat."""

    def __init__(self, directory: Path, logger: structlog.stdlib.BoundLogger) -> None:
        self._directory = directory
        self._logger = logger.bind(component="chat_session_store")

    def load(self, chat_id: TelegramChatId) -> tuple[InferenceMessage, ...]:
        """Прочитать историю сессии, пропуская битые строки и системные сообщения."""
        session_file = self._file_for(chat_id)
        if not session_file.exists():
            return ()
        try:
            raw_lines = session_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            self._logger.warning("session_storage_failed", chat_id=chat_id, operation="load")
            raise SessionStorageError("Failed to read the chat session") from exc
        messages: list[InferenceMessage] = []
        for line_number, line in enumerate(raw_lines, start=1):
            if not line.strip():
                continue
            message = self._parse_line(line)
            if message is None:
                self._logger.warning(
                    "session_line_skipped",
                    chat_id=chat_id,
                    line=line_number,
                )
                continue
            if message.role is MessageRole.SYSTEM:
                # Системный промпт не хранится: собирается заново при запросе.
                continue
            messages.append(message)
        return tuple(messages)

    def append(self, chat_id: TelegramChatId, *messages: InferenceMessage) -> None:
        """Дописать сообщения в конец сессии одной записью (атомарно на уровне вызова)."""
        persisted = [message for message in messages if message.role is not MessageRole.SYSTEM]
        if not persisted:
            return
        payload = "".join(
            json.dumps(self._record_for(message), ensure_ascii=False) + "\n"
            for message in persisted
        )
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with self._file_for(chat_id).open("a", encoding="utf-8") as session_file:
                session_file.write(payload)
        except OSError as exc:
            self._logger.warning("session_storage_failed", chat_id=chat_id, operation="append")
            raise SessionStorageError("Failed to append to the chat session") from exc

    def reset(self, chat_id: TelegramChatId) -> None:
        """Обнулить сессию: история отбрасывается (команда /new)."""
        try:
            self._file_for(chat_id).write_text("", encoding="utf-8")
        except OSError as exc:
            self._logger.warning("session_storage_failed", chat_id=chat_id, operation="reset")
            raise SessionStorageError("Failed to reset the chat session") from exc

    def _file_for(self, chat_id: TelegramChatId) -> Path:
        return self._directory / f"{chat_id}.jsonl"

    @staticmethod
    def _record_for(message: InferenceMessage) -> dict[str, object]:
        record: dict[str, object] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            record["tool_calls"] = [
                {"name": call.name, "arguments": call.arguments} for call in message.tool_calls
            ]
        if message.tool_name is not None:
            record["tool_name"] = message.tool_name
        return record

    @staticmethod
    def _parse_line(line: str) -> InferenceMessage | None:
        try:
            record = json.loads(line)
            role = MessageRole(record["role"])
            content = record["content"]
            if not isinstance(content, str):
                return None
            tool_calls = ChatSessionStore._parse_tool_calls(record.get("tool_calls"))
            if tool_calls is None:
                return None
            raw_tool_name = record.get("tool_name")
            if raw_tool_name is not None and not isinstance(raw_tool_name, str):
                return None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return InferenceMessage(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_name=ToolId(raw_tool_name) if raw_tool_name is not None else None,
        )

    @staticmethod
    def _parse_tool_calls(raw: object) -> tuple[ToolCall, ...] | None:
        """Разобрать вызовы инструментов строки; ``None`` — строка битая."""
        if raw is None:
            return ()
        if not isinstance(raw, list):
            return None
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                return None
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, str):
                return None
            calls.append(ToolCall(name=ToolId(name), arguments=arguments))
        return tuple(calls)
