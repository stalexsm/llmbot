"""Файловое хранилище чат-сессий (JSONL, append-only).

Одна чат-сессия — один файл ``<chat_id>.jsonl`` в каталоге данных; строка
файла — одно сообщение. Хранится только диалог «пользователь ↔ финальный
ответ». Системный промпт сессии не принадлежит хранилищу: он собирается
заново при каждом запросе. Вызовы инструментов и их результаты живут только
внутри одного агентного запуска и на диск не пишутся: результат инструмента
(особенно ошибка) в истории сбивает модель на следующих ответах.
"""

import json
from pathlib import Path

import structlog

from bot.application.errors import SessionStorageError
from bot.domain.ids import TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole


class ChatSessionStore:
    """Persisted history of chat sessions, one JSONL file per chat."""

    def __init__(self, directory: Path, logger: structlog.stdlib.BoundLogger) -> None:
        self._directory = directory
        self._logger = logger.bind(component="chat_session_store")

    def load(self, chat_id: TelegramChatId) -> tuple[InferenceMessage, ...]:
        """Прочитать историю сессии: только реплики диалога, битые строки пропускаются."""
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
            if not self._is_dialog_message(message):
                # Системные и инструментальные строки (в том числе из файлов,
                # записанных до отказа от хранения tool-обмена) — не диалог.
                continue
            messages.append(message)
        return tuple(messages)

    def append(self, chat_id: TelegramChatId, *messages: InferenceMessage) -> None:
        """Дописать сообщения в конец сессии одной записью (атомарно на уровне вызова)."""
        persisted = [message for message in messages if self._is_dialog_message(message)]
        if not persisted:
            return
        payload = "".join(
            json.dumps(
                {"role": message.role.value, "content": message.content},
                ensure_ascii=False,
            )
            + "\n"
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
    def _is_dialog_message(message: InferenceMessage) -> bool:
        """Реплика ли это диалога «пользователь ↔ финальный ответ».

        Не диалог: системный промпт (собирается заново при запросе),
        результаты инструментов, assistant-сообщения с вызовами инструментов
        и пустые сообщения (в старых файлах так выглядят строки tool-вызовов).
        """
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
            return False
        return bool(message.content) and not message.tool_calls

    @staticmethod
    def _parse_line(line: str) -> InferenceMessage | None:
        try:
            record = json.loads(line)
            role = MessageRole(record["role"])
            content = record["content"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if not isinstance(content, str):
            return None
        return InferenceMessage(role=role, content=content)
