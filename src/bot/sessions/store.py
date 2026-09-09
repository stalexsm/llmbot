"""Хранилище чат-сессий в SQLite (ADR-0003).

Одна БД — один файл ``chats.db``: таблица ``messages(id, chat_id, role,
content, created_at)`` — история всех чатов. Хранится только диалог
«пользователь ↔ финальный ответ»: системный промпт собирается заново при
каждом запросе, вызовы инструментов и их результаты живут только внутри
одного агентного запуска и на диск не пишутся — результат инструмента
(особенно ошибка) в истории сбивает модель на следующих ответах.

Схемой владеет alembic (``bot.sessions.migrations``): миграции применяет
композиционный корень на старте, хранилище работает только с готовой
схемой. stdlib ``sqlite3``, синхронно, без aiosqlite; соединение
открывается на операцию и закрывается после неё. Сбой БД маппится на
прикладное ``SessionStorageError``: Telegram-слой показывает заглушку,
работа бота не рвётся. Истории старого JSONL-формата не импортируются —
это dev-данные локальной машины.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import structlog

from bot.application.errors import SessionStorageError
from bot.domain.ids import TelegramChatId
from bot.domain.messages import InferenceMessage, MessageRole


class ChatSessionStore:
    """Persisted history of chat sessions in a SQLite database.

    ``history_limit`` — размер окна истории, читаемого из БД одним запросом
    (``None`` — без ограничения); application-слой дополнительно режет окно
    чистой функцией ``window.py``.
    """

    def __init__(
        self,
        database: Path,
        logger: structlog.stdlib.BoundLogger,
        history_limit: int | None = None,
    ) -> None:
        self._database = database
        self._history_limit = history_limit
        self._logger = logger.bind(component="chat_session_store")

    def load(self, chat_id: TelegramChatId) -> tuple[InferenceMessage, ...]:
        """Прочитать историю сессии: окно последних сообщений, хронологически.

        Окно читается запросом ``ORDER BY id DESC LIMIT N`` (ADR-0003)
        и разворачивается в хронологический порядок.
        """
        query = "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC"
        chat_key = int(chat_id)
        try:
            with self._connect() as connection:
                if self._history_limit is None:
                    rows = connection.execute(query, (chat_key,)).fetchall()
                else:
                    rows = connection.execute(
                        query + " LIMIT ?", (chat_key, self._history_limit)
                    ).fetchall()
                # Материализация внутри try: битая роль в БД (вручную правленный
                # файл) — тот же сбой чтения, а не сырой ValueError наружу.
                messages = [
                    InferenceMessage(role=MessageRole(str(row[0])), content=str(row[1]))
                    for row in reversed(rows)
                ]
        except (sqlite3.Error, OSError, ValueError) as exc:
            self._logger.warning(
                "session_storage_failed",
                chat_id=chat_id,
                operation="load",
                status="error",
            )
            raise SessionStorageError("Failed to read the chat session") from exc
        return tuple(messages)

    def append(self, chat_id: TelegramChatId, *messages: InferenceMessage) -> None:
        """Дописать сообщения в сессию одной транзакцией."""
        persisted = [message for message in messages if self._is_dialog_message(message)]
        if not persisted:
            return
        try:
            with self._connect() as connection:
                connection.executemany(
                    "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                    [(int(chat_id), message.role.value, message.content) for message in persisted],
                )
        except (sqlite3.Error, OSError) as exc:
            self._logger.warning(
                "session_storage_failed",
                chat_id=chat_id,
                operation="append",
                status="error",
            )
            raise SessionStorageError("Failed to append to the chat session") from exc

    def reset(self, chat_id: TelegramChatId) -> None:
        """Обнулить сессию: DELETE строк чата (команда /new)."""
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM messages WHERE chat_id = ?", (int(chat_id),))
        except (sqlite3.Error, OSError) as exc:
            self._logger.warning(
                "session_storage_failed",
                chat_id=chat_id,
                operation="reset",
                status="error",
            )
            raise SessionStorageError("Failed to reset the chat session") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Соединение на одну операцию; на успехе — commit, на сбое — rollback."""
        self._database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _is_dialog_message(message: InferenceMessage) -> bool:
        """Реплика ли это диалога «пользователь ↔ финальный ответ».

        Не диалог: системный промпт (собирается заново при запросе),
        результаты инструментов, assistant-сообщения с вызовами инструментов
        и пустые сообщения.
        """
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
            return False
        return bool(message.content) and not message.tool_calls
