"""Типизированные модели инструментов агентного цикла.

Инструмент — действие, которое модель запрашивает у харнесса по имени с
JSON-аргументами. Модели провайдеро-независимы: адаптер сериализует их в
нативный протокол инструментов (сегодня — Ollama function calling).
"""

from dataclasses import dataclass

from bot.domain.ids import ToolId


@dataclass(frozen=True)
class ToolParameter:
    """Один именованный аргумент инструмента в духе JSON Schema."""

    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ToolSpec:
    """Описание инструмента для модели: имя, назначение, схема аргументов."""

    name: ToolId
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    required: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    """Запрос модели на выполнение инструмента.

    ``arguments`` — JSON-кодированная строка с объектом аргументов, чтобы
    не протаскивать нетипизированные словари через доменные контракты.
    """

    name: ToolId
    arguments: str


@dataclass(frozen=True)
class ToolResult:
    """Результат выполнения инструмента.

    ``content`` уходит модели как содержимое tool-сообщения, ``succeeded``
    используется для отметки шага в чате пользователя.
    """

    content: str
    succeeded: bool
