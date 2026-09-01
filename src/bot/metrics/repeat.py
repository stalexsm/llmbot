"""Повторно передаваемый контекст: общий префикс двух запросов в символах.

Прямых данных кэша Ollama не существует, поэтому доля промпта, которую модель
уже видела, считается честной прокси-метрикой: списки сообщений текущего и
предыдущего запроса одного Запуска агента сравниваются по общему префиксу,
его размер берётся в символах и позже калибруется точным счётчиком токенов
промпта (см. ``RunMetricsCollector``). Функции смотрят на содержимое сообщений
только чтобы измерить длину и сравнить на равенство — наружу идут лишь числа.
"""

from collections.abc import Iterable

from bot.domain.messages import InferenceMessage


def message_chars(message: InferenceMessage) -> int:
    """Размер сообщения в символах промпта: контент плюс аргументы tool-вызовов.

    Роли, имена инструментов и структурная обвязка протокола не считаются —
    для прокси-метрики достаточно содержательной части промпта.
    """
    return len(message.content) + sum(len(call.arguments) for call in message.tool_calls)


def prompt_chars(messages: Iterable[InferenceMessage]) -> int:
    """Суммарный размер запроса в символах промпта."""
    return sum(message_chars(message) for message in messages)


def repeated_prefix_chars(
    previous: Iterable[InferenceMessage],
    current: Iterable[InferenceMessage],
) -> int:
    """Символы общего префикса двух списков сообщений.

    Префикс обрывается на первом несовпадающем сообщении: как только запросы
    разошлись, хвост предыдущего запроса моделям не переиспользуется.
    """
    total = 0
    for prev, curr in zip(previous, current, strict=False):
        if prev != curr:
            break
        total += message_chars(prev)
    return total
