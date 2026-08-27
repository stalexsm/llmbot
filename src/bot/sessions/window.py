"""Окно истории чат-сессии.

Чат-сессия — один длинный контекст, но провайдер получает только последние
сообщения в пределах окна. История — только реплики диалога (tool-обмен
в сессии не хранится), поэтому срез по любой границе корректен.
"""

from bot.domain.messages import InferenceMessage


def trim_to_window(
    messages: tuple[InferenceMessage, ...],
    limit: int,
) -> tuple[InferenceMessage, ...]:
    """Вернуть последние ``limit`` сообщений истории."""
    if limit < 1:
        raise ValueError(f"history window limit must be >= 1, got {limit}")
    return messages[-limit:]
