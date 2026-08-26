"""Окно истории чат-сессии.

Чат-сессия — один длинный контекст, но провайдер получает только последние
сообщения в пределах окна. Срез смещается дальше, если окно начинается с
осиротевшего результата инструмента: пара «вызов → результат» не разрывается.
"""

from bot.domain.messages import InferenceMessage, MessageRole


def trim_to_window(
    messages: tuple[InferenceMessage, ...],
    limit: int,
) -> tuple[InferenceMessage, ...]:
    """Вернуть последние ``limit`` сообщений без осиротевших результатов инструментов."""
    if limit < 1:
        raise ValueError(f"history window limit must be >= 1, got {limit}")
    window = messages[-limit:]
    while window and window[0].role is MessageRole.TOOL:
        window = window[1:]
    return window
