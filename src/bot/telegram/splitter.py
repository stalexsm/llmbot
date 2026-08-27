"""Разбиение длинных ответов на части телеграм-совместимой длины."""

TELEGRAM_MESSAGE_LIMIT = 4000


def split_long_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> tuple[str, ...]:
    """Разбить текст на части не длиннее ``limit`` символов.

    Разрыв приходится на границы строк (или слов, если переносов нет);
    соединение частей без потерь восстанавливает исходный текст.
    """
    if len(text) <= limit:
        return (text,)
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        # Приоритет разрыва: граница строки, затем граница слова. Поиск ведётся
        # во второй половине окна, чтобы часть не вышла вырожденно короткой.
        from_half = limit // 2
        newline = window.rfind("\n", from_half)
        space = window.rfind(" ", from_half)
        cut = newline + 1 if newline != -1 else space + 1 if space != -1 else limit
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        parts.append(remaining)
    return tuple(parts)
