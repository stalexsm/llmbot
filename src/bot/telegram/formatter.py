"""Единый форматтер исходящих сообщений: MarkdownV2 без сбоев отправки.

Все тексты бота пользователю проходят через :func:`format_for_telegram`:
длинный текст разбивается на телеграм-совместимые части, и каждая часть
после сплита экранируется. Части уходят отдельными сообщениями, поэтому
валидный MarkdownV2 должна быть каждая часть сама по себе: незакрытые
разрывом код-блоки и код-спаны экранируются как обычный текст.
"""

from collections.abc import Iterator

from bot.telegram.splitter import TELEGRAM_MESSAGE_LIMIT, split_long_text

# Спецсимволы MarkdownV2 вне код-блоков и код-спанов
# (https://core.telegram.org/bots/api#markdownv2-style).
_MARKDOWN_V2_SPECIAL = frozenset("_*[]()~`>#+-=|{}.!\\")

# Внутри код-блоков и код-спанов экранируются только эти символы.
_CODE_SPECIAL = frozenset("\\`")

_BACKTICK = "`"
_FENCE = "```"


def escape_markdown_v2(text: str) -> str:
    """Экранировать спецсимволы MarkdownV2 обратным слешем.

    Код-блоки (``` на границе строки) и код-спаны (`` `...` `` в одной
    строке) сохраняются как код: внутри них экранируются только `` \\ ``
    и `` ` ``. Незакрытые конструкции экранируются как обычный текст —
    Telegram отвергает сообщение с незакрытым код-блоком целиком, поэтому
    битая разметка превращается в литералы, а не в ошибку отправки.
    Чистая функция: тот же вход даёт тот же выход, состояния нет.
    """
    return "".join(_escape_chunks(text))


def format_for_telegram(text: str) -> tuple[str, ...]:
    """Разбить текст на части в лимите Telegram и экранировать каждую часть.

    Экранирование применяется к каждой части после сплита и удлиняет её
    максимум вдвое (каждый символ — не более двух: сам символ и слеш).
    Часть, не влезшая в лимит после экранирования, переразбивается
    половинным лимитом: её экранированные куски гарантированно влезают.
    """
    parts: list[str] = []
    for part in split_long_text(text, TELEGRAM_MESSAGE_LIMIT):
        escaped = escape_markdown_v2(part)
        if len(escaped) <= TELEGRAM_MESSAGE_LIMIT:
            parts.append(escaped)
            continue
        parts.extend(
            escape_markdown_v2(chunk)
            for chunk in split_long_text(part, TELEGRAM_MESSAGE_LIMIT // 2)
        )
    return tuple(parts)


def _escape_chunks(text: str) -> Iterator[str]:
    """Куски экранированного текста; конкатенация кусков — результат."""
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == _BACKTICK:
            chunks, consumed = _emit_backtick(text, i)
            yield from chunks
            i += consumed
            continue
        if char in _MARKDOWN_V2_SPECIAL:
            yield "\\" + char
            i += 1
            continue
        yield char
        i += 1


def _emit_backtick(text: str, start: int) -> tuple[list[str], int]:
    """Куски для обратной кавычки в позиции ``start`` и число съеденных символов.

    Код-блок и код-спан уходят как есть (содержимое — с экранированием
    кодовых символов); одиночная или незакрывающаяся кавычка экранируется.
    """
    n = len(text)
    if text.startswith(_FENCE, start) and (start == 0 or text[start - 1] == "\n"):
        line_end = text.find("\n", start)
        body_start = n if line_end == -1 else line_end + 1
        # Код языка с кавычкой внутри — не ограждение: экранируем как текст.
        fence_without_stray = _BACKTICK not in text[start + 3 : line_end]
        close = _closing_fence(text, body_start) if fence_without_stray else None
        if close is not None:
            close_end = close + 4 if close + 3 < n else close + 3
            return (
                [
                    text[start:body_start],
                    _escape_code(text[body_start:close]),
                    text[close:close_end],
                ],
                close_end,
            )
        return ["\\`\\`\\`"], 3
    end = text.find(_BACKTICK, start + 1)
    newline = text.find("\n", start + 1)
    if end != -1 and (newline == -1 or end < newline):
        return ["`", _escape_code(text[start + 1 : end]), "`"], end + 1 - start
    return ["\\`"], 1


def _closing_fence(text: str, start: int) -> int | None:
    """Позиция закрывающего ``` — строка ровно из трёх кавычек, или ``None``."""
    pos, n = start, len(text)
    while pos < n:
        if text.startswith(_FENCE, pos) and (pos + 3 == n or text[pos + 3] == "\n"):
            return pos
        newline = text.find("\n", pos)
        if newline == -1:
            return None
        pos = newline + 1
    return None


def _escape_code(chunk: str) -> str:
    """Экранирование внутри код-блока или код-спана: только слеш и кавычка."""
    return "".join("\\" + char if char in _CODE_SPECIAL else char for char in chunk)
