"""Чистка вывода команды для модели: без ANSI-кодов и прогресс-шума.

Сырой вывод консольных утилит полон шума, который тратит контекст модели
и сбивает её: цвета и стили, управление курсором, заголовки окон, прогресс-бары,
перерисовываемые через ``\r``, спиннеры. Терминал всё это визуально схлопывает;
модель видит сырые байты. Модуль делает то, что сделал бы терминал: оставляет
только финальное состояние каждой строки.

Ограничение: перерисовка через CSI-редактирование (cursor-up + erase без ``\r``)
снимается только как коды — промежуточные текстовые кадры остаются строками.
Полная эмуляция терминала сознательно не строится.
"""

import re

# ESC-последовательности: CSI (цвета, курсор, стирание строк), OSC (заголовки
# окон) и прочие короткие последовательности с промежуточными байтами.
_ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1b
    (?:
        \[[0-?]*[ -/]*[@-~]              # CSI: \x1b[2K, \x1b[32m, …
      | \][^\x07\x1b]*(?:\x07|\x1b\\)    # OSC: \x1b]0;title\x07, …
      | [ -/]*[\x30-\x7e]                # прочие: \x1b(B, \x1bc, …
    )
    """,
    re.VERBOSE,
)

# Брайль и геометрические глифы — типичные кадры спиннеров (⠋, ◐, ◴, …).
_SPINNER_RE = re.compile("[\u2800-\u28ff\u25d0-\u25d3\u25f4-\u25f7]")

# Остаточные C0/C1-управляющие, кроме \t и \n: BEL, NUL, VT, FF, CR (защитно —
# кадры уже съедены выше), одиночный ESC, C1-диапазон.
_CONTROL_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")

# Запас под маркер обрезки плюс многозначное число пропущенных символов.
_MARKER_RESERVE = 80

# Максимальный сдвиг среза до границы строки при обрезке головы и хвоста:
# строки не рвутся посередине, но и не отъедаются целиком.
_LINE_WINDOW = 200


def _marker(skipped: int) -> str:
    return f"\n…[середина вывода обрезана: пропущено {skipped} символов]\n"


def _line_aligned_head(text: str, size: int) -> str:
    """Голова ``size`` символов, сдвинутая назад до границы строки (в пределах окна)."""
    head = text[:size]
    newline = head.rfind("\n")
    if newline != -1 and newline >= size - _LINE_WINDOW:
        return head[: newline + 1]
    return head


def _line_aligned_tail(text: str, size: int) -> str:
    """Хвост ``size`` символов, сдвинутый вперёд до начала строки (в пределах окна)."""
    tail = text[-size:] if size else ""
    newline = tail.find("\n")
    if newline != -1 and newline <= _LINE_WINDOW:
        return tail[newline + 1 :]
    return tail


def _last_progress_frame(line: str) -> str:
    """Оставить последний непустой кадр строки, перерисованной через ``\\r``."""
    if "\r" not in line:
        return line
    frames = [frame for frame in line.split("\r") if frame]
    return frames[-1] if frames else ""


def _apply_backspaces(text: str) -> str:
    """Обработать backspace как терминал: каждый ``\b`` стирает прошлый символ."""
    chars: list[str] = []
    for char in text:
        if char == "\x08":
            if chars:
                chars.pop()
        else:
            chars.append(char)
    return "".join(chars)


def clean_output(text: str) -> str:
    """Снять ANSI-коды и прогресс-шум, оставив финальное состояние строк."""
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = "\n".join(_apply_backspaces(_last_progress_frame(line)) for line in text.split("\n"))
    text = _SPINNER_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


def truncate_middle(text: str, limit: int) -> str:
    """Обрезать текст до лимита, сохранив голову и хвост; середина сжимается.

    Итоги и ошибки обычно в начале или в конце вывода команды, середина —
    наименее ценна. Модель видит голову, маркер с числом пропущенных символов
    и хвост — всё в пределах ``limit``.
    """
    if len(text) <= limit:
        return text
    half = max((limit - _MARKER_RESERVE) // 2, 0)
    head = _line_aligned_head(text, half)
    tail = _line_aligned_tail(text, half)
    skipped = len(text) - len(head) - len(tail)
    result = f"{head}{_marker(skipped)}{tail}"
    # Цифры в маркере длиннее запаса невозможны (запас покрывает десятки
    # разрядов); срез остаётся защитой от превышения лимита при любом раскладе.
    return result[-limit:] if len(result) > limit else result
