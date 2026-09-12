"""Тесты единого форматтера исходящих сообщений: экранирование и лимит частей.

Экранировщик проверяется и таблично (точные строки), и по свойству
«разметка валидна»: в экранированном тексте нет неэкранированных спецсимволов
вне код-блоков и код-спанов, а код-блоки и код-спаны сбалансированы.
Валидатор в тестах устроен независимо от реализации — это внешний контракт
MarkdownV2, по которому Telegram принимает или отвергает сообщение.
"""

import pytest

from bot.telegram.formatter import escape_markdown_v2, format_for_telegram
from bot.telegram.splitter import TELEGRAM_MESSAGE_LIMIT
from tests.unit.telegram.markdown_v2 import SPECIAL_CHARS, assert_valid_markdown_v2

# Не спецсимволы: экранировать их нельзя — обратный слеш стал бы виден.
PLAIN_CHARS = "абв:;?/ «»—🚀 0\t"


@pytest.mark.parametrize("char", SPECIAL_CHARS)
def test_every_special_char_is_escaped(char: str) -> None:
    assert escape_markdown_v2(char) == "\\" + char


@pytest.mark.parametrize("char", PLAIN_CHARS)
def test_plain_chars_stay_unescaped(char: str) -> None:
    assert escape_markdown_v2(char) == char


def test_empty_text_is_empty() -> None:
    assert escape_markdown_v2("") == ""


def test_plain_text_is_unchanged() -> None:
    text = "Ответ модели: 2 из 4, путь C:/tmp готов"

    assert escape_markdown_v2(text) == text


def test_mixed_text_escapes_only_specials() -> None:
    assert escape_markdown_v2("a.b_c [x]") == r"a\.b\_c \[x\]"


def test_backslash_outside_code_is_escaped() -> None:
    assert escape_markdown_v2("C:\\путь") == "C:\\\\путь"


# --- Код-блоки: ``` на границе строки ---


def test_code_block_is_kept_raw() -> None:
    text = "```python\nprint(1). [ок]\n```"

    assert escape_markdown_v2(text) == text


@pytest.mark.parametrize(
    ("body", "escaped_body"),
    [
        ("a\\b", "a\\\\b"),  # обратный слеш внутри кода экранируется
        ("a`b", "a\\`b"),  # обратная кавычка внутри кода экранируется
        ("a.b_c", "a.b_c"),  # остальные спецсимволы внутри кода не трогаются
    ],
)
def test_code_block_escapes_only_backslash_and_backtick(body: str, escaped_body: str) -> None:
    text = f"```\n{body}\n```"

    assert escape_markdown_v2(text) == f"```\n{escaped_body}\n```"


def test_text_after_code_block_is_escaped() -> None:
    text = "```\ncode\n```\nПосле блока."

    assert escape_markdown_v2(text) == "```\ncode\n```\nПосле блока\\."


def test_unclosed_code_block_escapes_fence_and_body() -> None:
    # Незакрытый код-блок Telegram отверг бы («unclosed pre»): ограждение
    # экранируется, текст дальше уходит как обычный экранированный текст.
    assert escape_markdown_v2("```python\nx.") == "\\`\\`\\`python\nx\\."


# --- Код-спаны: `...` в одной строке ---


def test_code_span_is_kept_raw() -> None:
    assert escape_markdown_v2("используй `print(1)` тут") == "используй `print(1)` тут"


def test_code_span_escapes_only_backslash() -> None:
    assert escape_markdown_v2("`a\\b`") == "`a\\\\b`"


def test_text_around_code_span_is_escaped() -> None:
    assert escape_markdown_v2("код `x` и точка.") == "код `x` и точка\\."


def test_unclosed_code_span_escapes_backtick() -> None:
    assert escape_markdown_v2("начало `обрыв") == "начало \\`обрыв"


def test_code_span_does_not_cross_newline() -> None:
    # Код-спан обязан закрываться в той же строке: обратная кавычка перед
    # переносом — литерал, а не открытие спана.
    assert escape_markdown_v2("`a\nb`") == "\\`a\nb\\`"


# --- Свойство «разметка валидна»: независимый разбор MarkdownV2 ---


@pytest.mark.parametrize(
    "text",
    [
        "",
        "привет",
        SPECIAL_CHARS,
        SPECIAL_CHARS * 100,
        "2 * 2 = 4, 100%; путь C:/tmp, (см. пункт 3) — ок?",
        "```python\nprint(1). [ок]\n``` после.",
        "```\nн ё ` внутри\n``` хвост _с_ разметкой.",
        "формула `x = a.b` и точка. ```незакрытый",
        "строка\n\nс пустыми\nпереносами и `обрывом",
        "```вложенные ``` кавычки ``` в одной строке.",
    ],
)
def test_escaped_text_is_valid_markdown_v2(text: str) -> None:
    assert_valid_markdown_v2(escape_markdown_v2(text))


def test_escaping_is_a_pure_function() -> None:
    text = "a_b. `c` ```d"

    first = escape_markdown_v2(text)

    assert escape_markdown_v2(text) == first  # тот же вход — тот же выход


# --- Форматтер: сплит + экранирование каждой части ---


def test_format_short_text_is_single_escaped_part() -> None:
    assert format_for_telegram("Ответ _готов_.") == ("Ответ \\_готов\\_\\.",)


@pytest.mark.parametrize(
    "text",
    [
        "строка ответа\n" * 1500,  # длинный обычный текст
        "." * 10_000,  # худший случай: один спецсимвол
        SPECIAL_CHARS * 800,  # худший случай: только спецсимволы
        "```python\n" + "print(1). # " * 900 + "\n```",  # код-блок рвётся сплитом
        "_" * 4000,  # экранирование ровно удваивает часть на границе лимита
    ],
)
def test_every_part_stays_within_telegram_limit(text: str) -> None:
    # Требование тикета: после сплита и экранирования каждая часть ≤ 4096.
    # Бот использует запас (TELEGRAM_MESSAGE_LIMIT), проверяем обе границы.
    parts = format_for_telegram(text)

    assert parts
    assert all(len(part) <= TELEGRAM_MESSAGE_LIMIT for part in parts)
    assert all(len(part) <= 4096 for part in parts)


@pytest.mark.parametrize(
    "text",
    [
        "строка ответа\n" * 1500,
        "." * 10_000,
        (SPECIAL_CHARS.rstrip("\\")) * 800,  # только спецсимволы, без исходных слешей
        "```python\n" + "print(1). # " * 900 + "\n```",
    ],
)
def test_split_and_escape_lose_nothing(text: str) -> None:
    # Обратные слешы добавлены экранировщиком; без них части склеиваются
    # обратно в исходный текст — ни один символ не потерян.
    parts = format_for_telegram(text)

    assert "".join(parts).replace("\\", "") == text


@pytest.mark.parametrize(
    "text",
    [
        "строка ответа\n" * 1500,
        "." * 10_000,
        SPECIAL_CHARS * 800,
        "```python\n" + "print(1). # " * 900 + "\n```",
        "```незакрытый блок\n" + "хвост _разметки_. " * 500,
    ],
)
def test_every_part_is_valid_markdown_v2(text: str) -> None:
    # Каждая часть уходит отдельным сообщением со своим parse_mode, поэтому
    # валидна должна быть каждая часть по отдельности, а не только склейка.
    for part in format_for_telegram(text):
        assert_valid_markdown_v2(part)
