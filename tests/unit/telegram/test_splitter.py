"""Табличные тесты чистой функции разбиения длинных ответов."""

import pytest

from bot.telegram.splitter import split_long_text


@pytest.mark.parametrize(
    ("text", "limit"),
    [
        ("привет", 20),
        ("", 20),
        ("а" * 20, 20),  # ровно на границе
    ],
)
def test_short_text_returns_single_part(text: str, limit: int) -> None:
    assert split_long_text(text, limit) == (text,)


def test_long_text_is_cut_on_line_boundaries() -> None:
    text = "первая строка\nвторая\nтретья"  # len == 27 > 20

    parts = split_long_text(text, 20)

    assert parts == ("первая строка\n", "вторая\nтретья")


def test_line_break_exactly_at_limit_boundary() -> None:
    text = "123456789\nA"  # перенос — последний символ окна

    parts = split_long_text(text, 10)

    assert parts == ("123456789\n", "A")


def test_text_without_line_breaks_is_cut_on_word_boundaries() -> None:
    text = "aaa bbb ccc ddd"  # нет \n, только пробелы

    parts = split_long_text(text, 10)

    assert parts == ("aaa bbb ", "ccc ddd")
    assert "".join(parts) == text


def test_unsplittable_text_is_cut_hard() -> None:
    text = "abcdefghij" * 3  # ни переносов, ни пробелов

    parts = split_long_text(text, 10)

    assert parts == ("abcdefghij", "abcdefghij", "abcdefghij")


def test_default_limit_is_telegram_compatible() -> None:
    text = "строка\n" * 900  # 6300 символов

    parts = split_long_text(text)

    assert len(parts) == 2
    assert all(len(part) <= 4000 for part in parts)
    assert "".join(parts) == text
